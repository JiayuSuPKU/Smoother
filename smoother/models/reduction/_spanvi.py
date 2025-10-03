import logging
from typing import Iterable, List, Literal, Optional, Union, Sequence
from lightning import LightningDataModule
from copy import deepcopy
from timeit import default_timer as timer
import warnings

import numpy as np
import torch
from torch.distributions import kl_divergence as kl
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.module import SCANVAE
from scvi.model import SCVI, SCANVI
from scvi.train import TrainRunner, TrainingPlan
from scvi.data._constants import _SETUP_ARGS_KEY

from anndata import AnnData
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.utils._docstrings import devices_dsp

from smoother import SpatialLoss
from ._utils import broadcast_labels, get_max_epochs_heuristic, set_params_online_update


logger = logging.getLogger(__name__)


class SPANVAE(SCANVAE):
    """Add spatial loss to the latent representation in the SCANVAE model.

    Parameters:
        spatial_loss: Spatial loss to apply on the latent representation. If None, no spatial loss.
        lambda_spatial_loss: Weight of the spatial loss.
        sp_loss_as_kl: Whether to treat the spatial loss as a KL divergence term. If True, the spatial loss will
            replace the original KL divergence term in the loss function. If False, the spatial loss will be added
            to the original loss function.
    """

    def __init__(
        self,
        n_input: int,
        spatial_loss: Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1,
        sp_loss_as_kl: bool = False,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Iterable[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "zinb",
        use_observed_lib_size: bool = True,
        y_prior: torch.Tensor | None = None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        linear_classifier: bool = False,
        classifier_parameters: dict | None = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        **vae_kwargs,

    ):
        # initialize the SCANVAE model
        super().__init__(
            n_input=n_input,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            use_observed_lib_size=use_observed_lib_size,
            y_prior=y_prior,
            labels_groups=labels_groups,
            use_labels_groups=use_labels_groups,
            linear_classifier=linear_classifier,
            classifier_parameters=classifier_parameters,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs,
        )

        # spatial loss
        self.spatial_loss = spatial_loss
        self.l_sp_loss = lambda_spatial_loss
        self.sp_loss_as_kl = sp_loss_as_kl

        # cache the diagonal of the inverse covariance matrix
        self.diag_sp_inv_cov = None
        if self.spatial_loss is not None and self.sp_loss_as_kl:
            if self.spatial_loss.use_sparse:
                self.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov_2d_sp.to_dense())
            else:
                self.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov[0])

        # need to install jax for negative binomial likelihood in scvi-tools == 1.4.0
        if gene_likelihood == "nb":
            try:
                import jax
            except ImportError:
                raise ImportError(
                    "Please install jax to use the negative binomial likelihood (scvi-tools v1.4.0)."
                )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_ouputs,
        kl_weight=1,
        labelled_tensors=None,
        classification_ratio=None,
    ):
        """Compute the loss."""
        # from the original SCANVAE loss
        # unpack hidden and latent representation
        px: Distribution = generative_ouputs["px"]
        qz1: torch.Tensor = inference_outputs["qz"]
        z1: torch.Tensor = inference_outputs["z"]
        x: torch.Tensor = tensors[REGISTRY_KEYS.X_KEY]
        batch_index: torch.Tensor = tensors[REGISTRY_KEYS.BATCH_KEY]

        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels) # n_cells*n_labels x n_latent
        qz2, z2 = self.encoder_z2_z1(z1s, ys) # n_cells*n_labels x n_latent
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        reconst_loss = -px.log_prob(x).sum(-1)

        # Compute spatial loss on qz2.loc
        if not self.sp_loss_as_kl: # spatial loss as a separate term
            if self.spatial_loss is None or self.l_sp_loss == 0:
                sp_loss = torch.tensor(0.0, device=x.device)
            else:
                # Add the spatial loss whose scale matches the original KL divergence
                z2_m = qz2.mean.view(self.n_labels, -1, z2.shape[-1]) # n_labels x n_cells x n_latent
                # Calculate spatial loss for each label and average them
                sp_loss = torch.tensor(0.0, device=x.device)
                for i in range(self.n_labels):
                    sp_loss += self.spatial_loss(z2_m[i].T, normalize = False) # a scalar
                sp_loss = self.l_sp_loss * sp_loss / z2_m.shape[-2] / self.n_labels # a scalar

        else: # Compute KLD between z2 and MVN(0, \Sigma)
            assert self.l_sp_loss > 0 and self.spatial_loss is not None, "Spatial loss must be provided when sp_loss_as_kl is True."
            z2_m = qz2.mean.view(self.n_labels, -1, z2.shape[-1]) # n_labels x n_cells x n_latent
            z2_v = qz2.variance.view(self.n_labels, -1, z2.shape[-1]) # n_labels x n_cells x n_latent
            # Calculate spatial loss for each label
            sp_loss = torch.empty(self.n_labels, device=x.device)
            for i in range(self.n_labels):
                sp_loss[i] = 0.5 * (
                    self.spatial_loss(z2_m[i].T, normalize = False) + \
                    (self.diag_sp_inv_cov.unsqueeze(-1) * z2_v[i]).sum() - \
                    torch.log(z2_v[i]).sum()
                ) # a scalar

            # Scale the spatial loss to match the original KL divergence
            sp_loss *= self.l_sp_loss / z2_m.shape[0] # a vector of length n_labels

        # KL Divergence
        if not self.sp_loss_as_kl: # original KL divergence
            mean = torch.zeros_like(qz2.loc)
            scale = torch.ones_like(qz2.scale)
            kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1) # n_cells*n_labels
        else:
            # use the spatial loss as the KL divergence
            kl_divergence_z2 = sp_loss.repeat_interleave(z2_m.shape[-2]) # n_cells*n_labels

        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)

        probs = self.classifier(z1)
        if self.classifier.logits:
            probs = F.softmax(probs, dim=-1)

        if z1.ndim == 2:
            loss_z1_unweight_ = loss_z1_unweight.view(self.n_labels, -1).t()
            kl_divergence_z2_ = kl_divergence_z2.view(self.n_labels, -1).t()
        else:
            loss_z1_unweight_ = torch.transpose(
                loss_z1_unweight.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
            kl_divergence_z2_ = torch.transpose(
                kl_divergence_z2.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
        reconst_loss += loss_z1_weight + (loss_z1_unweight_ * probs).sum(dim=-1)
        kl_divergence = (kl_divergence_z2_ * probs).sum(dim=-1)
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(
                probs=self.y_prior.repeat(probs.size(0), probs.size(1), 1)
                if len(probs.size()) == 3
                else self.y_prior.repeat(probs.size(0), 1)
            ),
        )

        if not self.use_observed_lib_size:
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence)

        kl_divergence += kl_divergence_l

        if not self.sp_loss_as_kl:
            loss = torch.mean(reconst_loss + kl_divergence * kl_weight + sp_loss)
        else:
            loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        # a payload to be used during autotune
        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
                "labels": tensors[REGISTRY_KEYS.LABELS_KEY],
            }
        else:
            extra_metrics_payload = {}

        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)

            loss += ce_loss * classification_ratio
            return LossOutput(
                loss=loss,
                reconstruction_loss=reconst_loss,
                kl_local=kl_divergence,
                kl_global=sp_loss if not self.sp_loss_as_kl else sp_loss.mean(),
                classification_loss=ce_loss,
                true_labels=true_labels,
                logits=logits,
                extra_metrics=extra_metrics_payload,
            )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_divergence,
            kl_global=sp_loss if not self.sp_loss_as_kl else sp_loss.mean(),
            extra_metrics=extra_metrics_payload,
        )

    @classmethod
    def from_vae(
        cls,
        vae_module: SCANVAE,
        spatial_loss: Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1,
        sp_loss_as_kl: bool = False,
    ):

        # copy the VAE module
        spanvae = deepcopy(vae_module)

        # switch to the spatial module
        spanvae.__class__ = cls
        spanvae.spatial_loss = spatial_loss
        spanvae.l_sp_loss = lambda_spatial_loss
        spanvae.sp_loss_as_kl = sp_loss_as_kl
        spanvae.diag_sp_inv_cov = None

        if sp_loss_as_kl:
            if spatial_loss is None or lambda_spatial_loss == 0:
                raise ValueError("spatial_loss must be provided when sp_loss_as_kl is True.")
            if spatial_loss.use_sparse:
                spanvae.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov_2d_sp.to_dense())
            else:
                spanvae.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov[0])

        return spanvae



class SpatialANVI(SCANVI):
    """Spatially-aware ANnotation Variational Inference model.
    """

    _data_splitter_cls = DeviceBackedDataSplitter
    _module_cls = SPANVAE
    _training_plan_cls = TrainingPlan # UnsupervisedTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        st_adata: AnnData,
        spatial_loss: Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1,
        sp_loss_as_kl: bool = False,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        use_observed_lib_size: bool = True,
        linear_classifier: bool = False,
        datamodule: Optional[LightningDataModule] = None,
        **model_kwargs,
    ):
        super().__init__(
            st_adata,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_observed_lib_size=use_observed_lib_size,
            linear_classifier=linear_classifier,
            datamodule=datamodule,
            **model_kwargs
        )

        self.module = self._module_cls.from_vae(
            self.module,
            spatial_loss=spatial_loss,
            lambda_spatial_loss=lambda_spatial_loss,
            sp_loss_as_kl=sp_loss_as_kl,
        )
        self.module.minified_data_type = self.minified_data_type

        self._model_summary_string = self._model_summary_string.replace('ScanVI', 'SpatialANVI')
        self.init_params_ = self._get_init_params(locals())
        self.dr_logs = {'elapsed_time': None, 'total_loss': [],
                        'recon_loss': [], 'spatial_loss': []}


    @devices_dsp.dedent
    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Unsupervised training of the spatial model without minibatches."""
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))

        logger.info(f"Training for {max_epochs} epochs.")

        plan_kwargs = {} if plan_kwargs is None else plan_kwargs

        data_splitter = self._data_splitter_cls(
            adata_manager=self.adata_manager,
            train_size=1,
            validation_size=None,
            shuffle=False,
            shuffle_set_split=False,
            batch_size=None
        )
        training_plan = self._training_plan_cls(
            self.module, **plan_kwargs
        )

        if trainer_kwargs.get('early_stopping', False):
            # TODO: implement early stopping
            # Currently even if "early_stopping_monitor" is on training data
            # 'check_val_every_n_epoch' will still be set to one
            # in the Trainer class (scvi.train._train.Trainer L129)
            raise ValueError('Early stopping is not supported for SpatialANVI as there is no validation set.')

        t_start = timer()

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        out = runner()

        t_end = timer()
        self.dr_logs['elapsed_time'] = t_end - t_start
        self.dr_logs['total_loss'] = self.trainer.logger.history['elbo_train']
        self.dr_logs['recon_loss'] = self.trainer.logger.history['reconstruction_loss_train']
        self.dr_logs['spatial_loss'] = self.trainer.logger.history['kl_global_train']

        return out

    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model: SCANVI,
        spatial_loss: Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1,
        sp_loss_as_kl: bool = False,
        unfrozen: bool = False,
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
        **spanvae_kwargs,
    ):
        """Alternate constructor for exploiting a pre-trained model on RNA-seq data.

        Note that because of the dropout layer, even though the new instance is initialized
        with the same parameters as the pre-trained model, new_instance.get_latent_representation()
        may not return the same latent representation as the pre-trained model.

        Parameters
        ----------
        st_adata
            registed anndata object
        sc_model
            pretrained SCANVI model
        """

        # rechieve the parameters from the pretrained model
        init_params = sc_model.init_params_
        non_kwargs = deepcopy(init_params["non_kwargs"])
        kwargs = deepcopy(init_params["kwargs"])
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in list({**non_kwargs, **kwargs}.items()):
            if k in spanvae_kwargs.keys():
                warnings.warn(
                    f"Ignoring param '{k}' as it was already passed in to pretrained "
                    f"SCVI model with value {v}.",
                    UserWarning
                )
                del spanvae_kwargs[k]

        # overwrite the spatial loss parameters with the new ones
        if 'spatial_loss' in non_kwargs.keys():
            warnings.warn(
                "Overwriting the spatial_loss parameter of the pretrained model. "
                f"'spatial_loss': {non_kwargs['spatial_loss']} -> {spatial_loss}. "
                f"'lambda_spatial_loss': {non_kwargs['lambda_spatial_loss']} -> {lambda_spatial_loss}. "
                f"'sp_loss_as_kl': {non_kwargs['sp_loss_as_kl']} -> {sp_loss_as_kl}.",
                UserWarning
            )
            del non_kwargs['spatial_loss']
            del non_kwargs['lambda_spatial_loss']
            del non_kwargs['sp_loss_as_kl']

        # set up the anndata object
        registry = deepcopy(sc_model.adata_manager.registry)
        scanvi_setup_args = registry[_SETUP_ARGS_KEY]

        # prepare the query anndata and load the pretrained model
        st_adata_prep = SCANVI.prepare_query_anndata(st_adata, sc_model, inplace=False)
        cls.setup_anndata(
            st_adata_prep,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **scanvi_setup_args,
        )

        # initialize the new model
        sp_model = cls(
            st_adata_prep,
            spatial_loss=spatial_loss,
            lambda_spatial_loss=lambda_spatial_loss,
            sp_loss_as_kl=sp_loss_as_kl,
            **non_kwargs, **kwargs, **spanvae_kwargs
        )

        # copy the encoder and decoder state dict
        sc_state_dict = sc_model.module.state_dict()

        # model tweaking
        new_state_dict = sp_model.module.state_dict()
        for key, load_ten in sc_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size, need to pad the pretrained load_ten
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                sc_state_dict[key] = fixed_ten

        sp_model.module.load_state_dict(sc_state_dict)
        sp_model.module.eval()

        # freeze the pretrained model if neccessary
        set_params_online_update(
            sp_model.module,
            unfrozen=unfrozen,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_classifier=freeze_classifier,
        )
        sp_model.is_trained_ = False

        return sp_model
