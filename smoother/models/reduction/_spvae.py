from typing import List, Literal, Optional, Union
from copy import deepcopy
from timeit import default_timer as timer
import warnings
import inspect

import torch
from torch.distributions import kl_divergence as kl
from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.module import VAE
from scvi.model import SCVI

from anndata import AnnData
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.utils._docstrings import devices_dsp
from scvi.data._constants import _SETUP_ARGS_KEY

from smoother import SpatialLoss

class SPVAE(VAE):
	"""Add spatial loss to the latent representation in the VAE model.

	Parameters:
		spatial_loss: Spatial loss to apply on the latent representation. If None, no spatial loss.
		lambda_spatial_loss: Weight of the spatial loss.
		sp_loss_on: Whether to apply the spatial loss on the sampled latent representation ('z') or
			on the encoded mean of the latent distribution ('mean_z').
	"""

	def __init__(
		self,
		n_input: int,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z',
		n_hidden: int = 128,
		n_latent: int = 10,
		n_layers: int = 1,
		dropout_rate: float = 0.1,
		dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
		gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
		latent_distribution: Literal["normal", "ln"] = "normal",
		**model_kwargs
	):
		# initialize the VAE model
		super().__init__(
			n_input=n_input,
			n_hidden=n_hidden,
			n_latent=n_latent,
			n_layers=n_layers,
			dropout_rate=dropout_rate,
			dispersion=dispersion,
			gene_likelihood=gene_likelihood,
			latent_distribution=latent_distribution,
			**model_kwargs
		)

		# spatial loss
		self.spatial_loss = spatial_loss
		self.l_sp_loss = lambda_spatial_loss
		self.sp_loss_on = sp_loss_on

	def loss(
		self,
		tensors,
		inference_outputs,
		generative_outputs,
		kl_weight: float = 1.0,
	):
		# calculate the original VAE loss
		x = tensors[REGISTRY_KEYS.X_KEY]
		kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
			dim=-1
		)
		if not self.use_observed_lib_size:
			kl_divergence_l = kl(
				inference_outputs["ql"],
				generative_outputs["pl"],
			).sum(dim=1)
		else:
			kl_divergence_l = torch.tensor(0.0, device=x.device)

		reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

		kl_local_for_warmup = kl_divergence_z
		kl_local_no_warmup = kl_divergence_l

		weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

		# add the spatial loss
		if self.spatial_loss is not None and self.l_sp_loss > 0.0:
			if self.sp_loss_on == 'z':
				sp_loss = self.spatial_loss(inference_outputs['z'].T, normalize = False)
			elif self.sp_loss_on == 'mean_z':
				sp_loss = self.spatial_loss(inference_outputs['qz'].loc.T, normalize = False)
			else:
				raise NotImplementedError('Currently the spatial loss can only be applied on z or mean_z.')

			sp_loss *= self.l_sp_loss / self.n_latent
		else:
			sp_loss = torch.tensor(0.0)

		loss = torch.mean(reconst_loss + weighted_kl_local + sp_loss)

		kl_local = {
			"kl_divergence_l": kl_divergence_l,
			"kl_divergence_z": kl_divergence_z,
		}
		return LossOutput(
			loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local, kl_global=sp_loss
		)


class SpatialVAE(SCVI):
	"""Spatially-aware Variational Autoencoder model.
	"""

	_data_splitter_cls = DeviceBackedDataSplitter
	_module_cls = SPVAE

	def __init__(
		self,
		st_adata: AnnData,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z',
		n_hidden: int = 128,
		n_latent: int = 10,
		n_layers: int = 1,
		dropout_rate: float = 0.0,
		dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
		gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
		latent_distribution: Literal["normal", "ln"] = "normal",
		**model_kwargs,
	):
		super().__init__(st_adata)
		self.n_genes = self.summary_stats.n_vars

		self.module = self._module_cls(
			n_input=self.n_genes,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_on=sp_loss_on,
			n_hidden=n_hidden,
			n_latent=n_latent,
			n_layers=n_layers,
			dropout_rate=dropout_rate,
			dispersion=dispersion,
			gene_likelihood=gene_likelihood,
			latent_distribution=latent_distribution,
			**model_kwargs,
		)
		self._model_summary_string = self._model_summary_string.replace('SCVI', 'SpatialVAE')
		self.init_params_ = self._get_init_params(locals())
		self.dr_logs = {'elapsed_time': None, 'total_loss': [],
						'recon_loss': [], 'spatial_loss': []}

	@devices_dsp.dedent
	def train(
		self,
		max_epochs: int = 400,
		lr: float = 0.01,
		use_gpu: Optional[Union[str, int, bool]] = None,
		accelerator: str = "auto",
		devices: Union[int, List[int], str] = "auto",
		plan_kwargs: Optional[dict] = None,
		**kwargs,
	):
		"""Trains the model without mini-batch.
		"""
		update_dict = {
			"lr": lr,
		}
		if plan_kwargs is not None:
			plan_kwargs.update(update_dict)
		else:
			plan_kwargs = update_dict

		t_start = timer()

		# fit the model with all spots (no mini-batch)
		super().train(
			max_epochs=max_epochs,
			use_gpu=use_gpu,
			accelerator=accelerator,
			devices=devices,
			train_size=1,
			validation_size=None,
			shuffle_set_split=False,
			batch_size=None,
			plan_kwargs=plan_kwargs,
			**kwargs,
		)
		t_end = timer()
		self.dr_logs['elapsed_time'] = t_end - t_start
		self.dr_logs['total_loss'] = self.trainer.logger.history['elbo_train']
		self.dr_logs['recon_loss'] = self.trainer.logger.history['reconstruction_loss_train']
		self.dr_logs['spatial_loss'] = self.trainer.logger.history['kl_global_train']


	@classmethod
	def from_rna_model(
		cls,
		st_adata: AnnData,
		sc_model: SCVI,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z',
		**spvae_kwargs,
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
			pretrained SCVI model
		"""
		# rechieve the parameters from the pretrained model
		init_params = sc_model.init_params_
		non_kwargs = deepcopy(init_params["non_kwargs"])
		kwargs = deepcopy(init_params["kwargs"])
		kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
		for k, v in list({**non_kwargs, **kwargs}.items()):
			if k in spvae_kwargs.keys():
				warnings.warn(
					f"Ignoring param '{k}' as it was already passed in to pretrained "
					f"SCVI model with value {v}.",
					UserWarning
				)
				del spvae_kwargs[k]

		# overwrite the spatial loss parameters with the new ones
		if 'spatial_loss' in non_kwargs.keys():
			warnings.warn(
				"Overwriting the spatial_loss parameter of the pretrained model. "
				f"'spatial_loss': {non_kwargs['spatial_loss']} -> {spatial_loss}. "
				f"'lambda_spatial_loss': {non_kwargs['lambda_spatial_loss']} -> {lambda_spatial_loss}. "
				f"'sp_loss_on': {non_kwargs['sp_loss_on']} -> {sp_loss_on}.",
				UserWarning
			)
			del non_kwargs['spatial_loss']
			del non_kwargs['lambda_spatial_loss']
			del non_kwargs['sp_loss_on']

		# set up the anndata object
		scvi_setup_args = deepcopy(sc_model.adata_manager.registry[_SETUP_ARGS_KEY])
		valid_setup_args = inspect.getfullargspec(cls.setup_anndata).args
		for k in list(scvi_setup_args.keys()):
			if k not in valid_setup_args:
				warnings.warn(
					f"Argument '{k}' in the pretrained model is not valid for {cls.__name__}.setup_anndata()."
					" Will be ignored.",
					UserWarning
				)
				del scvi_setup_args[k]

		cls.setup_anndata(
			st_adata,
			**scvi_setup_args
		)

		# initialize the new instance
		sp_model = cls(
			st_adata,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_on=sp_loss_on,
			**non_kwargs, **kwargs, **spvae_kwargs
		)

		# copy the encoder and decoder state dict
		scvi_state_dict = sc_model.module.state_dict()
		sp_model.module.load_state_dict(scvi_state_dict, strict=False)

		return sp_model


