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
from ._utils import set_params_online_update

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

	@classmethod
	def from_vae(
		cls,
		vae_module: VAE,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z'
	):

		# copy the VAE module
		spvae = deepcopy(vae_module)

		# switch to the spatial module
		spvae.__class__ = cls
		spvae.spatial_loss = spatial_loss
		spvae.l_sp_loss = lambda_spatial_loss
		spvae.sp_loss_on = sp_loss_on

		return spvae


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
		super().__init__(
			st_adata,
			n_hidden=n_hidden,
			n_latent=n_latent,
			n_layers=n_layers,
			dropout_rate=dropout_rate,
			dispersion=dispersion,
			gene_likelihood=gene_likelihood,
			latent_distribution=latent_distribution,
			**model_kwargs,
		)

		self.module = self._module_cls.from_vae(
			self.module,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_on=sp_loss_on
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

		if kwargs.get('early_stopping', False):
			# TODO: implement early stopping
			# Currently even if "early_stopping_monitor" is on training data
			# 'check_val_every_n_epoch' will still be set to one
			# in the Trainer class (scvi.train._train.Trainer L129)
			raise ValueError('Early stopping is not supported for SpatialVAE as there is no validation set.')

		# fit the model with all spots (no mini-batch)
		super().train(
			max_epochs=max_epochs,
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
		unfrozen: bool = False,
		freeze_dropout: bool = False,
		freeze_expression: bool = True,
		freeze_decoder_first_layer: bool = True,
		freeze_batchnorm_encoder: bool = True,
		freeze_batchnorm_decoder: bool = False,
		freeze_classifier: bool = True,
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
		registry = deepcopy(sc_model.adata_manager.registry)
		scvi_setup_args = registry[_SETUP_ARGS_KEY]
		valid_setup_args = inspect.getfullargspec(cls.setup_anndata).args
		for k in list(scvi_setup_args.keys()):
			if k not in valid_setup_args:
				warnings.warn(
					f"Argument '{k}' in the pretrained model is not valid for {cls.__name__}.setup_anndata()."
					" Will be ignored.",
					UserWarning
				)
				del scvi_setup_args[k]

		# prepare the query anndata
		st_adata_prep = SCVI.prepare_query_anndata(st_adata, sc_model, inplace=False)
		cls.setup_anndata(
			st_adata_prep,
			source_registry=registry,
			extend_categories=True,
			allow_missing_labels=True,
			**scvi_setup_args,
		)

		# initialize the new model
		sp_model = cls(
			st_adata_prep,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_on=sp_loss_on,
			**non_kwargs, **kwargs, **spvae_kwargs
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
