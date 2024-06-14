from typing import List, Literal, Optional, Union
from copy import deepcopy
from timeit import default_timer as timer
import warnings
import inspect

import torch
from torch.distributions import kl_divergence as kld
from torch.distributions import Normal
from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.module import MULTIVAE
from scvi.model import MULTIVI

from anndata import AnnData
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.utils._docstrings import devices_dsp
from scvi.data._constants import _SETUP_ARGS_KEY

from smoother import SpatialLoss
from ._utils import set_params_online_update

class SPMULTIVAE(MULTIVAE):
	"""Add spatial loss to the latent representation in the MULTIVAE model.

	Parameters:
		spatial_loss: Spatial loss to apply on the latent representation. If None, no spatial loss.
		lambda_spatial_loss: Weight of the spatial loss.
		sp_loss_as_kl: Whether to apply the spatial loss as a KL term (i.e. replacing the original Normal(0, 1) prior)
			or as a separate global term.
	"""

	def __init__(
		self,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss: float = 1,
		sp_loss_as_kl: bool = True,
		**model_kwargs
	):
		# initialize the MULTIVAE model
		super().__init__(**model_kwargs)

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

	def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
		"""Computes the loss function for the model."""

		if (self.l_sp_loss == 0.0) or (self.spatial_loss is None):
			# no spatial loss, regular MULTIVAE
			return super().loss(tensors, inference_outputs, generative_outputs, kl_weight)

		# Get the data
		x = tensors[REGISTRY_KEYS.X_KEY]

		# TODO: CHECK IF THIS FAILS IN ONLY RNA DATA
		x_rna = x[:, : self.n_input_genes]
		x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
		if self.n_input_proteins == 0:
			y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
		else:
			y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

		mask_expr = x_rna.sum(dim=1) > 0
		mask_acc = x_chr.sum(dim=1) > 0
		mask_pro = y.sum(dim=1) > 0

		# Compute Accessibility loss
		p = generative_outputs["p"]
		libsize_acc = inference_outputs["libsize_acc"]
		rl_accessibility = self.get_reconstruction_loss_accessibility(x_chr, p, libsize_acc)

		# Compute Expression loss
		px_rate = generative_outputs["px_rate"]
		px_r = generative_outputs["px_r"]
		px_dropout = generative_outputs["px_dropout"]
		x_expression = x[:, : self.n_input_genes]
		rl_expression = self.get_reconstruction_loss_expression(
			x_expression, px_rate, px_r, px_dropout
		)

		# Compute Protein loss - No ability to mask minibatch (Param:None)
		if mask_pro.sum().gt(0):
			py_ = generative_outputs["py_"]
			rl_protein = get_reconstruction_loss_protein(y, py_, None)
		else:
			rl_protein = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

		# calling without weights makes this act like a masked sum
		# TODO : CHECK MIXING HERE
		recon_loss_expression = rl_expression * mask_expr
		recon_loss_accessibility = rl_accessibility * mask_acc
		recon_loss_protein = rl_protein * mask_pro
		recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_protein

		# Compute KL Divergence
		if not self.sp_loss_as_kl: # spatial loss as a separate term
			# Compute KLD between Z and N(0,I)
			qz_m = inference_outputs["qz_m"]
			qz_v = inference_outputs["qz_v"]
			kl_div_z = kld(
				Normal(qz_m, torch.sqrt(qz_v)),
				Normal(0, 1),
			).sum(dim=1) # a vector of size n_spots

			# Add the spatial loss whose scale matches the original KL divergence
			sp_loss = self.spatial_loss(qz_m.T, normalize = False)
			sp_loss *= self.l_sp_loss / qz_m.shape[0] # a scalar

		else:
			# Compute KLD between Z and MVN(0, \Sigma)
			qz_m = inference_outputs["qz_m"] # n_spots x n_latent
			qz_v = inference_outputs["qz_v"] # n_spots x n_latent
			kl_div_z = 0.5 * (
				self.spatial_loss(qz_m.T, normalize = False) + \
				(self.diag_sp_inv_cov.unsqueeze(-1) * qz_v).sum() - \
				torch.log(qz_v).sum()
			) # a scalar

			# Scale the spatial loss to match the original KL divergence
			kl_div_z *= self.l_sp_loss / qz_m.shape[0]
			sp_loss = kl_div_z

		# Compute KLD between distributions for paired data
		kl_div_paired = self._compute_mod_penalty(
			(inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
			(inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
			(inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
			mask_expr,
			mask_acc,
			mask_pro,
		)

		# KL WARMUP
		kl_local_for_warmup = kl_div_z
		weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired

		# TOTAL LOSS
		loss = torch.mean(recon_loss + weighted_kl_local)
		if not self.sp_loss_as_kl:
			loss += sp_loss

		recon_losses = {
			"reconstruction_loss_expression": recon_loss_expression,
			"reconstruction_loss_accessibility": recon_loss_accessibility,
			"reconstruction_loss_protein": recon_loss_protein,
		}
		kl_local = {
			"kl_divergence_z": kl_div_z,
			"kl_divergence_paired": kl_div_paired,
		}

		return LossOutput(loss=loss, reconstruction_loss=recon_losses, kl_local=kl_local, kl_global=sp_loss)

	@classmethod
	def from_vae(
		cls,
		vae_module: MULTIVAE,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss: float = 1,
		sp_loss_as_kl: bool = True,
	):

		# copy the VAE module
		spmultivae = deepcopy(vae_module)

		# switch to the spatial module
		spmultivae.__class__ = cls
		spmultivae.spatial_loss = spatial_loss
		spmultivae.l_sp_loss = lambda_spatial_loss
		spmultivae.sp_loss_as_kl = sp_loss_as_kl

		# cache the diagonal of the inverse covariance matrix
		spmultivae.diag_sp_inv_cov = None
		if spatial_loss is not None and sp_loss_as_kl:
			if spatial_loss.use_sparse:
				spmultivae.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov_2d_sp.to_dense())
			else:
				spmultivae.diag_sp_inv_cov = torch.diagonal(spatial_loss.inv_cov[0])

		return spmultivae


class SpatialMULTIVI(MULTIVI):
	"""Spatially-aware Multimodal Variational Autoencoder model.
	"""
	_data_splitter_cls = DeviceBackedDataSplitter
	_module_cls = SPMULTIVAE

	def __init__(
		self,
		st_adata: AnnData,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss: float = 1,
		sp_loss_as_kl: bool = True,
		**model_kwargs,
	):
		super().__init__(
			st_adata,
			**model_kwargs,
		)

		self.module = self._module_cls.from_vae(
			self.module,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_as_kl=sp_loss_as_kl,
		)

		self._model_summary_string = self._model_summary_string.replace('MULTIVI', 'SpatialMULTIVI')
		self.init_params_ = self._get_init_params(locals())
		self.dr_logs = {'elapsed_time': None, 'total_loss': [],
						'recon_loss': [], 'spatial_loss': []}

	@devices_dsp.dedent
	def train(
		self,
		max_epochs: int = 500,
		lr: float = 1e-4,
		accelerator: str = "auto",
		devices = "auto",
		weight_decay: float = 1e-3,
		eps: float = 1e-08,
		n_steps_kl_warmup = None,
		n_epochs_kl_warmup = 50,
		adversarial_mixing = False, # originally True
		datasplitter_kwargs = None,
		plan_kwargs = None,
		**kwargs,
	):
		"""Train the model without mini-batch."""
		t_start = timer()

		# check if any of the params are already passed in
		for k, v in list({**kwargs}.items()):
			if k in [
				'train_size', 'validation_size', 'shuffle_set_split', 'batch_size',
				'early_stopping', 'save_best', 'check_val_every_n_epoch'
			]:
				warnings.warn(
					f"Ignoring param '{k}' as there will be no validation set.",
					UserWarning
				)
				del kwargs[k]

		# fit the model with all spots (no mini-batch)
		train_size = 1
		validation_size = None
		shuffle_set_split = False
		early_stopping = False
		save_best = False
		batch_size = None
		check_val_every_n_epoch = None

		super().train(
			max_epochs = max_epochs,
			lr = lr,
			accelerator = accelerator,
			devices = devices,
			train_size = train_size,
			validation_size = validation_size,
			shuffle_set_split = shuffle_set_split,
			batch_size = batch_size,
			weight_decay = weight_decay,
			eps = eps,
			early_stopping = early_stopping,
			save_best = save_best,
			check_val_every_n_epoch = check_val_every_n_epoch,
			n_steps_kl_warmup = n_steps_kl_warmup,
			n_epochs_kl_warmup = n_epochs_kl_warmup,
			adversarial_mixing 	= adversarial_mixing,
			datasplitter_kwargs = datasplitter_kwargs,
			plan_kwargs = plan_kwargs,
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
		sc_model: MULTIVI,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss: float = 1,
		sp_loss_as_kl: bool = True,
		unfrozen: bool = False,
		freeze_dropout: bool = False,
		freeze_expression: bool = True,
		freeze_decoder_first_layer: bool = True,
		freeze_batchnorm_encoder: bool = True,
		freeze_batchnorm_decoder: bool = False,
		freeze_classifier: bool = True,
		**spmultivae_kwargs,
	):
		"""Alternate constructor for exploiting a pre-trained model on non-spatial data.

		Note that because of the dropout layer, even though the new instance is initialized
		with the same parameters as the pre-trained model, new_instance.get_latent_representation()
		may not return the same latent representation as the pre-trained model.

		Parameters
		----------
		st_adata
			registed anndata object
		sc_model
			pretrained MULTIVI model
		"""
		# rechieve the parameters from the pretrained model
		init_params = sc_model.init_params_
		non_kwargs = deepcopy(init_params["non_kwargs"])
		kwargs = deepcopy(init_params["kwargs"])
		kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
		for k, v in list({**non_kwargs, **kwargs}.items()):
			if k in spmultivae_kwargs.keys():
				warnings.warn(
					f"Ignoring param '{k}' as it was already passed in to pretrained "
					f"MULTIVI model with value {v}.",
					UserWarning
				)
				del spmultivae_kwargs[k]

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
		multivi_setup_args = registry[_SETUP_ARGS_KEY]
		valid_setup_args = inspect.getfullargspec(cls.setup_anndata).args
		for k in list(multivi_setup_args.keys()):
			if k not in valid_setup_args:
				warnings.warn(
					f"Argument '{k}' in the pretrained model is not valid for {cls.__name__}.setup_anndata()."
					" Will be ignored.",
					UserWarning
				)
				del multivi_setup_args[k]

		# prepare the query anndata
		st_adata_prep = MULTIVI.prepare_query_anndata(st_adata, sc_model, inplace=False)
		cls.setup_anndata(
			st_adata_prep,
			source_registry=registry,
			extend_categories=True,
			allow_missing_labels=True,
			**multivi_setup_args,
		)

		# initialize the new model
		sp_model = cls(
			st_adata_prep,
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_as_kl=sp_loss_as_kl,
			**non_kwargs, **kwargs, **spmultivae_kwargs
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
