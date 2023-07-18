import logging
from typing import Iterable, List, Literal, Optional, Union, Sequence
from copy import deepcopy
import warnings

import numpy as np
import torch
from torch.distributions import kl_divergence as kl
from torch.distributions import Categorical, Normal

from scvi import REGISTRY_KEYS, settings
from scvi.nn import one_hot
from scvi.autotune._types import Tunable
from scvi.module.base import LossOutput
from scvi.module import SCANVAE
from scvi.model import SCVI, SCANVI
from scvi.train import TrainRunner, TrainingPlan

from anndata import AnnData
from scvi.dataloaders import DeviceBackedDataSplitter
from scvi.utils._docstrings import devices_dsp

from smoother import SpatialLoss

logger = logging.getLogger(__name__)

# SCVI.module.base._utils.py
def iterate(obj, func):
	"""Iterates over an object and applies a function to each element."""
	t = type(obj)
	if t is list or t is tuple:
		return t([iterate(o, func) for o in obj])
	else:
		return func(obj) if obj is not None else None

def broadcast_labels(y, *o, n_broadcast=-1):
	"""Utility for the semi-supervised setting.

	If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
	If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other arguments if not None)
	"""
	if not len(o):
		raise ValueError("Broadcast must have at least one reference argument")
	if y is None:
		ys = enumerate_discrete(o[0], n_broadcast)
		new_o = iterate(
			o,
			lambda x: x.repeat(n_broadcast, 1)
			if len(x.size()) == 2
			else x.repeat(n_broadcast),
		)
	else:
		ys = one_hot(y, n_broadcast)
		new_o = o
	return (ys,) + new_o

def enumerate_discrete(x, y_dim):
	"""Enumerate discrete variables."""

	def batch(batch_size, label):
		labels = torch.ones(batch_size, 1, device=x.device, dtype=torch.long) * label
		return one_hot(labels, y_dim)

	batch_size = x.size(0)
	return torch.cat([batch(batch_size, i) for i in range(y_dim)])

def get_max_epochs_heuristic(
	n_obs: int, epochs_cap: int = 400, decay_at_n_obs: int = 20000
) -> int:
	"""Compute a heuristic for the default number of maximum epochs.

	If `n_obs <= decay_at_n_obs`, the number of maximum epochs is set to
	`epochs_cap`. Otherwise, the number of maximum epochs decays according to
	`(decay_at_n_obs / n_obs) * epochs_cap`, with a minimum of 1.

	Parameters
	----------
	n_obs
		The number of observations in the dataset.
	epochs_cap
		The maximum number of epochs for the heuristic.
	decay_at_n_obs
		The number of observations at which the heuristic starts decaying.

	Returns
	-------
	`int`
		A heuristic for the default number of maximum epochs.
	"""
	max_epochs = min(round((decay_at_n_obs / n_obs) * epochs_cap), epochs_cap)
	max_epochs = max(max_epochs, 1)

	if max_epochs == 1:
		warnings.warn(
			"The default number of maximum epochs has been set to 1 due to the large"
			"number of observations. Pass in `max_epochs` to the `train` function in "
			"order to override this behavior.",
			UserWarning,
			stacklevel=settings.warnings_stacklevel,
		)

	return max_epochs


class SPANVAE(SCANVAE):
	"""Add spatial loss to the latent representation in the SCANVAE model.

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
		n_batch: int = 0,
		n_labels: int = 0,
		n_hidden: Tunable[int] = 128,
		n_latent: Tunable[int] = 10,
		n_layers: Tunable[int] = 1,
		n_continuous_cov: int = 0,
		n_cats_per_cov: Optional[Iterable[int]] = None,
		dropout_rate: Tunable[float] = 0.1,
		dispersion: Tunable[
			Literal["gene", "gene-batch", "gene-label", "gene-cell"]
		] = "gene",
		log_variational: Tunable[bool] = True,
		gene_likelihood: Tunable[Literal["zinb", "nb"]] = "zinb",
		y_prior=None,
		labels_groups: Sequence[int] = None,
		use_labels_groups: bool = False,
		linear_classifier: bool = False,
		classifier_parameters: Optional[dict] = None,
		use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
		use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
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
		self.sp_loss_on = sp_loss_on

	def loss(
		self,
		tensors,
		inference_outputs,
		generative_ouputs,
		feed_labels=False,
		kl_weight=1,
		labelled_tensors=None,
		classification_ratio=None,
	):
		"""Compute the loss."""
		# from the original SCANVAE loss
		# unpack hidden and latent representation
		px = generative_ouputs["px"]
		qz1 = inference_outputs["qz"]
		z1 = inference_outputs["z"]
		x = tensors[REGISTRY_KEYS.X_KEY]
		batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

		if feed_labels:
			y = tensors[REGISTRY_KEYS.LABELS_KEY]
		else:
			y = None
		is_labelled = False if y is None else True

		# Enumerate choices of label
		ys, z1s = broadcast_labels(y, z1, n_broadcast=self.n_labels)
		qz2, z2 = self.encoder_z2_z1(z1s, ys)
		pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
		reconst_loss = -px.log_prob(x).sum(-1)

		# KL Divergence
		mean = torch.zeros_like(qz2.loc)
		scale = torch.ones_like(qz2.scale)

		kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=1)
		loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
		loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)
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
			kl_divergence_l = 0.0

		# add the spatial loss
		if self.spatial_loss is not None and self.l_sp_loss > 0.0:
			if self.sp_loss_on == 'z':
				sp_loss = self.spatial_loss(z1.T, normalize = False)
			elif self.sp_loss_on == 'mean_z':
				sp_loss = self.spatial_loss(qz1.loc.T, normalize = False)
			else:
				raise NotImplementedError('Currently the spatial loss can only be applied on z or mean_z.')

			sp_loss *= self.l_sp_loss / self.n_latent
		else:
			sp_loss = torch.tensor(0.0)

		if is_labelled:
			loss = reconst_loss + loss_z1_weight + loss_z1_unweight + sp_loss
			kl_locals = {
				"kl_divergence_z2": kl_divergence_z2,
				"kl_divergence_l": kl_divergence_l,
			}
			if labelled_tensors is not None:
				ce_loss, true_labels, logits = self.classification_loss(
					labelled_tensors
				)
				loss += ce_loss * classification_ratio
				return LossOutput(
					loss=loss,
					reconstruction_loss=reconst_loss,
					kl_local=kl_locals,
					kl_global=sp_loss,
					classification_loss=ce_loss,
					true_labels=true_labels,
					logits=logits,
					extra_metrics={
						"n_labelled_tensors": labelled_tensors[
							REGISTRY_KEYS.X_KEY
						].shape[0],
					},
				)
			return LossOutput(
				loss=loss,
				reconstruction_loss=reconst_loss,
				kl_local=kl_locals,
				kl_global=sp_loss,
			)

		probs = self.classifier(z1)
		reconst_loss += loss_z1_weight + (
			(loss_z1_unweight).view(self.n_labels, -1).t() * probs
		).sum(dim=1)

		kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(
			dim=1
		)
		kl_divergence += kl(
			Categorical(probs=probs),
			Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
		)
		kl_divergence += kl_divergence_l

		loss = torch.mean(reconst_loss + kl_divergence * kl_weight) + sp_loss

		if labelled_tensors is not None:
			ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)

			loss += ce_loss * classification_ratio
			return LossOutput(
				loss=loss,
				reconstruction_loss=reconst_loss,
				kl_local=kl_divergence,
				kl_global=sp_loss,
				classification_loss=ce_loss,
				true_labels=true_labels,
				logits=logits,
			)
		return LossOutput(
			loss=loss, reconstruction_loss=reconst_loss,
			kl_local=kl_divergence,
			kl_global=sp_loss,
		)


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
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z',
		n_hidden: int = 128,
		n_latent: int = 10,
		n_layers: int = 1,
		dropout_rate: float = 0.1,
		dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
		gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
		latent_distribution: Literal["normal", "ln"] = "normal",
		**model_kwargs,
	):
		super().__init__(st_adata)
		self.n_genes = self.summary_stats.n_vars

		self.module = SpatialANVI._module_cls(
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
		self._model_summary_string = self._model_summary_string.replace('ScanVI', 'SpatialANVI')
		self.init_params_ = self._get_init_params(locals())

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

		runner = TrainRunner(
			self,
			training_plan=training_plan,
			data_splitter=data_splitter,
			max_epochs=max_epochs,
			accelerator=accelerator,
			devices=devices,
			**trainer_kwargs,
		)
		return runner()

	@classmethod
	def from_rna_model(
		cls,
		st_adata: AnnData,
		sc_model: SCANVI,
		spatial_loss: Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		sp_loss_on: Literal['z', 'mean_z'] = 'mean_z',
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
			pretrained SCVI model
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
				f"'sp_loss_on': {non_kwargs['sp_loss_on']} -> {sp_loss_on}.",
				UserWarning
			)
			del non_kwargs['spatial_loss']
			del non_kwargs['lambda_spatial_loss']
			del non_kwargs['sp_loss_on']

		# prepare the query anndata and load the pretrained model
		st_adata_prep = SCANVI.prepare_query_anndata(st_adata, sc_model, inplace=False)
		sp_model = SCANVI.load_query_data(st_adata_prep, sc_model)

		# change to the SpatialANVI class
		sp_model.__class__ = cls
		sp_model.n_genes = sp_model.summary_stats.n_vars
		sp_model._model_summary_string = sp_model._model_summary_string.replace('ScanVI', 'SpatialANVI')
		sp_model.init_params_['non_kwargs'].update({'spatial_loss': spatial_loss,
													'lambda_spatial_loss': lambda_spatial_loss,
													'sp_loss_on': sp_loss_on})

		# switch to the spatial module
		sp_model.module = SpatialANVI._module_cls(
			spatial_loss=spatial_loss,
			lambda_spatial_loss=lambda_spatial_loss,
			sp_loss_on=sp_loss_on,
			n_input=sp_model.n_genes,
			n_batch=sp_model.summary_stats.n_batch,
			n_labels=sp_model.summary_stats.n_labels - 1, # ignores unlabeled catgegory
			n_continuous_cov=sp_model.summary_stats.get("n_extra_continuous_covs", 0),
			**kwargs,
			**non_kwargs,
			**spanvae_kwargs,
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

		return sp_model
