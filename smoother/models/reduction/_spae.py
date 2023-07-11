from typing import List, Literal, Optional, Union
from copy import deepcopy
from timeit import default_timer as timer
import warnings
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from anndata import AnnData
from scvi.nn import FCLayers

from smoother.losses import SpatialLoss


class AutoEncoderClass(nn.Module):
	"""Abstract AutoEncoder class.

	To mimic the behavior of PCA, the overall objective contains three losses:
		1. Reconstruction loss: reconstruction error of the observed data.
		2. Orthogonality loss: the loss of orthogonality on the hidden embeddings.
		3. Spatial loss: the spatial loss on the hidden embeddings.
	"""
	def __init__(self,
		n_feature: int,
		n_latent: int = 10,
		spatial_loss :Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		lambda_orth_loss = 0.1
	) -> None:
		super().__init__()

		self.n_latent = n_latent # dimension of the latent space
		self.n_feature = n_feature # num_feature to project

		# store data to project
		self._data = None

		# store spatial loss
		self.spatial_loss = spatial_loss
		self.l_sp_loss = lambda_spatial_loss

		# store orthogonality loss
		self.l_orth_loss = lambda_orth_loss

		# configs and logs for dimension reduction
		self.dr_configs = {}
		self.dr_logs = {'elapsed_time': None, 'total_loss': [],
						'recon_loss': [], 'orth_loss': [],
						'spatial_loss': []}

	def load_adata(self, adata: AnnData, layer: Optional[str] = None):
		"""Load data to project.

		Args:
			adata (AnnData): data to project.
			layer (str): layer to project. If None, use adata.X.
		"""
		assert adata.shape[1] == self.n_feature, "The new adata has different number of features!"

		if layer is None:
			self._data = torch.tensor(adata.X.toarray().T) # n_feature x n_sample
		else:
			self._data = torch.tensor(adata.layers[layer].toarray().T) # n_feature x n_sample

	def forward(self, x):
		return self.decode(self.encode(x))

	def encode(self, x):
		return NotImplementedError

	def decode(self, x_enc):
		return NotImplementedError

	def get_latent_representation(self):
		"""Get the latent representation of the loaded data."""
		return self.encode(self._data).detach().numpy()

	def get_recon_loss(self):
		"""Get the reconstruction loss."""
		raise NotImplementedError

	def get_orth_loss(self):
		"""Get the orthogonality loss (weighted by lambda_orth_loss)."""
		raise NotImplementedError

	def get_sp_loss(self):
		"""Get the spatial loss (weighted by lambda_spatial_loss)."""
		raise NotImplementedError

	def reduce(self, lr = 1e-2, max_epochs = 1000, patience = 10, tol = 1e-5,
			   optimizer = 'SGD', verbose = True, quite = False, clear_logs = True):

		self.dr_configs = {
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
			'optimizer':optimizer, 'verbose':verbose, 'quite':quite,
   			'clear_logs':clear_logs, 'return_model':False
		}
		return AutoEncoderClass._reduce_dim_ae(ae_model=self, **self.dr_configs)

	@classmethod
	def _reduce_dim_ae(
		cls, ae_model,
		lr = 1e-2, max_epochs = 1000, patience = 10, tol = 1e-5,
		optimizer = 'SGD', verbose = True, quite = False,
		clear_logs = False, return_model = False
	):
		"""Dimension reduction using auto-encoders.

		Two additional losses are added in addition to the reconstruction loss:
			1. Orthogonality loss: the loss of the orthogonality of the hidden embedding.
			2. Spatial loss: the spatial loss on the hidden embeddings.

		Args:
			ae_model (AutoEncoder): The autoencoder model.
			lr (float): The learning rate.
			max_epochs (int): The maximum number of epochs.
			patience (int): The patience for early stopping.
			tol (float): The tolerated convergence error.
			optimizer (str): The optimizer to be used. Can be 'SGD' or 'Adam'.
			verbose (bool): If True, print out loss while training.
			quite (bool): If True, no output printed.
			clear_logs (bool): If True, clear the logs in the autoencoder model.
			return_model (bool): If True, return the trained autoencoder model.
		Returns:
			ae_model (AutoEncoder): The trained autoencoder model.
		"""

		# start timer
		t_start = timer()

		# set optimizer
		assert optimizer in ['SGD', 'Adam']
		if optimizer == 'Adam':
			optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr)
		else:
			optimizer = torch.optim.SGD(ae_model.parameters(), lr=lr, momentum=0.9)

		# set scheduler
		scheduler = ReduceLROnPlateau(
			optimizer,
			patience = int(patience / 4) + 1,
			factor = 0.1,
			verbose = verbose
		)

		# clear logs
		if clear_logs:
			ae_model.dr_logs = {'elapsed_time': None, 'total_loss': [],
							'recon_loss': [], 'orth_loss': [],
							'spatial_loss': []}

		# start training
		ae_model.train()

		# set iteration limits
		epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
		max_epochs = 10000 if max_epochs == -1 else max_epochs
		patience = patience if patience > 0 else 1

		while epoch < max_epochs and d_loss >= tol or (d_loss < 0 and patience > 0):

			optimizer.zero_grad()

			# calculate reconstruction error
			recon_loss = ae_model.get_recon_loss()
			loss = recon_loss
			recon_loss = recon_loss.detach().item()

			# calculate orthogonality loss
			if ae_model.l_orth_loss > 0:
				orth_loss = ae_model.get_orth_loss()
				loss += orth_loss
				orth_loss = orth_loss.detach().item()
			else:
				orth_loss = 0.0

			# calculate spatial loss
			if ae_model.spatial_loss is not None and ae_model.l_sp_loss > 0:
				sp_loss = ae_model.get_sp_loss()
				loss += sp_loss
				sp_loss = sp_loss.detach().item()
			else:
				sp_loss = 0.0

			# backpropagate and update weights
			loss.backward()
			optimizer.step()

			# check convergence
			epoch += 1
			d_loss = prev_loss - loss.detach().item()
			if d_loss < 0: # if loss increases
				patience -= 1

			prev_loss = loss.detach().item()

			# update learning rate
			scheduler.step(prev_loss)

			if (verbose and not quite) and epoch % 10 == 0:
				print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
					f'(orth) {orth_loss:.4f}. (spatial) {sp_loss:.4f}.')

			# store losses to the log
			ae_model.dr_logs['total_loss'].append(prev_loss)
			ae_model.dr_logs['recon_loss'].append(recon_loss)
			ae_model.dr_logs['spatial_loss'].append(sp_loss)
			ae_model.dr_logs['orth_loss'].append(orth_loss)

		t_end = timer()
		ae_model.dr_logs['elapsed_time'] = t_end - t_start

		if not quite: # print final message
			print(f'=== Time {t_end - t_start : .2f}s. Total epoch {epoch}. '
					f'Final Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
						f'(orth) {orth_loss:.4f}. (spatial) {sp_loss:.4f}.')

		if d_loss >= 1e-5:
			warnings.warn("Fail to converge. Try to increase 'max_epochs'.")

		if return_model:
			return ae_model


class SpatialAutoEncoder(AutoEncoderClass):
	"""Generic spatial auto-encoder."""
	def __init__(
		self,
		adata: AnnData,
		layer: Optional[str] = None,
		spatial_loss :Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1,
		lambda_orth_loss = 0.1,
		recon_loss_mode: Literal['mse', 'poisson'] = 'poisson',
		n_layers: int = 1,
		n_hidden: int = 128,
		n_latent: int = 10,
		dropout_rate: float = 0.0,
		use_batch_norm: bool = True,
		use_activation: bool = True,
		activation_fn: Optional[nn.Module] = nn.ReLU,
		use_bias = True
	) -> None:
		# initialize the model
		n_feature = adata.shape[1]
		super().__init__(
			n_feature = n_feature,
			n_latent = n_latent,
			spatial_loss = spatial_loss,
			lambda_spatial_loss = lambda_spatial_loss,
			lambda_orth_loss = lambda_orth_loss
		)

		# store set up arguments
		self.setup_args = {
			'lambda_orth_loss': lambda_orth_loss,
			'recon_loss_mode': recon_loss_mode,
			'n_layers': n_layers,
			'n_hidden': n_hidden,
			'n_latent': n_latent,
			'dropout_rate': dropout_rate,
			'use_batch_norm': use_batch_norm,
			'use_activation': use_activation,
			'activation_fn': activation_fn,
			'use_bias': use_bias
		}

		# reconstruction loss function
		if recon_loss_mode	== 'mse':
			self._recon_loss_fn = nn.MSELoss(reduction='mean')
		elif recon_loss_mode == 'poisson':
			self._recon_loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean')
		else:
			raise NotImplementedError

		# load data
		self.load_adata(adata, layer)
		self.lib_size = self._data.sum(dim=0) # per-spot library size

		# initialize the encoder
		self.encoder = FCLayers(
			n_in = n_feature,
			n_out = n_latent,
			n_layers = n_layers,
			n_hidden = n_hidden,
			dropout_rate = dropout_rate,
			use_batch_norm = use_batch_norm,
			use_activation = use_activation,
			activation_fn = activation_fn,
			bias = use_bias
		)

		# initialize the decoder
		self.decoder = nn.Sequential(
  			FCLayers(
				n_in = n_latent,
				n_out = n_hidden,
				n_layers = n_layers,
				n_hidden = n_hidden,
				dropout_rate = 0,
				use_batch_norm = use_batch_norm,
				use_activation = use_activation,
				activation_fn = activation_fn,
				bias = use_bias
			),
			nn.Linear(n_hidden, n_feature, bias=use_bias),
			nn.Softplus()
		)

	def encode(self, x):
		"""Project the input to the hidden space.

		Args:
			x (2D tensor): The input matrix, num_gene x num_spot.

		Returns:
			x_enc (2D tensor): The hidden embedding, num_spot x num_hidden.
		"""
		return self.encoder(x.T)

	def decode(self, x_enc):
		"""Project the hidden embedding back to the input space.

		Args:
			x_enc (2D tensor): The hidden embedding, num_spot x num_hidden.

		Returns:
			x_dec (2D tensor): The decoded matrix, num_gene x num_spot.
		"""
		x_dec = self.decoder(x_enc).T

		return x_dec

	def get_recon_loss(self):
		x = self._data # num_gene x num_spot
		y = self(x) # num_gene x num_spot
		return self._recon_loss_fn(y, x)

	def get_orth_loss(self):
		x = self._data # num_gene x num_spot

		if self.l_orth_loss > 0:
			enc = self.encode(x) # num_spot x num_latent
			enc_norm = enc.norm(dim=0, keepdim=True) + 1e-8
			# loss to make sure the latent embedding is orthogonal
			orth_loss = torch.norm(
				(enc / enc_norm).T @ (enc / enc_norm) - torch.eye(enc.shape[1])
			)
			# add l2 loss to constrain the scale of the latent embedding
			orth_loss = self.l_orth_loss * orth_loss

			return orth_loss
		else:
			return 0.0

	def get_sp_loss(self):
		x = self._data # num_gene x num_spot

		if self.spatial_loss is not None and self.l_sp_loss > 0:
			enc = self.encode(x) # num_spot x num_latent
			sp_loss = self.l_sp_loss * self.spatial_loss(enc.T, normalize = True)

			return sp_loss
		else:
			return 0.0

	@classmethod
	def from_rna_model(
		cls,
		rna_model,
		st_adata: AnnData,
		layer: Optional[str] = None,
		spatial_loss :Optional[SpatialLoss] = None,
		lambda_spatial_loss = 0.1
	):
		"""Initialize a spatial model from a pre-trained RNA model."""
		model_args = deepcopy(rna_model.setup_args)
		sp_model = cls(
			adata = st_adata,
			layer = layer,
			spatial_loss = spatial_loss,
			lambda_spatial_loss = lambda_spatial_loss,
			**model_args
		)

		# copy the encoder and decoder state dict
		sp_model.load_state_dict(rna_model.state_dict(), strict=False)

		return sp_model
