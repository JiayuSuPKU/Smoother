from timeit import default_timer as timer
import warnings
import torch
import torch.nn as nn

# optional imports for tensor decomposition
try:
	import tensorly as tl
	from tensorly.decomposition import non_negative_parafac, parafac
	tl.set_backend('pytorch')
except ImportError:
	warnings.warn("Package 'tensorly' not installed. "
				  "Tensor decomposition is not supported.",
				  category=ImportWarning)
	tl, non_negative_parafac, parafac = None, None, None

from smoother.models.losses import SpatialLoss

class PCA(nn.Module):
	"""Solving PCA using stochastic gradient descent."""
	def __init__(self, num_feature, num_pc) -> None:
		super().__init__()
		self.k = num_pc # num_pc
		self.d = num_feature # num_feature to project

		# initialize PC basis matrix (num_feature x num_pc)
		U_init = self.gram_schmidt(nn.init.uniform_(torch.zeros(self.d, self.k)))
		self.U = nn.parameter.Parameter(U_init, requires_grad = True)

		self.dr_configs = None
		self.dr_time = 0

	def gram_schmidt(self, U):
		"""Project the PC basis matrix to the feasible space.

		Args:
			U (2D tensor): PC basis matrix, num_feature x num_pc.
		"""
		# U will be orthogonal after transformation
		return torch.linalg.qr(U, mode = 'reduced')[0]

	def forward(self, x):
		"""Project x to the lower PC space.

		Args:
			x (2D tensor): data to project, num_feature x num_sample.
		"""
		return self.U.T @ x # num_pc x num_sample

	def init_with_svd(self, x):
		"""Initialize model with svd solution.

		Args:
			x (2D tensor): data to project, num_feature x num_sample.
		"""
		with torch.no_grad():
			_, _, V = torch.svd_lowrank(x.T, self.k)
			self.U.copy_(V) # num_feature x num_pc

	def reduce(self, st_expr,
			   spatial_loss : SpatialLoss = None, lambda_spatial_loss = 0.1,
			   lr = 1e-3, max_epochs = 1000, patience = 10, tol = 1e-5,
			   init_with_svd = True, verbose = True, quite = False):
		"""Reduce the dimension of the expression matrix.

		Args:
			st_expr (2D tensor): The expression matrix to be reduced, num_gene x num_spot.
			spatial_loss (SpatialLoss): The spatial loss object to be used.
			lambda_spatial_loss (float): The strength of the spatial loss.
			lr (float): The learning rate.
			max_epochs (int): The maximum number of epochs.
			patience (int): The patience for early stopping.
			tol (float): The tolerated convergence error.
			init_with_svd (bool): Whether to initialize with analytical solution calculated
				using `torch.svd_lowrank()`.
			verbose (bool): If True, print out loss while training.
			quite (bool): If True, no output printed.
		"""
		self.dr_configs = {
			'spatial_loss':spatial_loss, 'lambda_spatial_loss':lambda_spatial_loss,
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
			'init_with_svd':init_with_svd
		}

		# start timer
		t_start = timer()

		optimizer = torch.optim.SGD(self.parameters(), lr=lr)

		if init_with_svd:
			self.init_with_svd(st_expr)

		# start training
		self.train()

		# set iteration limits
		epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
		max_epochs = 10000 if max_epochs == -1 else max_epochs
		patience = patience if patience > 0 else 1

		while epoch < max_epochs and d_loss >= tol or (d_loss < 0 and patience > 0):

			# set loss to 0
			recon_loss, sp_loss = torch.tensor(0), torch.tensor(0)

			optimizer.zero_grad()

			loss = 0

			# project data to hidden subspace
			hidden = self(st_expr) # k x num_spot

			# calculate reconstruction error
			# here we want to maximize the variance in the subspace
			recon_loss = - torch.norm(hidden, dim = 0).pow(2).sum() / (hidden.shape[0] * hidden.shape[1])
			loss += recon_loss

			# add spatial loss on U.T @ x
			if spatial_loss is not None and lambda_spatial_loss > 0:
				sp_loss = lambda_spatial_loss * spatial_loss(hidden)
				loss += sp_loss

			# backpropagate and update weights
			loss.backward()
			optimizer.step()

			# project U to the feasible space
			with torch.no_grad():
				self.U.copy_(self.gram_schmidt(self.U))

			# check convergence
			epoch += 1
			d_loss = prev_loss - loss.detach().item()
			if d_loss < 0: # if loss increases
				patience -= 1

			prev_loss = loss.detach().item()
			recon_loss = recon_loss.detach().item()
			sp_loss = sp_loss.detach().item()

			if (verbose and not quite) and epoch % 10 == 0:
				print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
					  f'(spatial) {sp_loss:.4f}.')

		t_end = timer()
		self.dr_time = t_end - t_start

		if not quite: # print final message
			print(f'=== Time {self.dr_time : .2f}s. Total epoch {epoch}. '
				  f'Final Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
					  f'(spatial) {sp_loss:.4f}.')

		if d_loss >= tol:
			warnings.warn("Fail to converge. Try to increase 'max_epochs'.")


class AutoEncoder(nn.Module):
	"""AutoEncoder class.

	To mimic PCA, the overall objective contains three losses:
		1. Reconstruction loss: reconstruction error of the observed data.
		2. Orthogonality loss: the loss of orthogonality on the hidden embeddings.
		3. Spatial loss: the spatial loss on the hidden embeddings.
	"""
	def __init__(self) -> None:
		super().__init__()
		self.dr_configs = {}
		self.dr_time = 0

	def forward(self, x):
		return self.decode(self.encode(x))

	def encode(self, x):
		return NotImplementedError

	def decode(self, x_enc):
		return NotImplementedError

	def get_recon_loss(self):
		"""Get the reconstruction loss."""
		raise NotImplementedError

	def get_orth_loss(self):
		"""Get the orthogonality loss (weighted by lambda_orth_loss)."""
		raise NotImplementedError

	def get_sp_loss(self):
		"""Get the spatial loss (weighted by lambda_spatial_loss)."""
		raise NotImplementedError

	def init_with_pca(self, data):
		raise NotImplementedError

	def reduce(self, st_expr, spatial_loss : SpatialLoss = None,
			   lambda_spatial_loss = 0.1, lambda_orth_loss = 1,
			   lr = 1e-3, max_epochs = 1000, patience = 10, tol = 1e-5,
			   optimizer = 'SGD', init_with_pca = True,
			   verbose = True, quite = False):

		self.dr_configs = {
			'st_expr':st_expr, 'spatial_loss':spatial_loss,
			'lambda_spatial_loss':lambda_spatial_loss, 'lambda_orth_loss':lambda_orth_loss,
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
			'optimizer':optimizer, 'init_with_pca':init_with_pca,
			'verbose':verbose, 'quite':quite,
			'return_model':False
		}
		return _reduce_dim_ae(ae_model=self, **self.dr_configs)


def _reduce_dim_ae(ae_model : AutoEncoder,
				   st_expr, spatial_loss : SpatialLoss = None,
				   lambda_spatial_loss = 1, lambda_orth_loss = 1,
				   lr = 1e-3, max_epochs = 1000, patience = 10, tol = 1e-5,
				   optimizer = 'SGD', init_with_pca = True,
				   verbose = True, quite = False, return_model = False):
	"""Dimension reduction using auto-encoders.

	Two additional losses are added in addition to the reconstruction loss:
		1. Orthogonality loss: the loss of the orthogonality of the hidden embedding.
		2. Spatial loss: the spatial loss on the hidden embeddings.

	Args:
		ae_model (AutoEncoder): The autoencoder model.
		st_expr (2D tensor): The expression matrix to be reduced, num_gene x num_spot.
		spatial_loss (SpatialLoss): The spatial loss object to be used.
		lambda_spatial_loss (float): The strength of the spatial loss.
		lambda_orth_loss (float): The strength of the orthogonality loss on embedding.
		lr (float): The learning rate.
		max_epochs (int): The maximum number of epochs.
		patience (int): The patience for early stopping.
		tol (float): The tolerated convergence error.
		optimizer (str): The optimizer to be used. Can be 'SGD' or 'Adam'.
		init_with_pca (bool): Whether to initialize the weights of the autoencoder with PCA.
		verbose (bool): If True, print out loss while training.
		quite (bool): If True, no output printed.
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
		optimizer = torch.optim.SGD(ae_model.parameters(), lr=lr)

	# initialize with PCA matrices
	if init_with_pca:
		ae_model.init_with_pca(st_expr)

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
		if lambda_orth_loss > 0:
			orth_loss = ae_model.get_orth_loss()
			loss += orth_loss
			orth_loss = orth_loss.detach().item()
		else:
			orth_loss = 0.0

		# calculate spatial loss
		if spatial_loss is not None and lambda_spatial_loss > 0:
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

		if (verbose and not quite) and epoch % 10 == 0:
			print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
				  f'(orth) {orth_loss:.4f}. (spatial) {sp_loss:.4f}.')

	t_end = timer()
	ae_model.dr_time = t_end - t_start

	if not quite: # print final message
		print(f'=== Time {ae_model.dr_time : .2f}s. Total epoch {epoch}. '
				f'Final Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
					f'(orth) {orth_loss:.4f}. (spatial) {sp_loss:.4f}.')

	if d_loss >= 1e-5:
		warnings.warn("Fail to converge. Try to increase 'max_epochs'.")

	# Depending on the weight matrix, sometimes the inverse covariance matrix can
	# have negative eigenvalues, be careful about the regression coefficient
	if prev_loss < 0:
		warnings.warn(
			"Negative loss occurred because the inverse covariance matrix is not positive semidefinite. "
			"If contrastive loss is used, try a smaller 'neg2pos_ratio' value. "
			"Otherwise, try to restrict 'spatial_weights' or use a smaller 'smooth_l'."
		)

	if return_model:
		return ae_model


class LinearAutoEncoder(AutoEncoder):
	"""Generic linear auto-encoder."""
	def __init__(self, num_feature, num_hidden, use_bias = True) -> None:
		super().__init__()
		self.num_feature = num_feature
		self.num_hidden = num_hidden
		# encoder and decoder each uses a single linear layer
		self.encoder = nn.Linear(self.num_feature, self.num_hidden, bias=use_bias)
		self.decoder = nn.Linear(self.num_hidden, self.num_feature, bias=use_bias)

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
		return self.decoder(x_enc).T

	def init_with_pca(self, data):
		"""Initialize the weights of the linear auto-encoder with PCA solution.

		Args:
			data (2D tensor): The data matrix to be projected. num_gene x num_spot.
		"""
		with torch.no_grad():
			_, _, init_weights = torch.pca_lowrank(data.T, q = self.num_hidden)
			self.encoder.weight.copy_(init_weights.T) # dim_hidden x num_gene
			self.decoder.weight.copy_(init_weights) # num_gene x dim_hidden

	def get_recon_loss(self):
		x = self.dr_configs.get('st_expr') # num_gene x num_spot
		y = self(x) # num_gene x num_spot
		return nn.functional.mse_loss(y, x)

	def get_orth_loss(self):
		x = self.dr_configs.get('st_expr') # num_gene x num_spot
		lambda_orth_loss = self.dr_configs.get('lambda_orth_loss')

		if lambda_orth_loss > 0:
			enc = self.encode(x) # num_spot x num_hidden
			enc_norm = enc.norm(dim=0, keepdim=True) + 1e-8
			# loss to make sure the hidden embedding is orthogonal
			orth_loss = torch.norm(
				(enc / enc_norm).T @ (enc / enc_norm) - torch.eye(enc.shape[1])
			)
			# add l2 loss to constrain the scale of the hidden embedding
			orth_loss = lambda_orth_loss * orth_loss

			return orth_loss
		else:
			return 0.0

	def get_sp_loss(self):
		x = self.dr_configs.get('st_expr')
		spatial_loss = self.dr_configs.get('spatial_loss')
		lambda_spatial_loss = self.dr_configs.get('lambda_spatial_loss')

		if spatial_loss is not None and lambda_spatial_loss > 0:
			enc = self.encode(x)
			sp_loss = lambda_spatial_loss * spatial_loss(enc.T)

			return sp_loss
		else:
			return 0.0


class MultilayerAutoEncoder(LinearAutoEncoder):
	"""Multilayer autoencoder with non-linearity.

	The encoder and decoder are symmetric in dimensions.
	"""
	def __init__(self, layer_dim_list) -> None:
		super().__init__(1, 1)
		self.layer_dim_list = layer_dim_list
		self.num_layer = len(layer_dim_list)
		self.num_feature = layer_dim_list[0]
		self.num_hidden = layer_dim_list[-1]

		# multilayer encoder
		encoder_layers = []
		for i in range(len(layer_dim_list)-1):
			encoder_layers.append(nn.Linear(layer_dim_list[i], layer_dim_list[i+1]))
			# use non-linear activation function except for the last layer
			if i < len(layer_dim_list) - 2:
				encoder_layers.append(nn.ReLU())
			# else: # use tanh activation function so that the embedding is in [-1, 1]
			# 	encoder_layers.append(nn.Tanh())

		self.encoder = nn.Sequential(*encoder_layers)

		# multilayer decoder
		decoder_layers = []
		for i in range(len(layer_dim_list)-1):
			decoder_layers.append(nn.Linear(layer_dim_list[-i-1], layer_dim_list[-i-2]))
			decoder_layers.append(nn.ReLU())
		self.decoder = nn.Sequential(*decoder_layers)

	def init_with_pca(self, data):
		"""Initialize the weights of the linear auto-encoder with PCA solution.

		Args:
			data (2D tensor): The data matrix to be projected. num_gene x num_spot.
		"""
		with torch.no_grad():
			_, _, init_weights = torch.pca_lowrank(data.T, q = self.layer_dim_list[1])
			self.encoder[0].weight.copy_(init_weights.T) # dim_hidden1 x num_gene
			self.decoder[-2].weight.copy_(init_weights) # num_gene x dim_hidden1


class CPDecomposition(nn.Module):
	"""Tensor CP decomposition."""
	def __init__(self, shape_tensor, dim_hidden, nonneg = True) -> None:
		super().__init__()
		self.cpd_configs = {}
		self.nonneg = nonneg

		self.dim_x, self.dim_y, self.dim_z = shape_tensor
		self.dim_hidden = dim_hidden

		# factorized low-dimensional embedding
		# target = weights * outer(x_hidden, y_hidden, z_hidden)
		self.weights = nn.Parameter(torch.ones(dim_hidden), requires_grad=False)
		self.x_hidden = nn.Parameter(torch.rand(self.dim_x, dim_hidden), requires_grad=True)
		self.y_hidden = nn.Parameter(torch.rand(self.dim_y, dim_hidden), requires_grad=True)
		self.z_hidden = nn.Parameter(torch.rand(self.dim_z, dim_hidden), requires_grad=True)

	def forward(self):
		# target = weights * outer(x_hidden, y_hidden, z_hidden)
		if not self.nonneg:
			return tl.cp_to_tensor(
	  			(self.weights, (self.x_hidden, self.y_hidden, self.z_hidden))
			)
		return tl.cp_to_tensor(
			(self.weights, (torch.abs(self.x_hidden), torch.abs(self.y_hidden), torch.abs(self.z_hidden)))
		)

	def get_loss_fn(self):
		return torch.nn.MSELoss()

	def init_with_tensorly(self, target):
		"""Initialize tensor decomposition using tensorly."""
		if self.nonneg:
			weights, factors = non_negative_parafac(target, self.dim_hidden, init='svd')
		else:
			weights, factors = parafac(target, self.dim_hidden, init='svd')
		self.weights.data = weights
		self.x_hidden.data = factors[0]
		self.y_hidden.data = factors[1]
		self.z_hidden.data = factors[2]

	def decomposition_sp(self, target,
						spatial_loss : SpatialLoss = None, lambda_spatial_loss = 0.1,
						lr = 1e-3, max_epochs = 1000, patience = 10,
						init_with_tensorly = False, verbose = True, quite = False):
		self.cpd_configs = {
			'spatial_loss':spatial_loss, 'lambda_spatial_loss':lambda_spatial_loss,
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience,
			'init_with_tensorly':init_with_tensorly, 'verbose':verbose, 'quite':quite,
			'return_model':False
		}
		return _decomposition_sp(target, model=self, **self.cpd_configs)


def _decomposition_sp(target, model,
		   spatial_loss : SpatialLoss = None, lambda_spatial_loss = 0.1,
		   lr = 1e-3, max_epochs = 1000, patience = 10,
		   init_with_tensorly = True, verbose = True, quite = False, return_model = False):
	# start timer
	t_start = timer()

	# define loss function and optimizer
	loss_fn = model.get_loss_fn()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# initialize tensor decomposition
	if init_with_tensorly:
		model.init_with_tensorly(target)

	# start training
	model.train()

	# set iteration limits
	epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
	max_epochs = 10000 if max_epochs == -1 else max_epochs
	patience = patience if patience > 0 else 1

	while epoch < max_epochs and d_loss >= 1e-5 or (d_loss < 0 and patience > 0):

		optimizer.zero_grad()

		# reconstruct target from hidden factors
		outputs = model()

		# calculate reconstruction error
		loss = loss_fn(outputs, target) # mse loss

		# store the model loss
		model_loss = loss.detach().item()

		# add spatial loss
		if spatial_loss is not None:
			sp_rv = torch.cat([model.x_hidden, model.y_hidden], dim=1).T
			loss += lambda_spatial_loss * spatial_loss(sp_rv)

		# backpropagate and update weights
		loss.backward()
		optimizer.step()

		# check convergence
		epoch += 1
		d_loss = prev_loss - loss.detach().item()
		if d_loss < 0: # if loss increases
			patience -= 1

		prev_loss = loss.detach().item()

		if (verbose and not quite) and epoch % 10 == 0:
			print(f'Epoch {epoch}. Total loss {prev_loss:.4f}. Model loss {model_loss:.4f}.')

	t_end = timer()

	if not quite: # print final message
		print(f'=== Time {t_end - t_start : .2f}s. Total epoch {epoch}. '
			  f'Final loss {prev_loss:.2f}.')

	if d_loss >= 1e-5:
		warnings.warn("Fail to converge. Try to increase 'max_epochs'.")

	# Depending on the weight matrix, sometimes the inverse covariance matrix can
	# have negative eigenvalues, be careful about the regression coefficient
	if prev_loss < 0:
		warnings.warn(
	  		"Negative loss occurred because the inverse covariance matrix is not positive semidefinite."
			"Try to restrict 'spatial_weights' or use a smaller 'smooth_l'."
		)

	if return_model:
		return model
