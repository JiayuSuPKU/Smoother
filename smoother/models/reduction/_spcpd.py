
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

from smoother.losses import SpatialLoss

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
