"""
Deconvolute spatial-omics data with spatial loss
"""
import warnings
from functools import partial
from timeit import default_timer as timer
import numpy as np
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from torch.autograd import Variable
from smoother.losses import SpatialLoss, TopLoss

# optional imports for convex optimization
try:
	import cvxpy as cp
	from cvxpy.atoms.affine.wraps import psd_wrap
except ImportError:
	warnings.warn("Package 'cvxpy' not installed so that "
				  "Convex optimization solvers are not supported.",
				  category=ImportWarning)
	cp, psd_wrap = None, None

torch.set_default_dtype(torch.float32)

class DeconvModel():
	"""Interface for convex deconvolution models."""
	def get_model_loss(self):
		"""Get regression reconstruction loss."""
		raise NotImplementedError

	def get_sp_loss(self):
		"""Get spatial loss (weighted by lambda_spatial_loss)."""
		raise NotImplementedError

	def get_props(self):
		"""Get the predicted celltype abundance from the trained deconvolution model."""
		raise NotImplementedError

def LinearRegression(backend = 'pytorch', **kwargs) -> DeconvModel:
	assert backend in ['pytorch', 'cvxpy']
	if backend == 'pytorch':
		return LinearRegressionTorch(**kwargs)
	else:
		return LinearRegressionConvex(**kwargs)

def NNLS(backend = 'pytorch', **kwargs) -> DeconvModel:
	assert backend in ['pytorch', 'cvxpy']
	if backend == 'pytorch':
		return NNLSTorch(**kwargs)
	else:
		return NNLSConvex(**kwargs)

def NuSVR(backend = 'pytorch', **kwargs) -> DeconvModel:
	assert backend in ['pytorch', 'cvxpy']
	if backend == 'pytorch':
		return NuSVRTorch(**kwargs)
	else:
		return NuSVRConvex(**kwargs)

def DWLS(backend = 'pytorch', **kwargs) -> DeconvModel:
	assert backend in ['pytorch', 'cvxpy']
	if backend == 'pytorch':
		return DWLSTorch(**kwargs)
	else:
		return DWLSConvex(**kwargs)

def LogNormReg(backend = 'pytorch', **kwargs) -> DeconvModel:
	assert backend in ['pytorch']
	if backend == 'pytorch':
		return LogNormRegTorch(**kwargs)

class DeconvModelTorch(nn.Module, DeconvModel):
	"""Class for deconvolution models implemented using pytorch.

	Attributes:
		dim_in (int): Input dimension, also number of groups.
		dim_out (int): Output dimension, also number of spots.
		use_bias (bool): Whether to add bias term in regression.
		model_name (str): Name of the deconvolution model.
		nonneg (bool): Whether to apply nonnegative contraint
			on spatial variables.
		deconv_configs (dict): Dictionary of deconvolution configurations.
		deconv_time (float): Time spent on deconvolution.
	"""
	def __init__(self, dim_in, dim_out, model_name) -> None:
		super().__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.model_name = model_name
		self.use_bias = None
		self.nonneg = None # nonnegative constraint
		self.deconv_configs = None # deconvolution configs
		self.deconv_time = None # time spent in deconvolution

	def set_params(self, dim_in, dim_out, bias):
		raise NotImplementedError

	def get_top_loss(self):
		"""Get toplogical loss (weighted by lambda_top_loss)."""
		raise NotImplementedError

	def init_with_lr_sol(self, x, y):
		"""Initialize model with linear regression solution."""
		raise NotImplementedError

	def final_sanity_check(self):
		"""Make sure constraints are followed after deconvolution."""
		raise NotImplementedError

	def deconv(self, x, y,
			   spatial_loss : SpatialLoss = None, lambda_spatial_loss = 0.0,
			   top_loss : TopLoss = None, lambda_top_loss = 0.0,
			   lr = 1e-3, max_epochs = 1000, patience = 10, tol = 1e-5,
			   init_with_lr_sol = True, verbose = True, quiet = False):
		"""Deconvolute spatial-omics data with spatial loss.

		Loss_total = Loss_model + `lambda_spatial_loss` * Loss_spatial +
			`lambda_top_loss` * Loss_topological.

		Args:
			x (2D tensor): Bulk feature signiture matrix, num_feature x num_group.
			y (2D tensor): Spatial feature matrix, num_feature x num_spot.
			spatial_loss (SpatialLoss): The spatial smoothing loss.
				Spatial prior can be one of 'none', 'kl', 'sma', 'sar', 'car', 'icar'
				- KL : KL divergence of cell type proportion vectors of neiboring spots
				- SMA : Spatial moving average
				- SAR : Simultaneous auto-regressive model
				- CAR : Conditional auto-regressive model
				- ICAR : Intrinsic conditional auto-regressive model
				See model description for more details.
			lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
			top_loss (TopLoss): The topological loss (expected betti number for each group).
				top_loss.betti_priors = {
					group_id : {betti_k : expected number of barcodes (prior)}, ...
				}
			lambda_top_loss (float): Specifies the strength of the topological constraints.
			lr (float): Learning rate.
			max_epochs (int): Maximum number of training epochs. If -1, iterate until
				convergence (`d_loss` < 1e-5).
			patient (int): Number of epochs to wait for the loss to decrease before stopping.
			tol (float): Tolerance of loss convergence.
			init_with_lr_sol (bool): Whether to initialize regression weights with the OLE solution.
			verbose (bool): If True, print out loss while training.
			quiet (bool): If True, no output printed.
		"""
		self.deconv_configs = {
			'x':x, 'y':y,
	  		'spatial_loss':spatial_loss, 'lambda_spatial_loss':lambda_spatial_loss,
			'top_loss':top_loss, 'lambda_top_loss':lambda_top_loss,
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
			'init_with_lr_sol':init_with_lr_sol, 'verbose':verbose, 'quiet':quiet,
			'return_model':False
		}
		return _deconv_torch(deconv_model=self, **self.deconv_configs)


def _deconv_torch(x, y, deconv_model : DeconvModelTorch,
		   spatial_loss : SpatialLoss = None, lambda_spatial_loss = 1.0,
		   top_loss : TopLoss = None, lambda_top_loss = 0.0,
		   lr = 1e-3, max_epochs = 1000, patience = 10, tol = 1e-5,
		   init_with_lr_sol = True, verbose = True, quiet = False, return_model = True):
	"""Train a regression-based spatial deconvolution model.

	Loss_total = Loss_model + `lambda_spatial_loss` * Loss_spatial +
		`lambda_top_loss` * Loss_topological.

	Args:
		x (2D tensor): Bulk feature signiture matrix, num_feature x num_group.
		y (2D tensor): Spatial feature matrix, num_feature x num_spot.
		deconv_model (DeconvModelTorch): A deconvolution model to be used.
			Model can be one of 'linear', 'nnls', 'svr', or 'dwls'.
			The 'svr' model takes 'C' and 'nu' as two additional inputs, see
			https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html.
		spatial_loss (SpatialLoss): The spatial smoothing loss.
			Spatial prior can be one of 'none', 'kl', 'sma', 'sar', 'car', 'icar'
			- KL : KL divergence of cell type proportion vectors of neiboring spots
			- SMA : Spatial moving average
			- SAR : Simultaneous auto-regressive model
			- CAR : Conditional auto-regressive model
			- ICAR : Intrinsic conditional auto-regressive model
			See model description for more details.
		lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
		top_loss (TopLoss): The topological loss (expected betti number for each group).
			top_loss.betti_priors = {
				group_id : {betti_k : expected number of barcodes (prior)}, ...
			}
		lambda_top_loss (float): Specifies the strength of the topological constraints.
		lr (float): Learning rate.
		max_epochs (int): Maximum number of training epochs. If -1, iterate until
  			convergence (`d_loss` < 1e-5).
		patient (int): Number of epochs to wait for the loss to decrease before stopping.
		tol (float): Tolerance of loss convergence.
		init_with_lr_sol (bool): Whether to initialize regression weights with the OLE solution.
		verbose (bool): If True, print out loss while training.
		quiet (bool): If True, no output printed.
		return_model (bool): If True, return the trained model.
	"""

	# check model dimensions
	dim_in, dim_out = x.shape[1], y.shape[1]
	if (deconv_model.dim_in, deconv_model.dim_out) != (dim_in, dim_out):
		deconv_model.set_params(dim_in, dim_out, deconv_model.use_bias)

	# start timer
	t_start = timer()

	# initialize optimizer
	optimizer = torch.optim.Adam(deconv_model.parameters(), lr=lr)

	# initialize with regression solution (beta = (x^T*x)^-1 * x^T * y)
	if init_with_lr_sol:
		deconv_model.init_with_lr_sol(x, y)

	# start training
	deconv_model.train()

	# set iteration limits
	epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
	max_epochs = 10000 if max_epochs == -1 else max_epochs
	patience = patience if patience > 0 else 1

	while epoch < max_epochs and (d_loss >= tol or (d_loss < 0 and patience > 0)):

		optimizer.zero_grad()

		# prediction of y from linear regression
		# outputs = deconv_model(x)
		# calculate model loss
		loss = deconv_model.get_model_loss()

		# store the model loss
		model_loss = loss.detach().item()

		# add spatial loss
		if spatial_loss is not None and lambda_spatial_loss > 0:
			# sp_loss = lambda_spatial_loss * spatial_loss(deconv_model.lr.weight.T)
			sp_loss = deconv_model.get_sp_loss()
			loss += sp_loss
			sp_loss = sp_loss.detach().item()
		else:
			sp_loss = 0.0

		# add topological loss
		if top_loss is not None and lambda_top_loss > 0:
			# loss += lambda_top_loss * top_loss(deconv_model.lr.weight.T)
			loss += deconv_model.get_top_loss()

		# backpropagate and update weights
		loss.backward()
		optimizer.step()

		# check convergence
		epoch += 1
		d_loss = prev_loss - loss.detach().item()
		if d_loss < 0: # if loss increases
			patience -= 1

		prev_loss = loss.detach().item()

		if (verbose and not quiet) and epoch % 10 == 0:
			print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. '
		 		  f'(model) {model_loss:.4f}. (spatial) {sp_loss:.4f}.')

	# make sure constraints are satisfied
	deconv_model.final_sanity_check()

	t_end = timer()

	# save runtime
	deconv_model.deconv_time = t_end - t_start

	if not quiet: # print final message
		print(f'=== Time {deconv_model.deconv_time : .2f}s. Total epoch {epoch}. '
			  f'Final loss: (total) {prev_loss:.3f}. (spatial) {sp_loss:.3f}.')

	if d_loss >= tol:
		warnings.warn("Fail to converge. Try to increase 'max_epochs'.")

	# Depending on the weight matrix, sometimes the inverse covariance matrix can
	# have negative eigenvalues, be careful about the regression coefficient
	if prev_loss < 0:
		warnings.warn(
			"Negative loss occurred because the inverse covariance matrix is not positive semidefinite. "
			"Try to restrict 'spatial_weights' or use a smaller 'smooth_l'."
		)

	if return_model:
		return deconv_model


class LinearRegressionTorch(DeconvModelTorch):
	"""Linear regression implemented using pytorch."""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1):
		super().__init__(dim_in, dim_out, model_name = 'linear_torch')

		# initialize the linear layer
		self.lr = None
		self.nonneg = False
		self.use_bias = bias
		self.set_params(dim_in, dim_out, bias)

	def set_params(self, dim_in, dim_out, bias = True) -> None:
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.use_bias = bias
		self.lr = nn.Linear(dim_in, dim_out, bias=bias)

	def forward(self, x):
		if self.nonneg: # update with abs(weights)
			if self.use_bias:
				return torch.matmul(x, torch.abs(self.lr.weight.T)) + self.lr.bias
			else:
				return torch.matmul(x, torch.abs(self.lr.weight.T))
		return self.lr(x)

	def get_model_loss_fn(self):
		return torch.nn.MSELoss()

	def get_model_loss(self):
		"""Calculate regression loss."""
		loss_fn = self.get_model_loss_fn() # MSE loss
		# extract deconv inputs
		x, y = self.deconv_configs.get('x'), self.deconv_configs.get('y')
		# self(x) is the model prediction
		return loss_fn(self(x), y)

	def get_sp_loss(self):
		"""Calculate spatial loss (weighted by lambda_spatial_loss)."""
		# extract spatial loss parameters
		l_sp_loss = self.deconv_configs.get('lambda_spatial_loss')
		sp_loss = self.deconv_configs.get('spatial_loss')
		# calculate spatial loss
		if self.nonneg:
			return l_sp_loss * sp_loss(self.lr.weight.T.abs())
		else:
			return l_sp_loss * sp_loss(self.lr.weight.T)

	def get_top_loss(self):
		"""Calculate topological loss (weighted by lambda_top_loss)."""
		# extract topological loss parameters
		l_top_loss = self.deconv_configs.get('lambda_top_loss')
		top_loss = self.deconv_configs.get('top_loss')
		# calculate topological loss
		return l_top_loss * top_loss(self.lr.weight.T)

	def get_props(self):
		"""Get the predicted celltype abundance from the trained deconvolution model."""
		# remove negative abundances
		props = self.lr.weight.data.clamp(min=0) # num_out x num_in
		# normalize
		props = props / (props.sum(1, keepdim=True) + 1e-8)
		return props

	def init_with_lr_sol(self, x, y):
		"""Initialize model with linear regression solution."""
		init_weights = torch.linalg.lstsq(x, y)[0]
		self.lr.weight.data = init_weights.T

	def final_sanity_check(self):
		"""Make sure constraints are satisfied."""
		# for models with non-negative contraint, ensure the final weight is non-negative
		if self.nonneg:
			with torch.no_grad():
				self.lr.weight.copy_(torch.abs(self.lr.weight.data))


class NNLSTorch(LinearRegressionTorch):
	"""Non-negative least square implemented using pytorch."""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1):
		super().__init__(bias, dim_in, dim_out)
		self.model_name = 'nnls_torch'
		self.nonneg = True


class NuSVRTorch(LinearRegressionTorch):
	"""Nu-Support vector regression implemented using pytorch.

	Attributes:
		C (float): SVR regularization parameter.
		nu (float): Nu-SVR parameter.
		epsilon (float): Epsilon in the epsilon-SVR model, specifying the the epsilon-tube within which
			no penalty is associated in the training loss function with points predicted within a distance
			epsilon from the actual value.
		loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
	"""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1,
				 C = 1.0, nu = 0.1, nonneg = False, loss_mode = 'l2'):
		"""Initialize a Nu-SVR deconvolution model.

		Args:
			C (float): SVR regularization parameter.
			nu (float): Nu-SVR parameter.
			nonneg (bool): Whether to require nonnegative regression coefficients.
			loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
		"""
		super().__init__(bias, dim_in, dim_out)

		# store model configs
		self.model_name = 'svr_torch'
		self.nonneg = nonneg
		self.loss_mode = loss_mode
		self.C = Variable(torch.tensor([C]), requires_grad=False)
		self.nu = Variable(torch.tensor([nu]), requires_grad=False)
		self.epsilon = nn.parameter.Parameter(torch.tensor([1.0]), requires_grad=True)

	def get_model_loss_fn(self):
		return partial(_nu_svr_loss, coefs = self.lr.weight.T, epsilon = torch.abs(self.epsilon),
						C = self.C, nu = self.nu, loss = self.loss_mode)

	def final_sanity_check(self):
		"""Make sure constraints are satisfied."""
		super().final_sanity_check()
		# for svr, ensure the final epsilon is always non-neg
		with torch.no_grad():
			self.epsilon.copy_(torch.abs(self.epsilon.data))

def _nu_svr_loss(input, target, coefs, C, nu, epsilon, reduction = 'mean', loss = 'l2'):
	"""Calculate nu-SVR loss.

	Args:
		input (2D tensor): Predicted spatial features (pred = X * beta), num_feature x num_spot.
		target (2D tensor): Observed spatial features, num_feature x num_spot.
		coefs (2D tenor): Columns of regression coefficients, num_group x num_spot.
		C (float): SVR regularization parameter.
		nu (float): Nu-SVR parameter.
		epsilon (float): Epsilon in the epsilon-SVR model, specifying the the epsilon-tube within which
			no penalty is associated in the training loss function with points predicted within a distance
			epsilon from the actual value.
		reduction (str): Specifies the reduction to apply to the output.
		loss (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.

	Return:
		loss (float): Aggregated nu-SVR loss.
	"""
	# calculate regression loss
	if reduction != 'mean':
		raise NotImplementedError("Currently only support mean reduction")

	if loss not in ['l2', 'l1']:
		raise NotImplementedError("Currently only support l2 and l1 loss")

	regression_loss = torch.clamp(torch.abs(target - input) - epsilon, min=0)
	if loss == 'l2':
		regression_loss = regression_loss.pow(2).mean()
	else:
		regression_loss = regression_loss.mean()

	# add l2 loss and penalty on large epsilon
	loss = C * (nu * epsilon + regression_loss) + torch.mean(coefs ** 2) / 2

	return loss


class DWLSTorch(LinearRegressionTorch):
	"""Damped weighted least square implemented using pytorch.

	Here the error term for each feature is scaled by its observed value, not the predicted
	value as described in https://www.nature.com/articles/s41467-019-10802-z.

	Attributes:
		max_weights (float): The upper limit of weights.
	"""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1, nonneg = True, max_weights = 8) -> None:
		super().__init__(bias, dim_in, dim_out)

		# store model configs
		self.model_name = 'dwls_torch'
		self.nonneg = nonneg
		self.max_weights = max_weights

	def get_model_loss_fn(self):
		return partial(_dwls_loss, max_weights = self.max_weights)

def _dwls_loss(input, target, max_weights = 8):
	"""Calculate dampened weighted least square loss.

	Here the error term for each feature is scaled by its observed value, not the predicted
 	value as described in https://www.nature.com/articles/s41467-019-10802-z.

	Args:
		input (2D tensor): Predicted spatial features (pred = X * beta), num_feature x num_spot.
		target (2D tensor): Observed spatial features, num_feature x num_spot.
		max_weights (float): The upper limit of weights.
	"""
	# calculate ordinary least square
	ols = torch.nn.MSELoss(reduction='none')(input, target) # num_feature x num_spot

	# calculate weights for each gene at each spot
	weights = (target ** (-2)).clamp(max = max_weights ** 2) # num_feature x num_spot
	min_weights, _ = torch.min(weights, dim=0, keepdim=True)
	weights = (weights / min_weights).clamp(max = max_weights)

	# calculate weighted least square
	return torch.mul(ols, weights).mean()


class LogNormRegTorch(LinearRegressionTorch):
	"""Log-normal deconvolution implemented using pytorch.

	Minimize MSE(log(Y), log(X @ W)). Here Y and X are all in the raw count space.

	Instead of minimizing the least square loss, this model will minimize MSE after
	log-transforming both observation and prediction, as described in
	in https://www.nature.com/articles/s41467-022-28020-5 as Algorithm 1.

	Attributes:
		epsilon (float): pseudo count added before log transformation.
	"""
	def __init__(self, bias = True, epsilon = 1, dim_in = 1, dim_out = 1):
		super().__init__(bias, dim_in, dim_out)
		self.model_name = 'lognormreg_torch'
		self.nonneg = True
		self.epsilon = epsilon

	def get_model_loss_fn(self):
		def _log_reg_loss(input, target):
			"""Input and target should be in the raw count space."""
			log_i = torch.log(input + self.epsilon)
			log_t = torch.log(target + self.epsilon)
			return torch.nn.MSELoss(reduction='mean')(log_i, log_t)
		return _log_reg_loss


class DeconvModelConvex(DeconvModel):
	"""Class for convex deconvolution models implemented using cvxpy.

	Attributes:
		dim_in (int): Input dimension, also number of groups.
		dim_out (int): Output dimension, also number of spots.
		model_name (str): Name of the deconvolution model.
		nonneg (bool): Whether to apply nonnegative contraint
			on spatial variables.
		deconv_configs (dict): Dictionary of deconvolution configurations.
		deconv_time (float): Time spent on deconvolution.
	"""
	def __init__(self, dim_in, dim_out, model_name) -> None:
		super().__init__()
		self.dim_in = dim_in # num_groups
		self.dim_out = dim_out # num_spots
		self.model_name = model_name
		self.nonneg = None # nonnegative constraint
		self.deconv_configs = None # deconvolution configs
		self.obj = None # objective function
		self.problem = None # convex problem to solve
		self.deconv_time = None # time spent on deconvolution

	def set_params(self, configs) -> bool:
		"""Set model parameters. Return True if warm start."""
		raise NotImplementedError

	def get_model_loss(self):
		"""Get regression reconstruction loss."""
		raise NotImplementedError

	def get_sp_loss(self):
		"""Get spatial loss (weighted by lambda_spatial_loss)."""
		raise NotImplementedError

	def get_props(self):
		"""Get the predicted celltype abundance from the trained deconvolution model."""
		raise NotImplementedError

	def deconv(self, x, y,
			   spatial_loss : SpatialLoss = None, lambda_spatial_loss = 0.0,
			   verbose = False, quiet = False, solver = None, **kwargs):
		"""Solve the regression-based spatial deconvolution problem using cvxpy.

		Loss_total = Loss_model + `lambda_spatial_loss` * Loss_spatial

		See https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
		for solver options.

		Args:
			x (2D tensor): Bulk feature signiture matrix, num_feature x num_group.
			y (2D tensor): Spatial feature matrix, num_feature x num_spot.
			spatial_loss (SpatialLoss): The spatial smoothing loss.
				Spatial prior can be one of 'none', 'kl', 'sma', 'sar', 'car', 'icar'
				- KL : KL divergence of cell type proportion vectors of neiboring spots
				- SMA : Spatial moving average
				- SAR : Simultaneous auto-regressive model
				- CAR : Conditional auto-regressive model
				- ICAR : Intrinsic conditional auto-regressive model
				See model description for more details.
			lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
				Notice that setting `spatial_loss = None` will be more efficient than
				setting `lambda_spatial_loss = 0`, but the former won't benefit from
				warm start (i.e., speedup when only lambda_spatial_loss is changed).
			verbose (bool): If True, print solver output.
			quiet (bool): If True, no output printed, including final message.
			solver (str): The solver to use.
			kwargs: Additional keyword arguments specifying solver specific options.
		"""
  		# start timer
		t_start = timer()

		# set model parameters according to configs
		dim_in, dim_out = x.shape[1], y.shape[1]
		deconv_configs = {
			'x': x, 'y': y, 'dim_in': dim_in, 'dim_out': dim_out,
	  		'spatial_loss': spatial_loss, 'lambda_spatial_loss': lambda_spatial_loss,
			'verbose': verbose, 'quiet':quiet,
			'solver': solver, 'kwargs': kwargs
		}
		warm_start = self.set_params(deconv_configs)

		if not warm_start:
			# initialize optimization problem
			self.obj = cp.Minimize(self.get_model_loss() + self.get_sp_loss())
			self.problem = cp.Problem(self.obj)

		# pass solver arguments
		print_sol_out = verbose and (not quiet)
		self.problem.solve(solver = solver, verbose = print_sol_out,
						   warm_start=warm_start, **kwargs)

		# check problem status
		if not self.problem.status == cp.OPTIMAL:
			if self.problem.status == cp.UNBOUNDED:
				raise ValueError('Solver encountered numeric issues. '
								 'Try torch-based models or reduce number of genes.')
			warnings.warn(f"Problem status: {self.problem.status}.")

		# extract losses
		total_loss = self.obj.value
		model_loss = self.get_model_loss().value.item()
		if spatial_loss is None or lambda_spatial_loss == 0:
			sp_loss = 0
		else:
			sp_loss = self.get_sp_loss().value.item()

		t_end = timer()

		# save runtime
		self.deconv_time = t_end - t_start

		if not quiet: # print final message
			print(f'=== Time {self.deconv_time : .2f}s. '
				f'Loss: (total) {total_loss : .3f}, '
				f'(model) {model_loss : .3f}, '
				f'(spatial) {sp_loss : .3f}')


class LinearRegressionConvex(DeconvModelConvex):
	"""Linear regression implemented using cvxpy."""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1):
		super().__init__(dim_in, dim_out, model_name = 'linear_cvx')
		self.nonneg = False
		self.use_bias = bias # whether to add the bias term
		if self.use_bias:
			self.bias = None

		# assume row-wise (group-wise) independency, aka each row is controlled
		# by an independent MVN distribution (can share the same cov).
		self.weights = None
		self.list_w_row = None

		# loss expressions
		self.model_loss_exp  = None # regression loss
		self.lambda_spatial_loss = None
		self.spatial_loss_exp = None # spatial_loss * lambda_spatial_loss

		self.warm_start = None # flag indicating whether to warm start

	def set_lambda_only(self, configs) -> bool:
		"""Check if the same configuration is used except lambda_spatial_loss.

		If True, will only update lambda_spatial_loss and enable warm start.
		"""
		if self.deconv_configs is not None:
			# check for shared configs and see whether to warm start
			l_is_same_item = [
				(configs['x'] == self.deconv_configs['x']).all(),
				(configs['y'] == self.deconv_configs['y']).all()
			] + [configs[key] == self.deconv_configs.get(key)
				 for key in (configs.keys() - \
					['x', 'y', 'lambda_spatial_loss', 'verbose', 'quiet'])]
			warm_start = all(l_is_same_item)

			if warm_start: # update only the lambda value to warm up the solver
				if configs['verbose'] and not configs['quiet']:
					print("Only the lambda_spatial_loss will be updated "
						  f"({self.deconv_configs['lambda_spatial_loss']:.3f} -> "
						  f"{configs['lambda_spatial_loss']:.3f}).")
				# update lambda
				self.lambda_spatial_loss.value = configs['lambda_spatial_loss']

				return True

		return False

	def set_params(self, configs) -> bool:
		# for sharing configs, update lambda only
		self.warm_start = self.set_lambda_only(configs)

		# update configurations
		self.deconv_configs = configs

		if self.warm_start: # exit if warm start
			return True

		# set model parameters to optimize
		self.dim_in = self.deconv_configs['dim_in']
		self.dim_out = self.deconv_configs['dim_out']
		self.list_w_row = [cp.Variable((1, self.dim_out), nonneg=self.nonneg)
						   for _ in range(self.dim_in)]
		self.weights = cp.vstack(self.list_w_row)

		# set reconstruction loss
		self.set_model_loss()

		# set spatial loss
		self.set_sp_loss()

		return False

	def set_model_loss(self):
		"""Set model reconstruction loss."""
		x, y = self.deconv_configs['x'], self.deconv_configs['y']

		if self.use_bias:
			self.bias = cp.Variable((1, self.dim_out))
			self.model_loss_exp = cp.sum_squares(x @ self.weights + self.bias - y) \
				/ (y.shape[0] * y.shape[1])
		else:
			self.model_loss_exp = cp.sum_squares(x @ self.weights - y) \
				/ (y.shape[0] * y.shape[1])

	def set_sp_loss(self):
		"""Set spatial loss."""
		sp_loss = self.deconv_configs['spatial_loss']
		# store as lambda as cp.Parameter for future warm start
		self.lambda_spatial_loss = cp.Parameter(nonneg=True)
		self.lambda_spatial_loss.value = self.deconv_configs['lambda_spatial_loss']
		self.spatial_loss_exp = 0

		# note that setting `spatial_loss = None` will be much faster than setting
		# `lambda_spatial_loss = 0`, but the former won't benefit from warm start
		if sp_loss is None: # exit if no spatial loss
			return None

		# extract inverse covariance matrix
		if not sp_loss.standardize_cov:
			raise ValueError("Spatial covariance must be standardized.")
		inv_cov = self.deconv_configs['spatial_loss'].inv_cov

		# use scipy sparse matrix to speed up computation
		if sp_loss.use_sparse:
			inv_cov = [psd_wrap(coo_matrix(
				(t.coalesce().values().numpy(), t.coalesce().indices().numpy()),
				shape=t.shape
    		)) for t in inv_cov]

		if len(inv_cov) == 1: # use the same covariance for all groups
			for w_row in self.list_w_row:
				self.spatial_loss_exp += \
					cp.quad_form(w_row.T, inv_cov[0]) * sp_loss.confidences[0]
		else: # use different covariances for different groups
			for i, w_row in enumerate(self.list_w_row):
				self.spatial_loss_exp += \
					cp.quad_form(w_row.T, inv_cov[i]) * sp_loss.confidences[i]

		self.spatial_loss_exp *= self.lambda_spatial_loss / (self.dim_in * self.dim_out)

	def get_model_loss(self):
		"""Get model regression reconstruction loss."""
		return self.model_loss_exp

	def get_sp_loss(self):
		"""Get spatial loss (weighted by lambda_spatial_loss)."""
		return self.spatial_loss_exp

	def get_props(self):
		"""Get the predicted celltype abundance from the trained deconvolution model."""
		# remove negative abundances
		props = np.clip(self.weights.value, 0, None).T # num_out x num_in
		# normalize
		props = props / (props.sum(1, keepdims=True) + 1e-8)
		return props


class NNLSConvex(LinearRegressionConvex):
	"""Non-negative least square implemented using cvxpy."""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1):
		super().__init__(bias, dim_in, dim_out)
		self.model_name = 'nnls_cvx'
		self.nonneg = True


class DWLSConvex(LinearRegressionConvex):
	"""Damped weighted least square implemented using cvxpy.

	Here the error term for each feature is scaled by its observed value, not the predicted
	value as described in https://www.nature.com/articles/s41467-019-10802-z.

	Attributes:
		max_weights (float): The upper limit of the scaling weights for regression loss.
	"""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1, nonneg = True, max_weights = 8) -> None:
		super().__init__(bias, dim_in, dim_out)
		self.model_name = 'dwls_cvx'
		self.nonneg = nonneg
		self.max_weights = max_weights

	def set_model_loss(self):
		"""Set DWLS regression loss."""
		x, y = self.deconv_configs['x'], self.deconv_configs['y']
		# calculate weights for each gene at each spot
		model_loss_scales = (y.pow(-2)).clamp(max = self.max_weights ** 2) # num_feature x num_spot
		# set the minimum weight to 1
		min_weights, _ = torch.min(model_loss_scales, dim=0, keepdim=True)
		model_loss_scales = (model_loss_scales / min_weights).clamp(max = self.max_weights)

		if self.use_bias:
			self.bias = cp.Variable((1, self.dim_out))
			model_loss_exp = cp.square(x @ self.weights + self.bias - y)
		else:
			model_loss_exp = cp.square(x @ self.weights - y)

		self.model_loss_exp = cp.sum(cp.multiply(model_loss_exp, model_loss_scales)) \
				/ (y.shape[0] * y.shape[1])


class NuSVRConvex(LinearRegressionConvex):
	"""Nu-Support vector regression implemented using cvxpy.

	Attributes:
		C (float): SVR regularization parameter.
		nu (float): Nu-SVR parameter.
		epsilon (float): Epsilon in the epsilon-SVR model, specifying the the epsilon-tube within which
			no penalty is associated in the training loss function with points predicted within a distance
			epsilon from the actual value.
		loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
	"""
	def __init__(self, bias = True, dim_in = 1, dim_out = 1,
			  	 C = 1.0, nu = 0.1, nonneg = False, loss_mode = 'l2'):
		"""Initialize a Nu-SVR deconvolution model.

		Args:
			C (float): SVR regularization parameter.
			nu (float): Nu-SVR parameter.
			nonneg (bool): Whether to require nonnegative regression coefficients.
			loss_mode (str): Specifies the regression loss function to use. Either 'l2' or 'l1'.
		"""
		super().__init__(bias, dim_in, dim_out)

		# store model configs
		self.model_name = 'svr_cvx'
		self.nonneg = nonneg

		# check regression loss fn, 'l2' or 'l1', default 'l2' for faster convergence
		if loss_mode not in ['l2', 'l1']:
			raise ValueError("loss_mode must be either 'l2' or 'l1'.")
		self.loss_mode = loss_mode

		self.C = C
		self.nu = cp.Parameter(nonneg=True)
		self.nu.value = nu
		self.epsilon = None

	def set_model_loss(self):
		"""Set Nu-SVR regression loss."""
		x, y = self.deconv_configs['x'], self.deconv_configs['y']
		self.epsilon = cp.Variable(nonneg=True)

		# calculate regression loss
		if self.use_bias:
			self.bias = cp.Variable((1, self.dim_out))
			res = cp.pos(cp.abs(x @ self.weights + self.bias - y) - self.epsilon)
		else:
			res = cp.pos(cp.abs(x @ self.weights - y) - self.epsilon)

		# use either L2 or L1 loss
		if self.loss_mode == 'l2':
			model_loss = cp.sum_squares(res) / (y.shape[0] * y.shape[1])
		else:
			model_loss = cp.sum(res) / (y.shape[0] * y.shape[1])

		# add regularization loss for epsilon and weights
		self.model_loss_exp = self.C * (self.nu * self.epsilon + model_loss) + \
			0.5 * cp.sum_squares(self.weights) / (self.dim_in * self.dim_out)

	def set_params(self, configs) -> bool:
		# for sharing configs, update lambda only
		self.warm_start = self.set_lambda_only(configs)

		# update configurations
		self.deconv_configs = configs

		if self.warm_start: # exit if warm start
			return True

		# set model parameters to optimize
		self.dim_in = self.deconv_configs['dim_in']
		self.dim_out = self.deconv_configs['dim_out']
		self.list_w_row = [cp.Variable((1, self.dim_out), nonneg=self.nonneg)
						   for _ in range(self.dim_in)]
		self.weights = cp.vstack(self.list_w_row)

		# set reconstruction loss
		self.set_model_loss()

		# set spatial loss
		self.set_sp_loss()

		return False
