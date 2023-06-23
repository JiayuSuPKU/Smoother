"""
Impute spatial feature of interest with spatial loss
"""
import warnings
from timeit import default_timer as timer
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from smoother.losses import SpatialLoss

# optional imports for convex optimization
try:
	import cvxpy as cp
	from cvxpy.atoms.affine.wraps import psd_wrap
except ImportError:
	warnings.warn("Package 'cvxpy' not installed so that "
				  "Convex optimization solvers are not supported.",
				  category=ImportWarning)
	cp, psd_wrap = None, None

class ImputeModel():
	"""Class for different implementations of spatial imputation.

	This class is designed for coorperative inheritance (mixin).

	Attributes:
		y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
		spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
			spatial coordinates containing both observed (the first n_obs rows)
			and missing spots (the rest rows).
		fixed_obs (bool): Whether to allow updates in the observed data.
		nonneg (bool): Whether to enforce nonnegativity on spatial features.
		n_feature (int): Number of spatial features to process.
		n_all (int): Number of total spots. n_all = n_obs + n_missing.
		n_obs (int): Number of observed spots.
		n_missing (int): Number of missing spots.
		var_obs : Observed spatial feature matrix, n_obs x n_feature.
		var_missing : Missing spatial feature matrix to impute, n_missing x n_feature.
		var_all : Concatenated observed and missing spatial feature matrix, n_all x n_feature.
		impute_configs (dict): Dictionary of imputation configurations.
		impute_time (float): Time spent on deconvolution.
		impute_flag (bool): Whether the imputation was successful.
	"""
	def __init__(self, y_obs, spatial_loss_all : SpatialLoss,
				 fixed_obs = True, nonneg = False, **kwargs):
		"""Initialize ImputeModel object."""
		# for the ImputeModel class to be compatible with nn.module,
		# here we need to forward all unused arguments
		super().__init__(**kwargs)

		self.y_obs = None
		self.spatial_loss_all = None
		self.fixed_obs = None # whether to allow updates in the observed data
		self.nonneg = None # whether to enforce nonnegativity on spatial features

		self.n_feature = None # number of spatial features to process
		# the first n_obs spots modelled in spatial_loss are the observed data
		# and the rest are the missing data to impute
		self.n_obs = None # number of observed spots
		self.n_all = None # number of total spots
		self.n_missing = None # number of missing spots

		# set model parameters
		self.var_obs = None # (n_obs, n_feature)
		self.var_missing = None # (n_missing, n_feature)
		self.var_all = None # (n_all, n_feature)
		self.set_params(y_obs, spatial_loss_all, fixed_obs, nonneg)

		# imputation attributes
		self.impute_configs = None
		self.impute_time = 0

	def set_params(self, y_obs, spatial_loss_all, fixed_obs = True, nonneg = False):
		"""Set configuration parameters and initialize variables."""
		# currently only support one covariance structure
		if len(spatial_loss_all.inv_cov) > 1:
			raise NotImplementedError

		# store raw data and spatial loss
		self.y_obs = y_obs
		self.spatial_loss_all = spatial_loss_all
		self.fixed_obs = fixed_obs # whether to allow updates in the observed data

		# sanity check for the nonneg argument
		if nonneg and self.y_obs.min() < 0:
			raise ValueError("Nonnegativity is enforced but the observed data contains negative values.")
		self.nonneg = nonneg # whether to enforce nonnegativity on spatial features

		self.n_feature = y_obs.shape[1] # number of spatial features to process
		# the first n_obs spots are the observed data
		# and the rest are the missing data to impute
		self.n_obs = y_obs.shape[0] # number of observed spots
		self.n_all = spatial_loss_all.inv_cov[0].shape[0] # number of total spots
		self.n_missing = self.n_all - self.n_obs # number of missing spots

		# initialze variables to impute()
		self.set_variables()

	def set_variables(self):
		"""Set variables to impute."""
		raise NotImplementedError

	def get_recon_loss(self):
		"""Get reconstruction loss."""
		raise NotImplementedError

	def get_sp_loss(self):
		"""Get spatial loss (weighted by lambda_spatial_loss)."""
		raise NotImplementedError

	def get_results(self):
		"""Get spatial features after imputation."""
		raise NotImplementedError


class ImputeTorch(ImputeModel, nn.Module):
	"""Imputation solver implemented using pytorch.

	Attributes:
		var_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
		var_missing (2D tensor): Missing spatial feature matrix to impute, n_missing x n_feature.
		var_all (2D tensor): Concatenated observed and missing spatial feature matrix, n_all x n_feature.
		See super class attributes for additional arguments.
	"""
	def __init__(self, y_obs, spatial_loss_all : SpatialLoss,
				 fixed_obs = True, nonneg = False, **kwargs) -> None:
		"""Initialize and run the imputation.

		Args:
			y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
			spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
						spatial coordinates containing both observed (the first n_obs rows)
						and missing spots (the rest rows).
			See class attributes for additional arguments.
		"""
		super().__init__(y_obs, spatial_loss_all, fixed_obs, nonneg)

		# run imputation
		self.impute_flag = self.impute(**kwargs)

	def set_variables(self):
		"""Initialize torch parameters to estimate."""
		# set torch variables
		self.var_obs = nn.parameter.Parameter(self.y_obs.clone(), requires_grad = (not self.fixed_obs))
		self.var_missing = nn.parameter.Parameter(
			torch.ones(self.n_missing, self.n_feature) * self.y_obs.mean(dim=0, keepdim=True),
			requires_grad = True)
		self.var_all = torch.concat([self.var_obs, self.var_missing], dim = 0)

	def forward(self, y_obs, **kwargs):
		"""Run imputation for a new set of spatial features."""
		self.set_params(y_obs, self.spatial_loss_all, self.fixed_obs, self.nonneg)
		self.impute_flag = self.impute(**kwargs)
		return self.var_all.detach()

	def get_recon_loss(self):
		"""Calculate loss of deviation from observation.

		If the observed data is not fixed, then the loss is used to make sure data does not deviate
		too much from the observed value after imputation.
		"""
		if self.fixed_obs:
			return 0
		else:
			loss_fn = torch.nn.MSELoss() # mean squared error
			if self.nonneg:
				return loss_fn(self.y_obs, self.var_obs.abs())
			else:
				return loss_fn(self.y_obs, self.var_obs)

	def get_sp_loss(self):
		"""Calculate loss of spatial smoothness."""
		l_sp_loss = self.impute_configs.get('lambda_spatial_loss')
		sp_loss = self.impute_configs.get('spatial_loss')
		# calculate spatial loss
		if self.nonneg:
			return l_sp_loss * sp_loss(self.var_all.abs().T)
		else:
			return l_sp_loss * sp_loss(self.var_all.T)

	def get_results(self):
		"""Get spatial features of all spots after imputation.

		The first n_obs spots are the observed spots and the rest are the missing ones.
		If fixed_obs is True, then the returned observed data is not updated.
		"""
		return self.var_all.detach()

	def final_sanity_check(self):
		if self.nonneg:
			with torch.no_grad():
				self.var_obs.copy_(self.var_obs.abs())
				self.var_missing.copy_(self.var_missing.abs())
				self.var_all.copy_(self.var_all.abs())

	def impute(self, lambda_spatial_loss = 1.0, lr = 1e-3, max_epochs = 1000,
			   patience = 10, tol = 1e-5, verbose = True, quiet = False) -> bool:
		"""Run imputation algorithm.

		Args:
			lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
			lr (float): Learning rate.
			max_epochs (int): Maximum number of training epochs. If -1, iterate until
				convergence (`d_loss` < tol).
			patient (int): Number of epochs to wait for the loss to decrease before stopping.
			tol (float): Tolerance of loss convergence.
			verbose (bool): If True, print out loss while training.
			quiet (bool): If True, no output printed.

		Returns:
			bool: True if imputation is successful.
		"""
		self.impute_configs = {
			'y_obs':self.y_obs, 'spatial_loss':self.spatial_loss_all,
			'n_feature':self.n_feature, 'n_obs':self.n_obs, 'n_missing':self.n_missing,
			'fixed_obs':self.fixed_obs, 'nonneg':self.nonneg,
			'lambda_spatial_loss':lambda_spatial_loss,
			'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
			'verbose':verbose, 'quiet':quiet
		}

		if self.n_missing == 0 and self.fixed_obs:
			warnings.warn("Imputation stopped: Observation is fixed with no missing position.")
			return False

		# start timer
		t_start = timer()

		# initialize optimizer
		optimizer = torch.optim.Adam(self.parameters(), lr=lr)

		# start training
		self.train()

		# set iteration limits
		epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
		max_epochs = 10000 if max_epochs == -1 else max_epochs
		patience = patience if patience > 0 else 1

		while epoch < max_epochs and (d_loss >= tol or (d_loss < 0 and patience > 0)):
			optimizer.zero_grad()

			# calculate reconstruction loss
			loss = self.get_recon_loss()

			# store the reconstruction loss
			if self.fixed_obs:
				recon_loss = 0.0
			else:
				recon_loss = loss.detach().item()

			# add spatial loss
			sp_loss = self.get_sp_loss()
			loss += sp_loss
			sp_loss = sp_loss.detach().item()

			# backpropagate and update weights
			loss.backward()
			optimizer.step()

			# update the merged vector
			self.var_all = torch.concat([self.var_obs, self.var_missing], dim = 0)

			# check convergence
			epoch += 1
			d_loss = prev_loss - loss.detach().item()
			if d_loss < 0: # if loss increases
				patience -= 1

			# print out loss
			prev_loss = loss.detach().item()

			if (verbose and not quiet) and epoch % 10 == 0:
				print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. '
					f'(recon) {recon_loss:.4f}. (spatial) {sp_loss:.4f}.')

		# make sure constraints are satisfied
		self.final_sanity_check()

		t_end = timer()

		# save runtime
		self.impute_time = t_end - t_start

		if not quiet: # print final message
			print(f'=== Time {self.impute_time : .2f}s. Total epoch {epoch}. '
				f'Final loss: (total) {prev_loss : .3f}. (spatial) {sp_loss : .3f}.')

		# if d_loss >= tol:
		if epoch == max_epochs:
			warnings.warn("Fail to converge. Try to increase 'max_epochs'.")
			return False

		# Depending on the weight matrix, sometimes the inverse covariance matrix can
		# have negative eigenvalues, be careful about the regression coefficient
		if prev_loss < 0:
			warnings.warn(
				"Negative loss occurred because the inverse covariance matrix is not positive semidefinite. "
				"Try to restrict 'spatial_weights' or use a smaller 'smooth_l'."
			)
			return False

		return True


class ImputeConvex(ImputeModel):
	"""Imputation solver implemented using cvxpy.

	Attributes:
		var_obs : Observed spatial feature matrix, n_obs x n_feature.
		var_missing : Missing spatial feature matrix to impute, n_missing x n_feature.
		var_all : Concatenated observed and missing spatial feature matrix, n_all x n_feature.
		recon_loss_exp (cp.Expression): Reconstruction loss expression.
		sp_loss_exp (cp.Expression): Spatial loss expression.
		See super class attributes for additional arguments.
	"""
	def __init__(self, y_obs, spatial_loss_all : SpatialLoss,
				 fixed_obs = True, nonneg = False, **kwargs) -> None:
		"""Initialize and run the imputation.

		Args:
			y_obs (2D tensor): Observed spatial feature matrix, n_obs x n_feature.
			spatial_loss_all (SpatialLoss): Spatial loss object build from the combined
						spatial coordinates containing both observed (the first n_obs rows)
						and missing spots (the rest rows).
			See class attributes for additional arguments.
		"""
		self.recon_loss_exp = None # cvxpy expression for reconstruction loss
		self.spatial_loss_exp = None # cvxpy expression for spatial loss
		super().__init__(y_obs, spatial_loss_all, fixed_obs, nonneg)

		# run imputation
		self.impute_flag = self.impute(**kwargs)

	def set_variables(self):
		"""Initialize parameters to estimate."""
		# set cvxpy variables
		if not self.fixed_obs:
			self.var_all = cp.Variable((self.n_all, self.n_feature), nonneg=self.nonneg)
			self.var_obs = self.var_all[0:self.n_obs, :]
			self.var_missing = self.var_all[self.n_obs:, :]
		else:
			self.var_missing = cp.Variable((self.n_missing, self.n_feature), nonneg=self.nonneg)
			self.var_obs = self.y_obs
			self.var_all = cp.vstack([self.var_obs, self.var_missing])

		# set cvxpy expressions
		self.set_recon_loss()
		self.set_sp_loss()

	def set_recon_loss(self):
		"""Set loss expression of deviation from observation.

		If the observed data is not fixed, then the loss is used to make sure data does not deviate
		too much from the observed value after imputation.
		"""
		if self.fixed_obs:
			self.recon_loss_exp = cp.Constant(0.0)
		else:
			self.recon_loss_exp = cp.sum_squares(self.var_obs - self.y_obs) / (self.n_obs * self.n_feature)

	def set_sp_loss(self):
		"""Set spatial loss expression (before weighted by lambda_spatial_loss)."""
		sp_loss = self.spatial_loss_all
		self.spatial_loss_exp = 0

		# extract inverse covariance matrix
		if not sp_loss.standardize_cov:
			raise ValueError("Spatial covariance must be standardized.")
		inv_cov = sp_loss.inv_cov[0].coalesce()

		# use scipy sparse matrix to speed up computation
		if sp_loss.use_sparse:
			inv_cov = psd_wrap(coo_matrix(
				(inv_cov.values().numpy(), inv_cov.indices().numpy()), inv_cov.shape
			))

		for i in range(self.n_feature):
			self.spatial_loss_exp += \
				cp.quad_form(self.var_all[:,i], inv_cov)

		self.spatial_loss_exp /= (self.n_all * self.n_feature)

	def get_recon_loss(self):
		"""Calculate loss of deviation from observation.

		If the observed data is not fixed, then the loss is used to make sure data does not deviate
		too much from the observed value after imputation.
		"""
		return self.recon_loss_exp

	def get_sp_loss(self):
		"""Calculate loss of spatial smoothness (weighted by lambda_spatial_loss)."""
		l_sp_loss = self.impute_configs.get('lambda_spatial_loss')
		return self.spatial_loss_exp * l_sp_loss

	def get_results(self):
		"""Get spatial features of all spots after imputation.

		The first n_obs spots are the observed spots and the rest are the missing ones.
		If fixed_obs is True, then the returned observed data is not updated.
		"""
		return self.var_all.value

	def impute(self, lambda_spatial_loss = 1.0,
			   verbose = False, quiet = False, solver = None, **kwargs) -> bool:
		"""Run imputation algorithm.

		Args:
			lambda_spatial_loss (float): Specifies the strength of the spatial smoothing.
			verbose (bool): If True, print out loss while training.
			quiet (bool): If True, no output printed.

		Returns:
			bool: True if imputation is successful.
		"""
		self.impute_configs = {
			'y_obs':self.y_obs, 'spatial_loss':self.spatial_loss_all,
			'n_feature':self.n_feature, 'n_obs':self.n_obs, 'n_missing':self.n_missing,
			'fixed_obs':self.fixed_obs, 'nonneg':self.nonneg,
			'lambda_spatial_loss':lambda_spatial_loss,
			'verbose':verbose, 'quiet':quiet
		}

		if self.n_missing == 0 and self.fixed_obs:
			warnings.warn("Imputation stopped: Observation is fixed with no missing position.")
			return False

		# start timer
		t_start = timer()

		# initialize optimization problem
		self.obj = cp.Minimize(self.get_recon_loss() + self.get_sp_loss())
		self.problem = cp.Problem(self.obj)

		# pass solver arguments
		print_sol_out = verbose and (not quiet)
		self.problem.solve(solver = solver, verbose = print_sol_out, **kwargs)

		# check problem status
		if not self.problem.status == cp.OPTIMAL:
			if self.problem.status == cp.UNBOUNDED:
				raise ValueError('Solver encountered numeric issues. '
								 'Try torch-based models or reduce number of genes.')
			warnings.warn(f"Problem status: {self.problem.status}.")

		# extract losses
		total_loss = self.obj.value
		recon_loss = self.get_recon_loss().value.item()
		sp_loss = self.get_sp_loss().value.item()

		t_end = timer()

		# save runtime
		self.impute_time = t_end - t_start

		if not quiet: # print final message
			print(f'=== Time {self.impute_time : .2f}s. '
				f'Loss: (total) {total_loss : .3f}, '
				f'(recon) {recon_loss : .3f}, '
				f'(spatial) {sp_loss : .3f}')
		return True
