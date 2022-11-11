"""
Transform spatial covariance into spatial loss
"""
from typing import List
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance

# optional imports for the topological loss
try:
	from topologylayer.nn import LevelSetLayer2D, PartialSumBarcodeLengths
except ImportError:
	warnings.warn("Package 'TopologyLayer' not installed. "
				  "The topological loss is not supported.",
				  category=ImportWarning)
	LevelSetLayer2D, PartialSumBarcodeLengths = None, None

from smoother.weights import SpatialWeightMatrix, _normalize_coords

def kl_loss_scipy(p, weights = None):
	"""Calculate KL divergence using scipy.

	Args:
		p (2D array): Probability distributions, num_group x num_spot.
			Each column is one discrete distribution over num_group groups.
		weights (2D array): Spatial weight matrix, num_spot x num_spot.
			If not None, will scale pairwise KL divergence by the weight matrix.

	Returns:
		kl (float): The mean pairwise KL divergence between spots, num_spot x num_spot.
	"""
	p = p + 1e-20 # for stability

	# calculate kl divergence
	kl = stats.entropy(p[:,:,None], p[:,None,:])

	# return loss
	return kl.mean() if weights is None else np.multiply(kl, weights).mean()


def kl_loss(p, weights = None):
	"""Calculate kl divergence using pytorch.

	Args:
		p (2D array): Probability distributions, num_group x num_spot.
			Each column is one discrete distribution over num_group groups.
		weights (2D array): Spatial weight matrix, num_spot x num_spot.
			If not None, will scale pairwise KL divergence by the weight matrix.

	Returns:
		kl (float): The mean pairwise KL divergence between spots, num_spot x num_spot.
	"""
	p = p + 1e-20 # for stability
	#p = 1.0*p / torch.sum(p, axis=0, keepdims=True)

	# calculate kl divergence
	ce = torch.matmul(p.T, torch.log(p)) # cross entropy, n x n
	kl = torch.diag(ce).reshape(-1,1) - ce # kl divergence, n x n

	# return loss
	return kl.mean() if weights is None else torch.mul(kl, weights).sum() / (weights != 0).sum()


def quadratic_loss(beta, inv_cov, group_scales, normalize = True):
	"""Calculate quadratic smoothing loss using torch.

	This is the main loss that transforms spatial covariance into spatial loss. It is equivalent to
	adding a multivariate normal prior with precision matrix `inv_cov` on `beta` in a regression
	setting. See model description for more details.

	Args:
		beta (2D tensor): Columns of regression coefficients, num_group x num_spot.
		inv_cov (3D tensor): Inverse covariance (precision) matrix of spatial variables of each group,
			(num_group or 1) x n x n. If the first dimension has length 1, all groups will have the
			same covariance structure.
		group_scales (1D tensor or float): Relative prior confidence of each group. The higher the
			confidence, the stronger the smoothing will be. If float, all groups will have
			the same confidence.
		normalize (bool): If True, normalize the likelihood over the size of beta for comparability.

	Returns:
		neg_loglik (float): Sum of quadratic loss, i.e., the negative likelihood of the corresponding
			multivariate normal prior on `beta`.
	"""
	if inv_cov.shape[0] == 1: # use the same spatial weight matrix for all celltypes
		neg_loglik = (torch.diag(beta @ inv_cov[0,:,:] @ beta.T) * group_scales).sum()
	else: # use different spatial weight matrices for different cell types
		neg_loglik = ((beta.unsqueeze(1) @ inv_cov @ beta.unsqueeze(2)).squeeze() * group_scales).sum()

	if normalize: # normalize the likelihood over the size of beta for better comparability
		neg_loglik /= (beta.shape[0] * beta.shape[1])

	return neg_loglik

def sparse_quadratic_loss(beta, inv_cov_2d_sp, group_scales, normalize = True):
	"""Calculate quadratic smoothing loss using torch.sparse.

	This function uses the sparsity of the inverse covariance matrix to speed up the calculation.

	Args:
		beta (2D tensor): Columns of regression coefficients, num_group x num_spot.
		inv_cov_2d_sp (2D sparse tensor): Sparse block diagonal inverse covariance (precision) matrix
			of each group. There are in total (num_group or 1) num_spot x num_spot blocks.
			If 1, all groups will have the same covariance structure.
		group_scales (1D tensor or float): Relative prior confidence of each group. The higher the
			confidence, the stronger the smoothing will be. If float, all groups will have
			the same confidence.
		normalize (bool): If True, normalize the likelihood over the size of beta for comparability.

	Returns:
		neg_loglik (float): Sum of quadratic loss, i.e., the negative likelihood of the corresponding
			multivariate normal prior on `beta`.
	"""
	if inv_cov_2d_sp.shape[0] == beta.shape[1]: # use the same spatial weight matrix for all celltypes
		neg_loglik = (torch.diag(beta @ torch.mm(inv_cov_2d_sp, beta.T)) * group_scales).sum()
	else: # use different spatial weight matrices for different cell types
		neg_loglik = ((beta * group_scales[:,None]).reshape(1, -1) @ \
			torch.mm(inv_cov_2d_sp, beta.reshape(-1, 1))).squeeze()

	if normalize: # normalize the likelihood over the size of beta for better comparability
		neg_loglik /= (beta.shape[0] * beta.shape[1])

	return neg_loglik


class SpatialLoss(nn.Module):
	"""Spatial loss.

	The spatial smoothing loss on a spatial random variable (num_group x num_spot).

	Attributes:
		prior (str): The prior spatial process model, can be one of 'sma','sar', 'car', 'icar'.
		spatial_weights (List[SpatialWeightMatrix]): Spatial weight matrix collection of
			length num_group or 1. If 1 then all groups will be subject to the same covariance.
		scale_weights (float): Smoothing effect size (i.e., `l` in SpatialWeightMatrix.get_inv_cov).
		use_sparse (bool): Whether to use sparse inverse covariance matrix in the calculation.
		standardize_cov (bool): Whether to standardize the covariance matrix to have same variance (1)
			  across locations. Only proper covariance can be standardized.
		inv_cov (3D tensor): Inverse covariance (precision) matrix of spatial variables of each group,
			(num_group or 1) x n x n. If the first dimension has length 1, all groups will have the
			same covariance structure.
		inv_cov_2d_sp (2D sparse tensor): Sparse block diagonal inverse covariance (precision) matrix.
		confidences (1D tensor or float): Relative prior confidence of each group. The higher the
			confidence, the stronger the smoothing will be. If float, all groups will have
			the same confidence.
	"""
	def __init__(self, prior, spatial_weights : List[SpatialWeightMatrix] = None,
				 scale_weights = 1, use_sparse = True, standardize_cov = False) -> None:
		super(SpatialLoss, self).__init__()

		# store configs
		self.prior = prior
		self.spatial_weights = spatial_weights
		self.scale_weights = scale_weights
		self.use_sparse = use_sparse
		self.standardize_cov = standardize_cov
		self._sanity_check()

		# calculate inverse covariance matrix
		if self.prior != 'kl':
			if isinstance(self.spatial_weights, SpatialWeightMatrix):
				# will use the same spatial weight matrix for all celltypes
				# self.inv_cov: 1 x n_spot x n_spot
				self.inv_cov = self.spatial_weights.get_inv_cov(
					self.prior, scale_weights, cached=False, standardize=self.standardize_cov).unsqueeze(0)
				self.confidences = torch.ones(1) # group-specific variance
			else: # use different spatial weight matrix per celltype
				# self.inv_cov: n_group x n_spot x n_spot
				self.inv_cov = torch.stack(
					[swm.get_inv_cov(self.prior, self.scale_weights, cached=False,
						standardize=self.standardize_cov)
					for swm in self.spatial_weights], dim=0)
				self.confidences = torch.ones(len(self.spatial_weights)) # group-specific variance

			# convert the inverse covariance matrix into a sparse matrix for efficient computation
			if self.prior != 'sma' and self.use_sparse:
				inv_cov_2d_sp = torch.block_diag(*[self.inv_cov[i] for i in range(self.inv_cov.shape[0])])
				self.inv_cov_2d_sp = inv_cov_2d_sp.to_sparse()

	def _sanity_check(self) -> None:
		"""Check whether the spatial loss is defined properly."""
		# check spatial prior model
		valid_priors = ['kl', 'sma', 'sar', 'isar', 'car', 'icar']
		if self.prior not in valid_priors:
			raise NotImplementedError(f"Spatial prior currently must be one of {valid_priors}")

		# check spatial weight matrix
		if self.spatial_weights is None and self.prior != 'kl':
			raise ValueError("A spatial weight matrix must be supplied "
							 "if you intend to apply spatial smoothing!")

	def estimate_confidence(self, ref_exp, st_exp, method = 'lr') -> None:
		"""Estimate the relative confidence for each group.

		The covariance matrix will be scaled accordingly.

		Args:
			ref_exp (2D tensor): Bulk expression signiture matrix, num_gene x num_group.
			st_exp (2D tensor): Spatial expression matrix, num_gene x num_spot.
			method (str): Method used to estimate variance.
		"""
		if method != 'lr':
			raise NotImplementedError

		# calculate linear regression solution
		solution = torch.linalg.lstsq(ref_exp, st_exp)[0] # num_group x num_spot

		# calculate variances, which indicate the relative confidence
		# the higher the variance, the stronger the smoothing will be
		self.confidences = torch.std(solution, dim = 1) / torch.std(solution, dim=1).max()


	def forward(self, coefs):
		"""Calculate spatial loss.

		Args:
			coefs (2D tensor): Columns of regression coefficients, num_group x num_spot.
		"""
		if self.prior == 'kl':
			p = torch.nn.functional.softmax(coefs, dim=0)
			return kl_loss(p = p, weights = self.spatial_weights.swm)

		if self.prior != 'sma' and self.use_sparse: # use sparse matrix
			return sparse_quadratic_loss(
				beta = coefs, inv_cov_2d_sp=self.inv_cov_2d_sp,
				group_scales=self.confidences, normalize=True
			)

		return quadratic_loss(
			beta = coefs, inv_cov=self.inv_cov,
			group_scales=self.confidences, normalize=True
		)

	def calc_cov_decay(self, coords, max_k = 50, step_k = 1, topk = False, cov_id = 0,
					   quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]):
		"""Calculate spatial covariance decay over distance.

		Covariance measured between k-nearest neighbors. If the number of covariance
		matrices (i.e. self.inv_cov.shape[0]) is larger than 1, use 'cov_id' to select
		the covariance matrix to use.

		Args:
			coords (2D tensor): Coordinates of spots, num_spot x 2.
			max_k: Maximum number of k in k-nearest neighbors to extract covariance.
			step_k: Step size of k.
			topk: Whether to use top-k neighbors. If True, for each k, the covariance
				distribution represents all j-th nearest neighbors where j <= k.
			cov_id (int): Index of the covariance matrix to use.
			quantiles: Quantiles of covariance distribution to report.

		Returns:
			corr_decay_quantiles_df (pd.DataFrame): Correlation decay quantiles.
			var_quantiles_df (pd.DataFrame): Per-spot variance quantiles.
		"""
		# each element in self.inv_cov specifies a different covariance matrix
		# if self.inv_cov.shape[0] > 1, use cov_id to select the covariance matrix
		assert cov_id < self.inv_cov.shape[0] # 1 or n_group

		# calculate covariance according to the spatial loss setting
		if self.prior == 'kl':
			raise ValueError("Covariance is not supported for KL prior.")
		elif self.prior == 'sma':
			# calculate covariance matrix directly to avoid numeric error
			if isinstance(self.spatial_weights, SpatialWeightMatrix):
				weights = self.spatial_weights.swm_scaled
			else:
				weights = self.spatial_weights[cov_id].swm_scaled

			# covariance
			cov = torch.eye(weights.shape[0]) + self.scale_weights * (weights + weights.T) + \
				(self.scale_weights ** 2) * torch.matmul(weights, weights.T)
		else:
			try:
				cov = torch.cholesky_inverse(torch.linalg.cholesky(self.inv_cov[cov_id]))
			except RuntimeError as exc:
				raise RuntimeError(f"The current loss ({self.prior}, l={self.scale_weights}) "
									"contains an improper spatial covariance structure. "
									"Please use a different spatial prior or scale "
									"if you intend to calculate covariance decay.") from exc

		# separate correlation and variance
		var = torch.diagonal(cov)
		inv_sd = torch.diagflat(var ** -0.5)
		corr = inv_sd @ cov @ inv_sd

		# calculate pairwise distances
		coords_norm = _normalize_coords(coords, min_zero=True)
		dist = torch.tensor(distance.squareform(distance.pdist(coords_norm, metric='euclidean')))

		# calculate correlation decay
		corr_decay_k = []
		for k in range(1, max_k + 1, step_k):
			# calculate binary matrix indexing k-nearest neighbors
			if topk:
				_, knn_id = torch.topk(dist, k = k, dim = 1, largest = False)
			else:
				_, knn_id = torch.kthvalue(dist, k, dim = 1)
				knn_id = knn_id[:,None]

			knn_index = torch.zeros(dist.shape[0], dist.shape[0])
			knn_index = knn_index.scatter_(1, knn_id, 1)

			# extract pairwise correlations and calculate quantiles
			corr_vec = (corr * knn_index)[knn_index != 0]

			corr_decay_k.append(
				torch.nanquantile(
					corr_vec, torch.tensor(quantiles)
				)
			)

		# calculate variance quantile distribution
		var_quantiles = torch.nanquantile(var, torch.tensor(quantiles))

		# convert to dataframe
		quantile_names = [f'Q{str(int(q * 100))}' for q in quantiles]
		corr_decay_quantiles_df = pd.DataFrame(torch.stack(corr_decay_k, dim=0).numpy(),
											   columns=quantile_names)
		corr_decay_quantiles_df['k'] = range(1, max_k + 1, step_k)
		var_quantiles_df = pd.DataFrame(var_quantiles.numpy().reshape(1,-1),
										columns=quantile_names)

		return corr_decay_quantiles_df, var_quantiles_df


class TopLoss(nn.Module):
	"""Topological loss.

	The topological smoothing loss on a spatial random variable (num_group x num_spot).

	Attributes:
		betti_prior (dict): A nested dictionary that specifies topological priors for each group.
			{group_id: {betti_k : expectation}}.
		coords_xy (2D array): Spatial coordinates (2D), num_spot x 2.
		pdfn (topologylayer.nn.LevelSetLayer2D): Super-level set layer.
		topfn (dict): Topological features for each group.
			{group_id: {betti_k: topologylayer.nn.PartialSumBarcodeLengths}}.
	"""
	def __init__(self, betti_priors : dict, coords_xy):
		super(TopLoss, self).__init__()

		# store configs
		self.betti_priors = betti_priors
		self.coords_xy = np.array(coords_xy)
		self._num_spot = self.coords_xy.shape[0]
		self._dim_x, self._dim_y = self.coords_xy.max(0) + 1

		# no topological prior applied
		if self.betti_priors is None:
			return None

		if LevelSetLayer2D is None:
			print("Please install the package 'TopologyLayer' before "
				  "using the topological loss.")
			return None

		# initialize topological layers
		self.pdfn = LevelSetLayer2D(size=(self._dim_x, self._dim_y),  sublevel=False)
		self.topfn = {} # featurization layers for each group
		for group_id in self.betti_priors:
			self.topfn[group_id] = {
				# should always skip the first one barcode that
				# represents captured vs uncaptured regions
				betti_k : PartialSumBarcodeLengths(dim = betti_k, skip=prior)
				for betti_k, prior in betti_priors[group_id].items()
			}
		return None

	def reshape_coefs(self, beta, pad_value : float = None):
		"""Reshape 2D coefficients into 3D and pad with zeros.

		The original coefficient matrix has 1 location axis (the first dimension)
		while the transformed matrix will have 2 location axes (x and y, the first two).
		Spatial arrangement specified in `self.coords_xy`.

		Args:
			beta (2D array): Columns of regression coefficients, num_group x num_spot.
			pad_value (float): Value of coefficients at unobserved regions.
		"""
		# pad with the smallesy coef
		if pad_value is None:
			pad_value = beta.min()

		num_group = beta.shape[0]
		beta_3d = torch.ones((self._dim_x, self._dim_y, num_group)) * pad_value
		beta_3d[self.coords_xy[:,0], self.coords_xy[:,1]] = beta.T
		return beta_3d

	def forward(self, coefs):
		"""Calculate topological loss.

		Args:
			coefs (2D tensor): Columns of regression coefficients, num_group x num_spot.
		"""
		if self.betti_priors is None:
			return 0
		# reshape coefficients
		beta_3d = self.reshape_coefs(coefs)

		# calculate topological loss
		loss = 0
		for group_id in self.betti_priors:
			dgms_g = self.pdfn(beta_3d[:,:,group_id])
			for betti_k in self.topfn[group_id]:
				loss += self.topfn[group_id][betti_k](dgms_g)

		return loss / (self._num_spot ** 0.5 * len(self.betti_priors))


class ContrastiveSpatialLoss(SpatialLoss):
	"""Spatial loss for contrastive learning.

	The spatial loss that maximizes similarity of a spatial random variable (num_group x num_spot)
	over true neighbors while minimizing similarity over randomly generated neighbors. See
	Zhu, Hao, Ke Sun, and Peter Koniusz. "Contrastive laplacian eigenmaps."
	 Advances in Neural Information Processing Systems 34 (2021).
	https://arxiv.org/abs/2201.05493

	By default, the inverse covariance matrix generated from `prior = 'icar'` and
	 `scale_weights = 1` is the graph laplacian. Set `standardize_cov = True` to normalize
	the graph laplacian.

	Attributes: See `SpatialLoss`.
		num_perm (int): Number of negative graphs (generated by randomly shuffling spots).
		neg2pos_ratio (float): Relative importance of negative samples to positive samples.
		lower_bound (float): Lower bound of the loss in case that cov is not positive semi-definite.
		check_neg_samples (bool): Whether to check if resulting cov is positive definite.
	"""
	def __init__(self, prior = 'icar', spatial_weights: List[SpatialWeightMatrix] = None,
				 scale_weights=1, use_sparse=True, standardize_cov=True,
				 num_perm = 10, neg2pos_ratio = 0.5, lower_bound = -1,
				 check_positive_definite = False) -> None:
		# recommend using the defaul settings, i.e., `prior = 'icar'`, `scale_weights = 1`
		# and standardize_cov = True
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore')
			super().__init__(prior, spatial_weights, scale_weights, use_sparse, standardize_cov)

		self.num_perm = num_perm
		self.neg2pos_ratio = neg2pos_ratio
		self.lower_bound = lower_bound

		if check_positive_definite:
			self.check_positive_definite()

	def check_positive_definite(self, n_tests = 10):
		"""Check if the covariance matrix is positive semi-definite."""
		for _ in range(n_tests):
			inv_cov_perm = torch.zeros_like(self.inv_cov)
			for _ in range(self.num_perm): # randomly permute spots
				idx_perm = torch.randperm(self.inv_cov.shape[-1])
				inv_cov_perm += self.inv_cov[:,:,idx_perm][:,idx_perm,:]
			inv_cov = self.inv_cov - self.neg2pos_ratio / self.num_perm * inv_cov_perm

			try: # check if the covariance matrix is positive definite
				torch.linalg.cholesky(inv_cov)
			except RuntimeError:
				eigvals = np.sort(torch.linalg.eigvals(inv_cov))
				if not (eigvals[0].real > -1e-7).all():
					warnings.warn(
						"The constrastive covariance is not positive semi-definite. "
						f"you may want to use a smaller neg2pos_ratio than {self.neg2pos_ratio}."
					)
					break

	def forward(self, coefs):
		"""Calculate contrastive spatial loss."""
		# positive loss to maximize similarity with true neighbors
		pos_loss = super().forward(coefs)

		# negative loss to minimize similarity with randomly generated neighbors
		neg_loss = 0
		for _ in range(self.num_perm): # randomly permute spots
			idx_perm = torch.randperm(coefs.shape[1])
			neg_loss += super().forward(coefs[:, idx_perm])
		neg_loss /= self.num_perm

		# total contrastive loss clamped to lower bound
		total_loss = pos_loss - self.neg2pos_ratio * neg_loss
		return torch.clamp(total_loss, min=self.lower_bound)
