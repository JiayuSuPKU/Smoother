"""
Calculate spatial weight matrix from coordinates, histology images, and transcriptomic data
"""
from collections import defaultdict
from functools import partial
import warnings
import numpy as np
import torch
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from smoother.utils import *

def coordinate_to_weights_knn_sparse(coords, k = 6, symmetric = True, row_scale = True):
	"""Calculate spatial weight matrix using k-nearest neighbours (sklearn).

	Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
	interactions among k-nearest neighbours. Sparse tensor version.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		k (int): Number of nearest neighbours to keep.
		symmetric (bool): If True only keep mutual neighbors.
		row_scale (bool): If True scale row sum of the spatial weight matrix to 1.

	Returns:
		weights: A sparse 2D tensor containing spatial weights, num_spot x num_spot.
	"""

	# use sklearn to find the k-nearest neighbors
	nbrs = NearestNeighbors(n_neighbors = k + 1, metric='euclidean').fit(coords)
	indices = nbrs.kneighbors(coords, return_distance=False)

	# convert to binary sparse tensor (no self interaction)
	# w_i.shape = (2, num_spot * k)
	w_i = torch.tensor([[i, j] for i in range(len(indices)) for j in indices[i][1:]]).T
	w_v = torch.ones(w_i.shape[1])
	weights = torch.sparse_coo_tensor(w_i, w_v, (len(indices), len(indices)), dtype=torch.float32).coalesce()

	if symmetric: # make weight matrix symmetric by keeping only mutual neighbors
		weights = weights * weights.transpose(0, 1)

	# set diagonal to 1 for spots with no neighbors
	id_no_neighbors = torch.where(torch.sparse.sum(weights, 1).to_dense() == 0)[0]
	weights = weights + torch.sparse_coo_tensor(
		id_no_neighbors.repeat(2, 1),
		torch.ones(len(id_no_neighbors)),
		(len(indices), len(indices)), dtype=torch.float32
	)
	weights = weights.coalesce()

	if row_scale: # scale row to sum to 1
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		weights.values()[:] = weights.values() / row_sum[weights.indices()[0]]

	return weights

def coordinate_to_weights_dist_sparse(coords, scale_coords = True, radius_cutoff = 1.0,
							   		band_width = 0.1, dist_metric = 'euclidean', row_scale = True):
	"""Calculate spatial weight matrix using distance band (sklearn).

	Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
	interactions among neighbours defined by the distane threshold. Sparse tensor version.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		scale_coords (bool): If True, scale coordinates to [0, 1].
		radius_cutoff (float): Distance threshold (in the same unit as the coords input).
		band_width (float): Specify the width of the Gaussian kernel, which is proportional
			to the inverse rate of weight distance decay.
		dist_metric (str): Distance metric.
		row_scale (bool): If True scale row sum of the spatial weight matrix to 1.

	Returns:
		weights: A sparse 2D tensor containing spatial weights, num_spot x num_spot.
	"""
	if scale_coords: # rescale coordinates to [0, 1]
		coords_norm, coords_scale = normalize_minmax(coords, min_zero=True, return_scale=True)
		radius_cutoff = radius_cutoff / coords_scale
	else:
		coords_norm = coords.clone()

	# use sklearn to find the nearest neighbors
	nbrs = NearestNeighbors(radius=radius_cutoff, metric=dist_metric).fit(coords_norm)
	distances, indices = nbrs.radius_neighbors(coords_norm, return_distance=True)

	# convert to sparse tensor (no self interaction)
	w_i = torch.tensor([[i, j] for i in range(len(indices)) for j in indices[i] if j!= i]).T
	w_v = torch.tensor([distances[i][ind] for i in range(len(indices)) for ind, j in enumerate(indices[i]) if j!= i])
	w_v = torch.exp(- w_v.pow(2) / (2 * band_width ** 2))
	weights = torch.sparse_coo_tensor(w_i, w_v, (len(indices), len(indices)), dtype=torch.float32).coalesce()

	# set diagonal to 1 for spots with no neighbors
	id_no_neighbors = torch.where(torch.sparse.sum(weights, 1).to_dense() == 0)[0]
	weights = weights + torch.sparse_coo_tensor(
		id_no_neighbors.repeat(2, 1),
		torch.ones(len(id_no_neighbors)),
		(len(indices), len(indices)), dtype=torch.float32
	)
	weights = weights.coalesce()

	if row_scale: # scale row to sum to 1
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		weights.values()[:] = weights.values() / row_sum[weights.indices()[0]]

	return weights

def coordinate_to_weights_knn_dense(coords, k = 6, symmetric = True, row_scale = True):
	"""Calculate spatial weight matrix using k-nearest neighbours.

	Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
	interactions among k-nearest neighbours.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		k (int): Number of nearest neighbours to keep.
		symmetric (bool): If True only keep mutual neighbors.
		row_scale (bool): If True scale row sum of the spatial weight matrix to 1.

	Returns:
		weights: A 2D tensor containing spatial weights, num_spot x num_spot.
	"""
	# calculate pairwise distances
	coords_norm = normalize_minmax(coords, min_zero=True)
	dist = torch.tensor(distance.squareform(distance.pdist(coords_norm, metric='euclidean')))

	# find k-nearest neibours for each location
	_, knn_id = torch.topk(dist, k = k + 1, dim = 1, largest=False)

	# calculate binary weight matrix
	weights = torch.zeros(coords.shape[0], coords.shape[0])
	weights = weights.scatter_(1, knn_id, 1)

	# remove self interaction
	weights = weights.fill_diagonal_(0)

	if symmetric: # make weight matrix symmetric by keeping only mutual neighbors
		weights = (weights * weights.T)

	# set diagonal to 1 for spots with no neighbors
	id_no_neighbors = torch.sum(weights, 1) == 0
	weights[id_no_neighbors, id_no_neighbors] = 1

	if row_scale: # scale row to sum to 1
		weights = weights / weights.sum(1, keepdim=True)

	return weights

def coordinate_to_weights_dist_dense(coords, scale_coords = True, q_threshold = 0.001,
									 band_width = 0.1, dist_metric = 'euclidean', row_scale = True):
	"""Calculate spatial weight matrix using distance band.

	Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
	interactions among neighbours defined by the distane threshold.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		scale_coords (bool): If True, scale coordinates to [0, 1].
		q_threshold (float): Distance quantile threshold. Number of nonzero entries in the
			weight matrix (edges) = num_spot^2 * q_threshold.
		band_width (float): Specify the width of the Gaussian kernel, which is proportional
			to the inverse rate of weight distance decay.
		dist_metric (str): Distance metric.
		row_scale (bool): If True scale row sum of the spatial weight matrix to 1.

	Returns:
		weights: A 2D tensor containing spatial weights, num_spot x num_spot.
	"""
	if scale_coords: # rescale coordinates to [0, 1]
		coords_norm = normalize_minmax(coords, min_zero=True)
	else:
		coords_norm = coords.clone()

	# calculate pairwise distances
	dist = torch.tensor(distance.squareform(distance.pdist(coords_norm, metric=dist_metric)))

	# filter out large distances
	dist_keep = dist <= np.quantile(dist.reshape(-1), q_threshold)

	# transform distances into weights
	weights = torch.exp(-dist.pow(2) / (2 * band_width ** 2))
	weights = weights.masked_fill_(~dist_keep, 0)

	# remove self interaction
	weights = weights.fill_diagonal_(0)

	# set diagonal to 1 for spots with no neighbors
	id_no_neighbors = torch.sum(weights, 1) == 0
	weights[id_no_neighbors, id_no_neighbors] = 1

	if row_scale: # scale row to sum to 1
		weights = weights / weights.sum(1, keepdim=True)

	return weights

def sparse_weights_to_inv_cov(weights, model, rho = 1, standardize = False, return_sparse = False):
	"""Convert a spatial weight matrix to an inverse covariance matrix. Sparse version.

	Calculate the covariance structure using spatial weights. Different spatial process models
	impose different structures. Check model descriptions for more details.

	Args:
		weights (Sparse tensor): Spatial weight matrix (sparse), num_spot x num_spot.
		model (str): Spatial process model to use, can be one of 'sma','sar', 'isar', 'car', 'icar'.
		rho (float): Spatial autocorrelation parameter.
		standardize (bool): If True, return the standardized inverse covariance matrix (inv_corr).
		return_sparse (bool): If True, return a sparse tensor. Note that the inverse covariance matrix of
  			the SMA model is not sparse in general.

	Returns:
		inv_cov (2D tensor): An inverse covariance (precision) matrix, num_spot x num_spot.
	"""
	# check spatial priors
	valid_priors = ['sma', 'sar', 'isar', 'car', 'icar']
	if model not in valid_priors:
		raise ValueError(f"Spatial smoothing prior currently must be one of {valid_priors}")

	weights = weights.clone()
	num_spot = weights.shape[0] # number of spatial locations

	# make sure weights to be nonnegative for stability
	weights.values()[:] = weights.values().clamp(min=0)

	# sparse identity matrix
	sparse_i = torch.sparse_coo_tensor(
		torch.arange(num_spot).repeat(2,1),
		torch.ones(num_spot),
		weights.shape, dtype=torch.float32
	)

	# calculate inverse covariance matrix according to the weights
	if model == 'sma': # spatial moving average model
		if return_sparse:
			# the inverse covariance matrix of the SMA model is not sparse in general
			raise NotImplementedError("Sparse inverse covariance matrix is currently not available for SMA model.")

		# not recommended for large number of spots
		if num_spot > 10000:
			warnings.warn(
				f"Caution: SMA model of {num_spot} spots requires large dense matrix inversion."
			)

		cov = sparse_i + rho * (weights + weights.transpose(0,1)) + \
			(rho**2) * torch.matmul(weights, weights.transpose(0,1)) # covariance matrix
		try:
			inv_cov = torch.linalg.inv(cov.to_dense()) # inverse covariance, or precision matrix
		except RuntimeError:
			warnings.warn(
	   			"The covariance matrix is singular, return pseudo inverse instead. "
		 		"You may want to adjust the smoothing parameter l."
			)
			inv_cov = torch.linalg.pinv(cov.to_dense())

	elif model == 'sar': # simultaneous auto-regressive model
		inv_cov = torch.matmul(sparse_i - rho * weights.transpose(0,1),
							   sparse_i - rho * weights)

	elif model == 'isar': # intrinsic simultaneous auto-regressive model
		# row-scale the weights matrix
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		weights.values()[:] = weights.values() / row_sum[weights.indices()[0]]

		inv_cov = torch.matmul(sparse_i - rho * weights.transpose(0,1),
							   sparse_i - rho * weights)

	elif model == 'car': # conditional auto-regressive model
		inv_cov = sparse_i - rho * weights

	elif model == 'icar': # intrinsic conditional auto-regressive model
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		inv_cov = torch.sparse_coo_tensor(
			torch.arange(num_spot).repeat(2,1),
			row_sum,
			weights.shape, dtype=torch.float32
		) - rho * weights

	if standardize:
		# not recommended for large number of spots
		if num_spot > 10000:
			warnings.warn(
				"Caution: standardizing the inverse covariance matrix "
				f"of {num_spot} spots requires large dense matrix inversion."
			)
		inv_cov = _standardize_inv_cov(inv_cov.to_dense()).to_sparse()

	if return_sparse:
			return inv_cov
	else:
		return inv_cov.to_dense()


def weights_to_inv_cov(weights, model, rho = 1, standardize = False):
	"""Convert a spatial weight matrix to an inverse covariance matrix.

	Calculate the covariance structure using spatial weights. Different spatial process models
	impose different structures. Check model descriptions for more details.

	Args:
		weights (2D tensor): Spatial weight matrix, num_spot x num_spot.
		model (str): Spatial process model to use, can be one of 'sma','sar', 'isar', 'car', 'icar'.
		rho (float): Spatial autocorrelation parameter.
		standardize (bool): If True, return the standardized inverse covariance matrix (inv_corr).

	Returns:
		inv_cov (2D tensor): An inverse covariance (precision) matrix, num_spot x num_spot.
	"""
	# check spatial priors
	valid_priors = ['sma', 'sar', 'isar', 'car', 'icar']
	if model not in valid_priors:
		raise ValueError(f"Spatial smoothing prior currently must be one of {valid_priors}")

	weights = weights.clone()
	num_spot = weights.shape[0] # number of spatial locations

	# make sure weights to be nonnegative for stability
	weights = weights.clamp(min=0)

	# skip sanity check to save time if the sample is too large
	skip_sanity_check = num_spot >= 5000

	# calculate inverse covariance matrix according to the weights
	if model == 'sma': # spatial moving average model
		cov = torch.eye(num_spot) + rho * (weights + weights.T) + \
			(rho**2) * torch.matmul(weights, weights.T) # covariance matrix
		try:
			inv_cov = torch.linalg.inv(cov) # inverse covariance, or precision matrix
		except RuntimeError:
			warnings.warn(
	   			"The covariance matrix is singular, return pseudo inverse instead. "
		 		"You may want to adjust the smoothing parameter l."
			)
			inv_cov = torch.linalg.pinv(cov)

	elif model == 'sar': # simultaneous auto-regressive model
		inv_cov = torch.matmul(torch.eye(num_spot) - rho * weights.T,
							   torch.eye(num_spot) - rho * weights)

		try: # check if the covariance matrix is positive definite
			if not skip_sanity_check:
				torch.linalg.cholesky(inv_cov)
		except RuntimeError:
			eigvals = np.sort(torch.linalg.eigvals(weights))
			lb, ub = 1.0 / eigvals[0].real, 1.0 / eigvals[-1].real
			warnings.warn(
	   			"The covariance matrix is not positive definite. "
		 		"you may want to adjust the smoothing parameter l, "
				f"e.g. in ({max(lb, 0) : .3f}, {ub : .3f}), "
				"or row-scale the spatial weight matrix (ISAR)."
	 		)

	elif model == 'isar': # intrinsic simultaneous auto-regressive model
		# row-scale the weights matrix
		scaled_weights = weights / weights.sum(1, keepdim=True)

		inv_cov = torch.matmul(torch.eye(num_spot) - rho * scaled_weights.T,
							   torch.eye(num_spot) - rho * scaled_weights)

		# skip checking since the cov matrix is always singular when l=1
		if rho == 1:
			skip_sanity_check = True
			warnings.warn(
				"The covariance matrix is not positive definite. "
				"you may want to adjust the smoothing parameter l, "
				"e.g. in (0, 1)."
			)

		try: # check if the covariance matrix is positive definite
			if not skip_sanity_check:
				torch.linalg.cholesky(inv_cov)
		except RuntimeError:
			warnings.warn(
				"The covariance matrix is not positive definite. "
				"you may want to adjust the smoothing parameter l, "
				"e.g. in (0, 1)."
			)

	elif model == 'car': # conditional auto-regressive model
		inv_cov = torch.eye(num_spot) - rho * weights

		try: # check if the covariance matrix is positive definite
			if not skip_sanity_check:
				torch.linalg.cholesky(inv_cov)
		except RuntimeError:
			if not (weights.T == weights).all():
				warnings.warn(
					"The spatial weight matrix should be symmetric. "
					"You may accidentally row-scale the weight matrix. "
					"Perhaps you want to use the 'sar' model instead."
				)
			else:
				eigvals = np.sort(torch.linalg.eigvals(weights))
				lb, ub = 1.0 / eigvals[0].real, 1.0 / eigvals[-1].real
				warnings.warn(
					"The covariance matrix is not positive definite. "
					"you may want to adjust the smoothing parameter l, "
					f"e.g. in ({max(lb,0) : .3f}, {ub : .3f}), "
					"or try ICAR instead."
				)

	elif model == 'icar': # intrinsic conditional auto-regressive model
		inv_cov = torch.diag(weights.sum(dim=1)) - rho * weights

		# skip checking since the cov matrix is always singular when rho=1
		if rho == 1:
			skip_sanity_check = True
			warnings.warn(
				"The covariance matrix is not positive definite. "
				"you may want to adjust the smoothing parameter l, "
				"e.g. in (0, 1)."
			)

		try: # check if the covariance matrix is positive definite
			if not skip_sanity_check:
				torch.linalg.cholesky(inv_cov)
		except RuntimeError:
			if not (weights.T == weights).all():
				warnings.warn(
					"The spatial weight matrix should be symmetric. "
					"You may accidentally row-scale the weight matrix. "
					"Perhaps you want to use the 'sar' model instead."
				)
			else:
				scaled_weights = torch.diag(weights.sum(dim=1) ** (-0.5)) @ \
					weights @ torch.diag(weights.sum(dim=1) ** (-0.5))
				eigvals = np.sort(torch.linalg.eigvals(scaled_weights))
				lb, ub = 1.0 / eigvals[0].real, 1.0 / eigvals[-1].real
				warnings.warn(
					"The covariance matrix is not positive definite. "
					"you may want to adjust the smoothing parameter l, "
					f"e.g. in ({max(lb,0) : .3f}, {ub : .3f})."
				)

	if standardize:
		return _standardize_inv_cov(inv_cov)

	return inv_cov


def _standardize_inv_cov(inv_cov):
	"""Standardize the inverse covariance matrix.

	When the inverse covariance matrix is singular, the matrix will be normalized instead
	to have diagonal elements of 1 (i.e. return normalized laplacian for 'icar' model).
	"""
	# standardize the inverse covariance matrix
	try:
		cov = torch.linalg.inv(inv_cov)
		sds = torch.diagflat(torch.diagonal(cov) ** 0.5)
		inv_cov_sd = sds @ inv_cov @ sds
	except RuntimeError:
		warnings.warn(
			"The covariance matrix is not positive definite thus "
   			"cannot be standardized. Will return normalized inv_cov matrix instead."
		)
		# adjust for zero diagonal elements
		inv_diag = torch.diagflat((torch.diagonal(inv_cov) ** 0.5 + 1e-8) ** (-1))
		inv_cov_sd = inv_diag @ inv_cov @ inv_diag

	return inv_cov_sd


class SpatialWeightMatrix:
	"""Spatial weight matrix.

	The adjacency matrix that specifies connectivities and interactions between
	each pair of spots.

	Attributes:
		swm (sparse tensor): Unscaled spatial weight matrix.
		swm_scaled (sparse tensor): Spatial weight matrix scaled with external information
  			(e.g., expression, histology).
		inv_covs (dict): Cached inverse covariance matrices under different model settings
  			(for debugging).
		config (dict): Configurations.
	"""
	def __init__(self) -> None:
		# spatial weight matrix
		self.swm = None
		self.swm_scaled = None
		# cached spatial covariance matrix
		self.inv_covs = defaultdict(partial(defaultdict, None))
		# configurations
		self.config = defaultdict(partial(defaultdict, None))

	def _check_swm_stats(self, scaled = False) -> None:
		"""Check spatial weight matrix statistics."""
		if scaled and self.swm_scaled is not None:
			m = self.swm_scaled
		elif self.swm is not None:
			m = self.swm
		else:
			raise ValueError("Spatial weight matrix is not initialized.")

		print(f"Number of spots: {m.shape[0]}. "
			  f"Average number of neighbors per spot: {m._nnz() / m.shape[0] : .2f}.")

	def calc_weights_knn(self, coords, k = 6, symmetric = True, row_scale = False) -> None:
		"""Calculate spatial weight matrix using k-nearest neighbours.

		Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
		interactions among k-nearest neighbours.

		Args:
			coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
			k (int): Number of nearest neighbours to keep.
			symmetric (bool): If True only keep mutual neighbors.
			row_scale (bool): If True scale row sum of the spatial weight matrix to 1.
		"""
		# calculate spatial weight matrix and store as a sparse tensor
		self.swm = coordinate_to_weights_knn_sparse(coords, k = k, symmetric = symmetric, row_scale = row_scale)
		self.swm_scaled = self.swm.clone().float()

		# print out summary statistics of the spatial weight matrix
		self._check_swm_stats(scaled=False)

		# store configs
		self.config['weight'] = {'method':'knn', 'k' : k, 'symmetric' : symmetric,
						   		 'row_scale' : row_scale}

	def calc_weights_dist(self, coords, scale_coords = True, radius_cutoff = 1.0,
						  band_width = 0.1, dist_metric = 'euclidean', row_scale = True) -> None:
		"""Calculate spatial weight matrix using distance band.

		Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
		interactions among neighbours defined by the distane threshold.

		Args:
			coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
			scale_coords (bool): If True, scale coordinates to [0, 1].
			radius_cutoff (float): Distance threshold (in the same unit as the coords input).
			band_width (float): Specify the width of the Gaussian kernel, which is proportional
				to the inverse rate of weight distance decay.
			dist_metric (str): Distance metric.
			row_scale (bool): If True scale row sum of the spatial weight matrix to 1.
		"""
		# calculate spatial weight matrix and store as a sparse tensor
		self.swm = coordinate_to_weights_dist_sparse(
			coords, scale_coords, radius_cutoff, band_width, dist_metric, row_scale)
		self.swm_scaled = self.swm.clone().float()

		# print out summary statistics of the spatial weight matrix
		self._check_swm_stats(scaled=False)

		# store configs
		self.config['weight'] = {'method':'dist', 'scale_coords':scale_coords, 'radius_cutoff':radius_cutoff,
								 'band_width':band_width, 'dist_metric':dist_metric, 'row_scale':row_scale}

	def scale_by_similarity(self, pairwise_sim : torch.Tensor, row_scale = False, return_swm = False):
		"""Scale spatial weight matrix by external pairwise similarity.

		Args:
			pairwise_sim (2D tensor): External pairwise similarity, num_spot x num_spot.
			row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.
			return_swm (bool): Whether to output the scaled spatial weight matrix.
		"""
		# if pairwise_sim is sparse
		if isinstance(pairwise_sim, torch.sparse.FloatTensor):
			self.swm_scaled = torch.mul(self.swm, pairwise_sim).float()
		else:
			# keep only edges in the spatial weight matrix
			pairwise_sim_sparse = torch.sparse_coo_tensor(
				self.swm.indices(),
				pairwise_sim[self.swm.indices()[0], self.swm.indices()[1]],
				dtype=torch.float32
			).coalesce()
			self.swm_scaled = torch.mul(self.swm, pairwise_sim_sparse).float()

		# set diagonal to 1 for spots with no neighbors
		id_no_neighbors = torch.where(torch.sparse.sum(self.swm_scaled, 1).to_dense() == 0)[0]
		self.swm_scaled = self.swm_scaled + torch.sparse_coo_tensor(
			id_no_neighbors.repeat(2, 1),
			torch.ones(len(id_no_neighbors)),
			self.swm_scaled.shape, dtype=torch.float32
		).coalesce()

		if row_scale: # scale row to sum to 1
			row_sum = torch.sparse.sum(self.swm_scaled, 1).to_dense()
			self.swm_scaled.values()[:] = self.swm_scaled.values() / row_sum[self.swm_scaled.indices()[0]]

		# print out summary statistics of the spatial weight matrix
		self._check_swm_stats(scaled=True)

		if return_swm:
			return self.swm_scaled

	def scale_by_identity(self, spot_ids, boundary_connectivity = 0,
						  row_scale = False, return_swm = False):
		"""Scale spatial weight matrix by spot identity.

		Args:
			spot_ids (1D array): Spot identity of length num_spot.
			boundary_connectivity (float): Connectivity of spots with different identities.
				If 0 (default), no interaction across identities.
		"""
		if isinstance(spot_ids, pd.Series):
			spot_ids = spot_ids.to_numpy()

		neighbors = self.swm.indices()
		is_in_same_group = torch.tensor(spot_ids[neighbors[0]] == spot_ids[neighbors[1]])

		# remove edges between spots with different identities
		self.swm_scaled = self.swm.clone().float()
		self.swm_scaled.values()[~is_in_same_group] = boundary_connectivity

		# set diagonal to 1 for spots with no neighbors
		id_no_neighbors = torch.where(torch.sparse.sum(self.swm_scaled, 1).to_dense() == 0)[0]
		self.swm_scaled = self.swm_scaled + torch.sparse_coo_tensor(
			id_no_neighbors.repeat(2, 1),
			torch.ones(len(id_no_neighbors)),
			self.swm_scaled.shape, dtype=torch.float32
		).coalesce()

		if row_scale: # scale row to sum to 1
			row_sum = torch.sparse.sum(self.swm_scaled, 1).to_dense()
			self.swm_scaled.values()[:] = self.swm_scaled.values() / row_sum[self.swm_scaled.indices()[0]]

		# print out summary statistics of the spatial weight matrix
		self._check_swm_stats(scaled=True)

		if return_swm:
			return self.swm_scaled

	def scale_by_expr(self, expr, dist_metric = 'cosine',
					  reduce = 'pca', dim = 10, row_scale = False) -> None:
		"""Scale weight matrix using transcriptional similarity.

		Args:
			expr (2D array): Spatial gene expression count matrix, num_genes x num_spot.
			dist_metric (str): Distance metric.
			reduce (str): If `PCA`, calculate distance on the reduced PCA space.
			dim (int): Number of dimension of the reduced space.
			row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.
		"""
		pairwise_sim = calc_feature_similarity_sparse(
			expr, self.swm.indices(), dist_metric = dist_metric,
			reduce = reduce, dim = dim, nonneg=True, return_type = 'sparse'
		)
		self.scale_by_similarity(pairwise_sim, row_scale=row_scale)
		# store configs
		self.config['similarity'] = {'source' : 'expression', 'dist_metric' : dist_metric,
							   		 'reduce' : reduce, 'dim' : dim}

	def scale_by_histology(self, coords, image, scale_factors : dict,
						   dist_metric = 'euclidean', reduce = 'pca', dim = 10, row_scale = False):
		"""Calculate pairwise histology similarity between spots.

		Args:
			coords (2D array): Spatial coordinate matrix (in fullres pixel), num_spot x 2.
			image (3D array): Histology image, num_pixel x num_pixel x num_channel.
			scale_fators (dict): The JSON dictionary from 10x Visium's `scalefactors_json.json`
				'spot_diameter_fullres' (float): Spot size (fullres)
				'tissue_hires_scalef' (float): Scale factor that transforms fullres
					image to the given image.
			reduce (str): If `PCA`, calculate distance on the reduced PCA space.
			dist_metric (str): Distance metric used to calculate similarity.
			dim (int): Number of dimension of the reduced space.
			row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.
		"""
		pairwise_sim = calc_histology_similarity_sparse(
			coords, self.swm.indices(), image, scale_factors, \
			dist_metric, reduce, dim
		)
		self.scale_by_similarity(pairwise_sim, row_scale=row_scale)
		# store configs
		self.config['similarity'] = {'source' : 'histology',
									 'scale_factors' : scale_factors,
									 'dist_metric' : dist_metric,
									 'reduce' : reduce, 'dim' : dim}

	def get_inv_cov(self, model, rho = 1, cached = True, standardize = False, return_sparse = True):
		"""Calculate or extract cached inverse covariance matrix.

		Args:
			model (str): The spatial process model, can be one of 'sma','sar', 'car', 'icar'.
			rho (float): The spatial autocorrelation parameter.
			return_sparse (bool): If True, return sparse matrix.
		"""
		if not cached:
			return sparse_weights_to_inv_cov(
				self.swm_scaled, model=model, rho=rho, standardize=standardize,
				return_sparse=return_sparse
			)

		key = f"{model}_{rho}_{'sd' if standardize else 'nsd'}_{'sp' if return_sparse else 'ds'}"
		if self.inv_covs[key] is None:
			self.inv_covs[key] = sparse_weights_to_inv_cov(
				self.swm_scaled, model=model, rho=rho, standardize=standardize, return_sparse=return_sparse)

		return self.inv_covs[key]
