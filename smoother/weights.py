"""
Calculate spatial weight matrix from coordinates, histology images, and transcriptomic data
"""

from collections import defaultdict
import warnings
import numpy as np
import torch
from scipy.spatial import distance

def _normalize_coords(coords, min_zero = True):
	"""Re-scale spatial coordinates.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		min_zero (bool): If True, set minimum coordinate to 0.
	"""
	if not torch.is_tensor(coords):
		if not isinstance(coords, np.ndarray):
			coords = np.array(coords)
		coords = torch.tensor(coords)

	if min_zero: # set minimum coordinate to 0
		coords_min, _ = coords.min(dim = 0, keepdim = True)
		coords_norm = coords - coords_min
	else:
		coords_norm = coords.clone()

	# set maximum coordinate to 1 (or -1)
	coords_norm = coords_norm / coords_norm.abs().max()
	return coords_norm

def coordinate_to_weights_knn(coords, k = 6, symmetric = True, row_scale = True):
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
	coords_norm = _normalize_coords(coords, min_zero=True)
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


def coordinate_to_weights_dist(coords, scale_coords = True, q_threshold = 0.001,
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
		coords_norm = _normalize_coords(coords, min_zero=True)
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


def calc_feature_similarity(features : torch.Tensor, dist_metric = 'cosine',
							reduce = 'pca', dim = 10):
	"""Calculate pairwise feature similarity between spots.

	Similarity `s` is transformed from distance `d` by
	1) Cosine similarity: s = (1 - d).clamp(min = 0)
	2) Others: s = f_scale(exp(- 1/d.std()))

	Args:
		features (2D tensor): Feature matrix, num_feature x num_spot.
		dist_metric (str): Distance metric.
		reduce (str): If `PCA`, calculate distance on the reduced PCA space.
		dim (int): Number of dimension of the reduced space.

	Returns:
		feature_sim: A 2D tensor containing pairwise similarities, num_spot x num_spot.
	"""
	assert reduce in ['pca', 'mean', 'none']

	# Dimension reduction
	if reduce == 'pca':
		# standardize and remove constant features
		features_var = features.std(1) # feature level variance
		features_scaled = (features - features.mean(1, keepdim=True)) / features_var[:, None]
		features_scaled = features_scaled[features_var > 0,:]
		# pca
		_, _, v = torch.pca_lowrank(features_scaled.T, q = dim)
		features_reduced = torch.matmul(features_scaled.T, v)
	else:
		features_reduced = features.T

	if dist_metric in ['euclidean', 'eu']: # scale the maximum to 1
		features_reduced = _normalize_coords(features_reduced.T, min_zero=False).T

	# calculate pairwise distances and similarities
	features_dist = torch.tensor(distance.squareform(
		distance.pdist(features_reduced, metric=dist_metric)))

	if dist_metric in ['cosine', 'cos']:
		features_sim = (1 - features_dist).clamp(min = 0)
	else:
		band_width = 0.1 # fixed band width in the gaussian kernel
		features_sim = torch.exp(- features_dist.pow(2) / (2 * band_width ** 2))
		# features_sim = features_sim.fill_diagonal_(0)
		# # scale the largest similarity per row to 1
		# row_sf, _ = features_sim.max(1, keepdim=True)
		# features_sim = features_sim / row_sf

	return features_sim


def get_histology_vector(image, x_pixel, y_pixel, spot_radius, scale_factor, padding = True):
	"""Get the histology image vector of one spot.

	Args:
		image (3D array): Histology image, num_pixel x num_pixel x num_channel.
		x_pixel (float): Spot centric position (in fullres).
		y_pixel (float): Spot centric position (in fullres).
		spot_radius (float): Spot size (in fullres).
		scale_factor (float): Scale factor that transforms fullres image to the given image.
		paddings (bool): Whether to pad for boundary spots.
			If False, will return the averaged color vector.

	Returns:
		spot_vec (1D array): A vector containing histology information around the spot.
	"""
	# scale pixels and radius
	x_pixel, y_pixel = x_pixel * scale_factor, y_pixel * scale_factor
	spot_radius = int(spot_radius * scale_factor)

	# calculate spot region
	x_min = max(0, int(x_pixel - spot_radius))
	y_min = max(0, int(y_pixel - spot_radius))
	x_max = min(image.shape[0], int(x_pixel + spot_radius))
	y_max = min(image.shape[1], int(y_pixel + spot_radius))

	# extract image for the region
	spot_vec = image[x_min:x_max, y_min:y_max, :]

	if padding: # calculate padding
		x_before = spot_radius - int(x_pixel) + x_min
		y_before = spot_radius - int(y_pixel) + y_min
		x_after = spot_radius - x_max + int(x_pixel)
		y_after = spot_radius - y_max + int(y_pixel)

		# apply padding and concatenate all surrounding locations
		spot_vec = np.pad(spot_vec, pad_width=((x_before, x_after), (y_before, y_after), (0, 0)),
						  mode='mean').reshape(-1)
	else:
		spot_vec = spot_vec.mean(axis = (0,1))

	return spot_vec


def calc_histology_similarity(coords, image, scale_factors,
							  dist_metric = 'euclidean', reduce = 'pca', dim = 3):
	"""Calculate pairwise histology similarity between spots.

	Args:
		coords (2D array): Spatial coordinate matrix (in fullres pixel), num_spot x 2.
		image (3D array): Histology image, num_pixel x num_pixel x num_channel.
		scale_fators (dict): The JSON dictionary from 10x Visium's `scalefactors_json.json`
			'spot_diameter_fullres' (float): Spot size (fullres)
			'tissue_hires_scalef' (float): Scale factor that transforms fullres image to the given image.
		reduce (str): How to compute histological similarity. Can be one of 'pca', 'mean', and 'none'.
			If 'none', will concatenate pixel-level histology vectors of each spot and calculate distance.
			If 'pca', will concatenate pixel-level histology vectors of each spot, apply PCA to reduce
				the dimension of the histology space, then calculate the distance.
			If 'mean', will average the histology vector of each spot over its covering area.
		dim (int): Number of dimension of the reduced space.

	Returns:
		hist_sim: A 2D tensor containing pairwise histology similarities, num_spot x num_spot.
	"""
	spot_radius = scale_factors['spot_diameter_fullres']/2
	scale_factor = scale_factors['tissue_hires_scalef']

	assert reduce in ['pca', 'mean', 'none']

	# extract histology vectors per location
	padding = (reduce != 'mean')
	hist_vec = np.apply_along_axis(lambda arr: \
		get_histology_vector(image, arr[0], arr[1], spot_radius, scale_factor, padding), 1, coords)

	# num_hist_feature x num_spot
	hist_vec = torch.tensor(hist_vec).T

	# calculate similarity
	hist_sim = calc_feature_similarity(hist_vec, dist_metric = dist_metric,
									   reduce = reduce, dim = dim)

	return hist_sim


def calc_weights_spagcn(coords, image, scale_factors,
						histology_axis_scale = 1.0, band_width = 1.0):
	"""Calculate spatial weight matrix similar to the SpaGCN edge weight.

	https://www.nature.com/articles/s41592-021-01255-8#Sec9

	Args:
		coords (2D array): Spatial coordinate matrix (in fullres pixel), num_spot x 2.
		image (3D array): Histology image, num_pixel x num_pixel x num_channel.
		scale_fators (dict): The JSON dictionary from 10x Visium's `scalefactors_json.json`
			'spot_diameter_fullres' (float): Spot size (fullres)
			'tissue_hires_scalef' (float): Scale factor that transforms fullres image to the given image.
		histology_axis_scale (float): The relative strength of the histology axis (integrated color)
		band_width (float): Specify the width of the Gaussian kernel, which is proportional
			to the inverse rate of weight distance decay.

	Returns:
		swm: A 2D tensor containing spatial weights, num_spot x num_spot.
	"""
	spot_radius = scale_factors['spot_diameter_fullres']/2
	scale_factor = scale_factors['tissue_hires_scalef']

	# extract histology vectors (averaged)
	hist_vec = np.apply_along_axis(lambda arr: \
		get_histology_vector(image, arr[0], arr[1], spot_radius, scale_factor, padding=False), 1, coords)

	hist_vec = torch.tensor(hist_vec, dtype=torch.float32)

	# combine different color channels
	z_vec = (hist_vec * torch.var(hist_vec, dim=0)).sum(1) / torch.var(hist_vec, dim=0).sum()

	# rescale z to match x and y scales
	coords_xy = torch.tensor(coords, dtype=torch.float32)
	z_scaled = (z_vec - z_vec.mean()) / z_vec.std() * torch.max(torch.std(coords_xy, 0)) * \
				histology_axis_scale

	# calculate weight using euclidean distance in 3D
	coords_xyz = torch.cat([coords_xy, z_scaled[:, None]], dim=1)
	swm = coordinate_to_weights_dist(coords_xyz, scale_coords=False, band_width=band_width)

	return swm


def weight_to_inv_cov(weights, model, l = 1, standardize = False):
	"""Convert a spatial weight matrix to an inverse covariance matrix.

	Calculate the covariance structure using spatial weights. Different spatial process models
	impose different structures. Check model descriptions for more details.

	Args:
		weights (2D tensor): Spatial weight matrix, num_spot x num_spot.
		model (str): Spatial process model to use, can be one of 'sma','sar', 'isar', 'car', 'icar'.
		l (float): Smoothing effect size.
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
		cov = torch.eye(num_spot) + l * (weights + weights.T) + \
			(l**2) * torch.matmul(weights, weights.T) # covariance matrix
		try:
			inv_cov = torch.linalg.inv(cov) # inverse covariance, or precision matrix
		except RuntimeError:
			warnings.warn(
	   			"The covariance matrix is singular, return pseudo inverse instead. "
		 		"You may want to adjust the smoothing parameter l."
			)
			inv_cov = torch.linalg.pinv(cov)

	elif model == 'sar': # simultaneous auto-regressive model
		inv_cov = torch.matmul(torch.eye(num_spot) - l * weights.T,
							   torch.eye(num_spot) - l * weights)

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

		inv_cov = torch.matmul(torch.eye(num_spot) - l * scaled_weights.T,
							   torch.eye(num_spot) - l * scaled_weights)

		# skip checking since the cov matrix is always singular when l=1
		if l == 1:
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
		inv_cov = torch.eye(num_spot) - l * weights

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
		inv_cov = torch.diag(weights.sum(dim=1)) - l * weights

		# skip checking since the cov matrix is always singular when l=1
		if l == 1:
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
		swm (2D tensor): Unscaled spatial weight matrix.
		swm_scaled (2D tensor): Spatial weight matrix scaled with external information
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
		self.inv_covs = defaultdict(lambda: None)
		# configurations
		self.config = defaultdict(lambda: None)

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
		self.swm = coordinate_to_weights_knn(coords, k = k, symmetric = symmetric, row_scale = row_scale)
		self.swm_scaled = self.swm.clone().float()
		# store configs
		self.config['weight'] = {'method':'knn', 'k' : k, 'symmetric' : symmetric,
						   		 'row_scale' : row_scale}

	def calc_weights_dist(self, coords, scale_coords = True, q_threshold = 0.001, band_width = 0.1,
						  dist_metric = 'euclidean', row_scale = False) -> None:
		"""Calculate spatial weight matrix using distance band.

		Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
		interactions among neighbours defined by the distane threshold.

		Args:
			coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
			q_threshold (float): Distance quantile threshold. Number of nonzero entries in the
				weight matrix (edges) = num_spot^2 * q_threshold.
			band_width (float): Specify the width of the Gaussian kernel, which is proportional
				to the inverse rate of weight distance decay.
			dist_metric (str): Distance metric.
			row_scale (bool): If True scale row sum of the spatial weight matrix to 1.
		"""
		self.swm = coordinate_to_weights_dist(coords, scale_coords, q_threshold, band_width,
											  dist_metric, row_scale)
		self.swm_scaled = self.swm.clone().float()
		# store configs
		self.config['weight'] = {'method':'dist', 'scale_coords':scale_coords, 'q_threshold':q_threshold,
								 'band_width':band_width, 'dist_metric':dist_metric, 'row_scale':row_scale}

	def calc_weights_spagcn(self, coords, image, scale_factors : dict,
						 	histology_axis_scale = 1.0, band_width = 1.0):
		"""Calculate pairwise histology similarity between spots.

		Args:
			coords (2D array): Spatial coordinate matrix (in fullres pixel), num_spot x 2.
			image (3D array): Histology image, num_pixel x num_pixel x num_channel.
			scale_fators (dict): The JSON dictionary from 10x Visium's `scalefactors_json.json`
				'spot_diameter_fullres' (float): Spot size (fullres)
				'tissue_hires_scalef' (float): Scale factor that transforms fullres image to the given image.
			reduce (str): If `PCA`, calculate distance on the reduced PCA space.
			dim (int): Number of dimension of the reduced space.

		Returns:
			hist_sim: A 2D tensor containing pairwise histology similarities, num_spot x num_spot.
		"""
		self.swm = calc_weights_spagcn(coords, image, scale_factors,
						 			   histology_axis_scale, band_width)
		self.swm_scaled = self.swm.clone().float()
		# store configs
		self.config['weight'] = {'method':'SpaGCN',
						   		 'scale_factors' : scale_factors,
							  	 'histology_axis_scale' : histology_axis_scale,
								 'band_width' : band_width}

	def scale_by_similarity(self, pairwise_sim : torch.Tensor, row_scale = False, return_swm = False):
		"""Scale spatial weight matrix by external pairwise similarity.

		Args:
			pairwise_sim (2D tensor): External pairwise similarity, num_spot x num_spot.
			row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.
			return_swm (bool): Whether to output the scaled spatial weight matrix.
		"""
		self.swm_scaled = torch.mul(self.swm, pairwise_sim).float()

		# set diagonal to 1 for spots with no neighbours
		id_no_neighbors = torch.sum(self.swm_scaled, 1) == 0
		self.swm_scaled[id_no_neighbors, id_no_neighbors] = 1

		if row_scale: # scale row to sum to 1
			self.swm_scaled = self.swm_scaled / self.swm_scaled.sum(1, keepdim=True)

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
		is_in_same_group = torch.tensor(spot_ids[:,None] == spot_ids)
		is_neighbors = self.swm > 0

		self.swm_scaled = self.swm.masked_fill_((~is_in_same_group) & (is_neighbors),
										  		boundary_connectivity)

		# set diagonal to 1 for spots with no neighbours
		id_no_neighbors = torch.sum(self.swm_scaled, 1) == 0
		self.swm_scaled[id_no_neighbors, id_no_neighbors] = 1

		if row_scale: # scale row to sum to 1
			self.swm_scaled = self.swm_scaled / self.swm_scaled.sum(1, keepdim=True)

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
		pairwise_sim = calc_feature_similarity(expr, dist_metric = dist_metric,
											   reduce = reduce, dim = dim)
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
		pairwise_sim = calc_histology_similarity(coords, image, scale_factors, \
												 dist_metric, reduce, dim)
		self.scale_by_similarity(pairwise_sim, row_scale=row_scale)
		# store configs
		self.config['similarity'] = {'source' : 'histology',
									 'scale_factors' : scale_factors,
									 'dist_metric' : dist_metric,
									 'reduce' : reduce, 'dim' : dim}

	def get_inv_cov(self, model, l = 1, cached=True, standardize=False):
		"""Calculate or extract cached inverse covariance matrix.

		Args:
			model (str): The spatial process model, can be one of 'sma','sar', 'car', 'icar'.
			l (float): Smoothing effect size.
		"""
		if not cached:
			return weight_to_inv_cov(self.swm_scaled, model=model, l=l, standardize=standardize)

		key = f"{model}_{l}_{'sd' if standardize else 'nsd'}"
		if self.inv_covs[key] is None:
			self.inv_covs[key] = weight_to_inv_cov(
				self.swm_scaled, model=model, l=l, standardize=standardize)

		return self.inv_covs[key]
