"""
Calculate spatial weight matrix from coordinates, histology images, and transcriptomic data
"""

from collections import defaultdict
import warnings
import numpy as np
import torch
from torch.nn.functional import cosine_similarity, pairwise_distance
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

def _normalize_coords(coords, min_zero = True, return_scale = False):
	"""Re-scale spatial coordinates.

	Args:
		coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
		min_zero (bool): If True, set minimum coordinate to 0.
		return_scale (bool): If True, return the scaling factor.
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
	scale = coords_norm.abs().max()
	coords_norm = coords_norm / scale

	# return scaling factor
	if return_scale:
		return coords_norm, scale

	return coords_norm

def coordinate_to_weights_knn_fast(coords, k = 6, symmetric = True, row_scale = True):
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

def coordinate_to_weights_dist_fast(coords, scale_coords = True, radius_cutoff = 1.0,
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
		coords_norm, coords_scale = _normalize_coords(coords, min_zero=True, return_scale=True)
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

def calc_feature_similarity_sparse(
    features : torch.Tensor, indices : torch.Tensor,
    dist_metric = 'cosine', reduce = 'none', dim = 10, return_sparse = False):
	"""Calculate pairwise feature similarity between spots for a given set of spot pairs.

	Similarity `s` is transformed from distance `d` by
	1) Cosine similarity: s = (1 - d).clamp(min = 0)
	2) Others: s = exp(- d^2/(2 * band_width^2)))

	Args:
		features (2D tensor): Feature matrix, num_feature x num_spot.
		indices (2D tensor): Pairs of spot indices of which to calculate the similarity, 2 x num_pairs.
		dist_metric (str): Distance metric.
		reduce (str): If `PCA`, calculate distance on the reduced PCA space.
		dim (int): Number of dimension of the reduced space.
		return_sparse (bool): If True, return the similarity matrix as a sparse tensor.

	Returns:
		feature_sim: A 2D tensor containing pairwise similarities, num_spot x num_spot.
	"""

	assert dist_metric in ['cosine', 'cos', 'euclidean', 'eu']
	assert reduce in ['pca', 'mean', 'none']

	# Dimension reduction
	if reduce == 'pca':
		# standardize and remove constant features
		features_var = features.std(1) # feature level variance
		features_scaled = (features - features.mean(1, keepdim=True)) / features_var[:, None]
		features_scaled = features_scaled[features_var > 0,:]
		# run pca
		torch.manual_seed(0) # for repeatability, fix the random seed for pca_lowrank
		_, _, v = torch.pca_lowrank(
			features_scaled.T, center=False,
			q = min(dim, features_scaled.shape[0])
		)
		features_reduced_T = torch.matmul(features_scaled.T, v)
	else:
		features_reduced_T = features.T

	if dist_metric in ['euclidean', 'eu']: # euclidean distance
    	# scale the maximum value to 1
		features_reduced_T = _normalize_coords(features_reduced_T, min_zero=False)
		# calculate pairwise distance of selected pairs
		features_melt = features_reduced_T[indices] # 2 x num_pairs x dim
		features_dist = pairwise_distance(features_melt[0], features_melt[1])
		# convert distance to similarity
		band_width = 0.1 # fixed band width in the gaussian kernel
		features_sim_flat = torch.exp(- features_dist.pow(2) / (2 * band_width ** 2))
	else: # cosine similarity
		# calculate pairwise distance of selected pairs
		features_melt = features_reduced_T[indices] # 2 x num_pairs x dim
		features_sim_flat = cosine_similarity(features_melt[0], features_melt[1]).clamp(min = 0)

	# convert to sparse tensor
	features_sim = 	torch.sparse_coo_tensor(
		indices, features_sim_flat,
		torch.Size([features.shape[1], features.shape[1]]),
		dtype=torch.float32
	).coalesce()

	if return_sparse:
		return features_sim
	else:
		return features_sim.to_dense()


def calc_feature_similarity(features : torch.Tensor, dist_metric = 'cosine',
							reduce = 'pca', dim = 10):
	"""Calculate pairwise feature similarity between spots.

	Similarity `s` is transformed from distance `d` by
	1) Cosine similarity: s = (1 - d).clamp(min = 0)
	2) Others: s = exp(- d^2/(2 * band_width^2)))

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
		# run pca
		torch.manual_seed(0) # for repeatability, fix the random seed for pca_lowrank
		_, _, v = torch.pca_lowrank(
			features_scaled.T, center=False,
			q = min(dim, features_scaled.shape[0])
		)
		features_reduced_T = torch.matmul(features_scaled.T, v)
	else:
		features_reduced_T = features.T

	if dist_metric in ['euclidean', 'eu']: # scale the maximum to 1
		features_reduced_T = _normalize_coords(features_reduced_T, min_zero=False)

	# calculate pairwise distances and similarities
	features_dist = torch.tensor(distance.squareform(
		distance.pdist(features_reduced_T, metric=dist_metric)))

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

def calc_histology_similarity_sparse(
    	coords, indices, image, scale_factors,
		dist_metric = 'euclidean', reduce = 'none', dim = 3):
	"""Calculate pairwise histology similarity between spots for a given set of points.

	Args:
		coords (2D array): Spatial coordinate matrix (in fullres pixel), num_spot x 2.
		indices (2D tensor): Pairs of spot indices of which to calculate the similarity, 2 x num_pairs.
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
	hist_sim = calc_feature_similarity_sparse(
    	hist_vec, indices, dist_metric = dist_metric,
		reduce = reduce, dim = dim, return_sparse=True
	)

	return hist_sim

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


def sparse_weight_to_inv_cov(weights, model, l = 1, standardize = False, return_sparse = False):
	"""Convert a spatial weight matrix to an inverse covariance matrix. Sparse version.

	Calculate the covariance structure using spatial weights. Different spatial process models
	impose different structures. Check model descriptions for more details.

	Args:
		weights (Sparse tensor): Spatial weight matrix, num_spot x num_spot.
		model (str): Spatial process model to use, can be one of 'sma','sar', 'isar', 'car', 'icar'.
		l (float): Smoothing effect size.
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
			raise NotImplementedError("Sparse inverse covariance matrix is not available for SMA model.")

		# not recommended for large number of spots
		if num_spot > 10000:
			warnings.warn(
				f"Caution: SMA model of {num_spot} spots requires large dense matrix inversion."
			)

		cov = sparse_i + l * (weights + weights.transpose(0,1)) + \
			(l**2) * torch.matmul(weights, weights.transpose(0,1)) # covariance matrix
		try:
			inv_cov = torch.linalg.inv(cov.to_dense()) # inverse covariance, or precision matrix
		except RuntimeError:
			warnings.warn(
	   			"The covariance matrix is singular, return pseudo inverse instead. "
		 		"You may want to adjust the smoothing parameter l."
			)
			inv_cov = torch.linalg.pinv(cov.to_dense())

	elif model == 'sar': # simultaneous auto-regressive model
		inv_cov = torch.matmul(sparse_i - l * weights.transpose(0,1),
							   sparse_i - l * weights)

	elif model == 'isar': # intrinsic simultaneous auto-regressive model
		# row-scale the weights matrix
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		weights.values()[:] = weights.values() / row_sum[weights.indices()[0]]

		inv_cov = torch.matmul(sparse_i - l * weights.transpose(0,1),
							   sparse_i - l * weights)

	elif model == 'car': # conditional auto-regressive model
		inv_cov = sparse_i - l * weights

	elif model == 'icar': # intrinsic conditional auto-regressive model
		row_sum = torch.sparse.sum(weights, 1).to_dense()
		inv_cov = torch.sparse_coo_tensor(
			torch.arange(num_spot).repeat(2,1),
			row_sum,
			weights.shape, dtype=torch.float32
		) - l * weights

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
		self.inv_covs = defaultdict(lambda: None)
		# configurations
		self.config = defaultdict(lambda: None)

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
		self.swm = coordinate_to_weights_knn_fast(coords, k = k, symmetric = symmetric, row_scale = row_scale)
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
		self.swm = coordinate_to_weights_dist_fast(
    		coords, scale_coords, radius_cutoff, band_width, dist_metric, row_scale)
		self.swm_scaled = self.swm.clone().float()

		# print out summary statistics of the spatial weight matrix
		self._check_swm_stats(scaled=False)

		# store configs
		self.config['weight'] = {'method':'dist', 'scale_coords':scale_coords, 'radius_cutoff':radius_cutoff,
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
						 			   histology_axis_scale, band_width).to_sparse()
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
			reduce = reduce, dim = dim, return_sparse = True
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

	def get_inv_cov(self, model, l = 1, cached=True, standardize=False, return_sparse=True):
		"""Calculate or extract cached inverse covariance matrix.

		Args:
			model (str): The spatial process model, can be one of 'sma','sar', 'car', 'icar'.
			l (float): Smoothing effect size.
			return_sparse (bool): If True, return sparse matrix.
		"""
		if not cached:
			return sparse_weight_to_inv_cov(
				self.swm_scaled, model=model, l=l, standardize=standardize,
				return_sparse=return_sparse
			)

		key = f"{model}_{l}_{'sd' if standardize else 'nsd'}_{'sp' if return_sparse else 'ds'}"
		if self.inv_covs[key] is None:
			self.inv_covs[key] = sparse_weight_to_inv_cov(
				self.swm_scaled, model=model, l=l, standardize=standardize, return_sparse=return_sparse)

		return self.inv_covs[key]
