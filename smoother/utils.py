"""
Utility functions
"""
import warnings
import numpy as np
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity, pairwise_distance
from scipy.spatial import distance
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
try:
	from plotnine import *
except:
	warnings.warn(
    	"Package 'plotnine' not installed. Some visualization functions will not work.",
		category=ImportWarning
	)


def normalize_minmax(x, min_zero = True, return_scale = False):
	"""Normalize data to [0, 1] or [-1, 1].

	Args:
		x (2D array): data to be normalized.
		min_zero (bool): If True, set column minimum to 0.
		return_scale (bool): If True, return the scaling factor.
	"""
	if not torch.is_tensor(x):
		if not isinstance(x, np.ndarray):
			x = np.array(x)
		x = torch.tensor(x)

	if min_zero: # set column minimum to 0
		x_min, _ = x.min(dim = 0, keepdim = True)
		x_norm = x - x_min
	else:
		x_norm = x.clone()

	# set global maximum to 1 (or -1)
	scale = x_norm.abs().max()
	x_norm = x_norm / scale

	# return scaling factor
	if return_scale:
		return x_norm, scale

	return x_norm

def _z_score(features):
	"""Z-score standardization."""
	# standardize and remove constant features
	# features.shape: num_feature x num_spot
	features_var = features.std(1) # feature level variance
	features_scaled = (features - features.mean(1, keepdim=True)) / features_var[:, None]
	features_scaled = features_scaled[features_var > 0,:]
	return features_scaled

def _pca(features, dim):
	"""Dimension reduction by PCA."""
	# standardize and remove constant features
	features_scaled = _z_score(features)

	# run pca
	torch.manual_seed(0) # for repeatability, fix the random seed for pca_lowrank
	_, _, v = torch.pca_lowrank(
		features_scaled.T, center=False,
		q = min(dim, features_scaled.shape[0])
	)
	feature_reduced = torch.matmul(features_scaled.T, v).T

	return feature_reduced

def calc_feature_similarity_sparse(
	features : torch.Tensor, indices : torch.Tensor,
	dist_metric = 'cosine', reduce = 'none', dim = 10,
	nonneg = False, return_type = 'flat'):
	"""Calculate pairwise feature similarity between spots for a given set of spot pairs.

	Similarity `s` is transformed from distance `d` by
	1) Cosine similarity: s = (1 - d) (if nonneg, (1 - d).clamp(min = 0))
	2) Others: s = exp(- d^2/(2 * band_width^2)))

	Args:
		features (2D tensor): Feature matrix, num_feature x num_spot.
		indices (2D tensor): Pairs of spot indices of which to calculate the similarity, 2 x num_pairs.
		dist_metric (str): Distance metric.
		reduce (str): If `PCA`, calculate distance on the reduced PCA space.
		dim (int): Number of dimension of the reduced space.
		nonneg (bool): If True, set negative similarity to 0.
		return_type (str): If `sparse`, return a sparse tensor. If `dense`, return a dense tensor.
  			If `flat`, return a flat tensor of length num_pairs.

	Returns:
		feature_sim: A 2D tensor containing pairwise similarities, num_spot x num_spot.
	"""

	assert dist_metric in ['cosine', 'cos', 'euclidean', 'eu']
	assert reduce in ['pca', 'mean', 'none']
	assert return_type in ['sparse', 'dense', 'flat']

	# Dimension reduction
	if reduce == 'pca':
		features_reduced_T = _pca(features, dim).T
	else:
		features_reduced_T = features.T

	if dist_metric in ['euclidean', 'eu']: # euclidean distance
		# scale the maximum value to 1
		features_reduced_T = normalize_minmax(features_reduced_T, min_zero=False)
		# calculate pairwise distance of selected pairs
		features_melt = features_reduced_T[indices] # 2 x num_pairs x dim
		features_dist = pairwise_distance(features_melt[0], features_melt[1])
		# convert distance to similarity
		band_width = 0.1 # fixed band width in the gaussian kernel
		features_sim_flat = torch.exp(- features_dist.pow(2) / (2 * band_width ** 2))
	else: # cosine similarity
		# calculate pairwise distance of selected pairs
		features_melt = features_reduced_T[indices] # 2 x num_pairs x dim
		features_sim_flat = cosine_similarity(features_melt[0], features_melt[1])
		if nonneg: # remove negative similarity
			features_sim_flat = features_sim_flat.clamp(min = 0)

	if return_type == 'flat':
		return features_sim_flat

	# convert to sparse tensor
	features_sim = 	torch.sparse_coo_tensor(
		indices, features_sim_flat,
		torch.Size([features.shape[1], features.shape[1]]),
		dtype=torch.float32
	).coalesce()

	if return_type == 'sparse':
		return features_sim
	else:
		return features_sim.to_dense()


def calc_feature_similarity(
	features : torch.Tensor, dist_metric = 'cosine',
	reduce = 'pca', dim = 10, nonneg = False):
	"""Calculate pairwise feature similarity between spots.

	Similarity `s` is transformed from distance `d` by
	1) Cosine similarity: s = (1 - d) (if nonneg, (1 - d).clamp(min = 0))
	2) Others: s = exp(- d^2/(2 * band_width^2)))

	Args:
		features (2D tensor): Feature matrix, num_feature x num_spot.
		dist_metric (str): Distance metric.
		reduce (str): If `PCA`, calculate distance on the reduced PCA space.
		dim (int): Number of dimension of the reduced space.
		nonneg (bool): If True, set negative similarity to 0.

	Returns:
		feature_sim: A 2D tensor containing pairwise similarities, num_spot x num_spot.
	"""
	assert reduce in ['pca', 'mean', 'none']

	# Dimension reduction
	if reduce == 'pca':
		features_reduced_T = _pca(features, dim).T
	else:
		features_reduced_T = features.T

	if dist_metric in ['euclidean', 'eu']: # scale the maximum to 1
		features_reduced_T = normalize_minmax(features_reduced_T, min_zero=False)

	# calculate pairwise distances and similarities
	features_dist = torch.tensor(distance.squareform(
		distance.pdist(features_reduced_T, metric=dist_metric)))

	if dist_metric in ['cosine', 'cos']:
		features_sim = 1 - features_dist
		if nonneg: # remove negative similarity
			features_sim = features_sim.clamp(min = 0)
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
		dist_metric = 'euclidean', reduce = 'none', dim = 3, nonneg = True):
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
		nonneg (bool): Whether to remove negative similarity.

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
		reduce = reduce, dim = dim, nonneg = nonneg, return_type = 'sparse'
	)

	return hist_sim

def calc_histology_similarity(coords, image, scale_factors,
							  dist_metric = 'euclidean', reduce = 'pca', dim = 3, nonneg = True):
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
		nonneg (bool): Whether to remove negative similarity.

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
									   reduce = reduce, dim = dim, nonneg = nonneg)

	return hist_sim


def quantile_feature_similarity_neighbor(features, coords, k, reduce = "pca", dim = 20, n_null = 100):
	"""Calculate cosine similarity of features between physical k-nearest neighbors.

	Args:
		features (2D tensor): Feature matrix, num_gene x num_spot.
		coords (2D tensor): Coordinates of spots, num_spot x 2.
		k (int): Number of neighbors.
		reduce (str): Dimension reduction method, 'pca' or 'none'.
		dim (int): Number of dimensions to reduce to.
		n_null (int): Number of null distribution samples.

	Returns:
		df (dataframe): Quantiles of cosine similarity between neighbors.
	"""
	# check input dimensions
	assert features.shape[1] == coords.shape[0], 'input dimensions do not match'

	# find the k-nearest neighbors
	nbrs = NearestNeighbors(n_neighbors = k + 1).fit(coords)
	indices = nbrs.kneighbors(coords, return_distance=False)

	# convert to indices of the sparse weight matrix
	edges = torch.tensor([[i, j] for i in range(len(indices)) for j in indices[i][1:]]).T

	# precompute dimension reduction results
	if reduce == "pca":
		features_reduced = _pca(features, dim)
	else:
		features_reduced = features

	# calculate cosine similarity between physical neighbors
	obs_sim_vec = calc_feature_similarity_sparse(
		features_reduced, edges, dist_metric="cosine",
		reduce = 'none', nonneg=False, return_type='flat'
	).float()

	# calculate background cosine similarity level
	torch.manual_seed(0)
	null_sim_list = []

	for _ in tqdm(range(n_null)):
		new_edges = torch.stack(
			[edges[0, torch.randperm(edges.shape[1])],
			 edges[1, torch.randperm(edges.shape[1])]], dim=0)

		null_sim_vec = calc_feature_similarity_sparse(
			features_reduced, new_edges, dist_metric="cosine",
			reduce = reduce, nonneg=False, return_type='flat'
		)
		null_sim_list.append(null_sim_vec)

	null_sim_vec = torch.concat(null_sim_list, dim=0).float()

	# return quantiles
	q = torch.arange(0, 1.01, 0.01)
	df = pd.DataFrame({
		'Quantile': q,
		'Observed': torch.quantile(obs_sim_vec, q),
		'Shuffled': torch.quantile(null_sim_vec, q)
	})

	df = pd.melt(df, id_vars=['Quantile'], var_name='Data', value_name='CumulativeDensity')
	return df


def quantile_feature_similarity_decay(features, coords, max_k = 50, topk = False, reduce = "pca", dim = 20):
	"""Calculate cosine similarity of features between physical neighbors of varying degrees.

	Args:
		features (2D tensor): Feature matrix, num_gene x num_spot.
		coords (2D tensor): Coordinates of spots, num_spot x 2.
		max_k (int): Maximum number of neighbors.
		topk (bool): Whether to keep all neighbors within the threshold or only the k nearest neighbors.
		reduce (str): Dimension reduction method, 'pca' or 'none'.
		dim (int): Number of dimensions to reduce to.

	Returns:
		df (dataframe): Quantiles of cosine similarity between neighbors.
	"""
	# check input dimensions
	assert features.shape[1] == coords.shape[0], 'input dimensions do not match'

	# find the k-nearest neighbors
	nbrs = NearestNeighbors(n_neighbors = max_k + 1).fit(coords)
	# indices have already been sorted by distance
	indices = nbrs.kneighbors(coords, return_distance=False)

	# precompute dimension reduction results
	if reduce == "pca":
		features_reduced = _pca(features, dim)
	else:
		features_reduced = features

	# calculate similarity for each k
	results = []
	q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])

	for k in tqdm(range(max_k + 1)): # k = 0: self
		if topk: # keep all neighbors within the threshold
			edges_k = torch.tensor([[i, j]
								  for i in range(len(indices))
								  for j in indices[i][:k+1]]).T
		else: # keep only the k nearest neighbors
			edges_k = torch.tensor([[i, j]
								  for i in range(len(indices))
								  for j in indices[i][k:(k+1)]]).T

		# calculate cosine similarity between physical neighbors
		cos_sim_k_vec = calc_feature_similarity_sparse(
			features_reduced, edges_k, dist_metric="cosine",
			reduce = 'none', return_type='flat'
		).float()

		# store quantiles
		results.append(list(torch.quantile(cos_sim_k_vec, q).numpy()) + [k])

	df = pd.DataFrame(results, columns=['Q10', 'Q25', 'Q50', 'Q75', 'Q90', 'k'])
	return df


def get_neighbor_quantile_value_by_k(sq_mat, coords, min_k = 0, max_k = 50):
	"""Extract values of neighboring pairs of spots from the square matrix.

	Args:
		sq_mat (2D tensor): Data to extract, num_spot x num_spot.
		coords (2D tensor): Coordinates of spots, num_spot x 2.
		min_k (int): Minimum number of neighbors. k = 0: self.
		max_k (int): Maximum number of neighbors.

	"""
	# find the k-nearest neighbors
	nbrs = NearestNeighbors(n_neighbors = max_k + 1).fit(coords)
	# indices have already been sorted by distance
	indices = nbrs.kneighbors(coords, return_distance=False)

	# calculate quantiles for each k
	results = []
	q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
	for eps in tqdm(range(max_k - min_k + 1)):
		edges_k = torch.tensor(
			[[i, j] for i in range(len(indices)) for j in indices[i, (min_k+eps):(min_k+eps+1)]]
		).T # num_spot x 2
		sq_mat_k = sq_mat[edges_k[0], edges_k[1]]
		results.append(list(torch.quantile(sq_mat_k, q).numpy()) + [eps + min_k])

	df = pd.DataFrame(results, columns=['Q10', 'Q25', 'Q50', 'Q75', 'Q90', 'k'])
	return df


def plot_similarity_cdf(df_k, title = "", k = None):
	"""Plot cumulative distribution of pairwise cosine similarity between neighboring spots.

	Args:
		df_k (dataframe): Quantiles of cosine similarity between neighbors.
			Output of quantile_feature_similarity_neighbor.
		title (str): Plot title.
		k (int): Number of neighbors (labeling only).
	"""
	return (ggplot(df_k, aes(y = 'CumulativeDensity', x = 'Quantile', group = 'Data', color = 'Data')) +
			geom_line() +
			geom_vline(xintercept = 0.5, linetype = "dashed", color = "gray") +
			labs(
			  x = "Quantile",
			  y = "Cumulative density of cosine similarity\nbetween neighboring spots"
	 			  f"{' (k = ' + str(k) + ')' if k is not None else ''}",
			  color = "",
			  title = title
			) +
			scale_x_continuous(breaks = [0, 0.5, 1]) +
			theme_classic()
			)

def plot_similarity_decay(df_all, title = "", plot_type = "shadow"):
	"""Plot decay of pairwise cosine similarity between neighboring spots.

	Args:
		df_all (dataframe): Quantiles of cosine similarity between neighbors.
			Output of quantile_feature_similarity_decay.
		title (str): Plot title.
		plot_type (str): Type of plot, "shadow", "boxplot" or "lineplot".
	"""
	assert plot_type in ["shadow", "boxplot", "lineplot"]

	if plot_type == "shadow":
		g = (ggplot(df_all, aes(x = 'k')) +
			 geom_ribbon(aes(ymin = 'Q10', ymax = 'Q90'), alpha = 0.2) +
			 geom_ribbon(aes(ymin = 'Q25', ymax = 'Q75'), alpha = 0.3) +
			 geom_line(aes(x = 'k', y = 'Q50'), color = "darkred")
			)
	elif plot_type == "boxplot":
		g = (ggplot(df_all, aes(x = 'k')) +
			 geom_segment(aes(x = 'k', xend = 'k', y = 'Q10', yend = 'Q90')) +
			 geom_rect(
			   aes(xmin = 'k - 0.4', xmax = 'k + 0.4', ymin = 'Q25', ymax = 'Q75'),
			   fill = "white", color = "black"
			 ) +
			 geom_segment(aes(x = 'k - 0.4', xend = 'k + 0.4', y = 'Q50', yend = 'Q50'),
			   color = "darkred"
			 ) +
			 geom_segment(aes(x = 'k - 0.4', xend = 'k + 0.4', y = 'Q10', yend = 'Q10')) +
			 geom_segment(aes(x = 'k - 0.4', xend = 'k + 0.4', y = 'Q90', yend = 'Q90'))
			)
	else:
		df_all_2 = df_all.melt(id_vars=['k'], var_name="quantile", value_name="sim")
		g = (ggplot(df_all_2, aes(x = 'k', group = 'quantile')) +
			 geom_line(aes(y = 'sim', color = 'quantile')) +
			 geom_text(
			   data = df_all_2[df_all_2['k'] == df_all_2['k'].max()],
			   mapping = aes(
				 label = 'quantile', x = 'k + 0.5', y = 'sim + 0.05',
				 color = 'quantile'
			   )
			 )
			)

	return (g +
			geom_hline(yintercept = 0, linetype = 'dashed', color = 'black') +
			labs(
			  x = "Neighborhood degree", y = "Pairwise cosine similarity",
			  title = title
			) +
			theme_classic() +
			theme(legend_position = "none")
			)

def _mono_exp(x, alpha, beta):
	return alpha * np.exp(- beta * x)

def estimate_decay_rate(x, y, return_all_params = False):
	"""Estimate decay rate of a mono-exponential decay function.

	Args:
		x (array): x values.
		y (array): y values.
		return_all_params (bool): If False, only return decay rate (beta).
	"""

	# estimate initial condition
	a0, t0 = y.max(), y.min()
	p0 = (a0, np.log(a0 - t0 + 1e-5) / x.max())

	# fit the exponential model
	try:
		params, _ = curve_fit(_mono_exp, x, y, p0)
	except:
		params, _ = curve_fit(_mono_exp, x, y, (1, 0.05), maxfev=5000)

	if return_all_params:
		return params
	else: # return decay rate (beta)
		return params[1]
