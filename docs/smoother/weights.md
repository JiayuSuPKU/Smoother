Module smoother.weights
=======================
Calculate spatial weight matrix from coordinates, histology images, and transcriptomic data

Functions
---------

    
`calc_feature_similarity(features: torch.Tensor, dist_metric='cosine', reduce='pca', dim=10)`
:   Calculate pairwise feature similarity between spots.
    
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

    
`calc_histology_similarity(coords, image, scale_factors, dist_metric='euclidean', reduce='pca', dim=3)`
:   Calculate pairwise histology similarity between spots.
    
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

    
`calc_weights_spagcn(coords, image, scale_factors, histology_axis_scale=1.0, band_width=1.0)`
:   Calculate spatial weight matrix similar to the SpaGCN edge weight.
    
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

    
`coordinate_to_weights_dist(coords, scale_coords=True, q_threshold=0.001, band_width=0.1, dist_metric='euclidean', row_scale=True)`
:   Calculate spatial weight matrix using distance band.
    
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

    
`coordinate_to_weights_knn(coords, k=6, symmetric=True, row_scale=True)`
:   Calculate spatial weight matrix using k-nearest neighbours.
    
    Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
    interactions among k-nearest neighbours.
    
    Args:
            coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
            k (int): Number of nearest neighbours to keep.
            symmetric (bool): If True only keep mutual neighbors.
            row_scale (bool): If True scale row sum of the spatial weight matrix to 1.
    
    Returns:
            weights: A 2D tensor containing spatial weights, num_spot x num_spot.

    
`get_histology_vector(image, x_pixel, y_pixel, spot_radius, scale_factor, padding=True)`
:   Get the histology image vector of one spot.
    
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

    
`weight_to_inv_cov(weights, model, l=1, standardize=False)`
:   Convert a spatial weight matrix to an inverse covariance matrix.
    
    Calculate the covariance structure using spatial weights. Different spatial process models
    impose different structures. Check model descriptions for more details.
    
    Args:
            weights (2D tensor): Spatial weight matrix, num_spot x num_spot.
            model (str): Spatial process model to use, can be one of 'sma','sar', 'isar', 'car', 'icar'.
            l (float): Smoothing effect size.
            standardize (bool): If True, return the standardized inverse covariance matrix (inv_corr).
    
    Returns:
            inv_cov (2D tensor): An inverse covariance (precision) matrix, num_spot x num_spot.

Classes
-------

`SpatialWeightMatrix()`
:   Spatial weight matrix.
    
    The adjacency matrix that specifies connectivities and interactions between
    each pair of spots.
    
    Attributes:
            swm (2D tensor): Unscaled spatial weight matrix.
            swm_scaled (2D tensor): Spatial weight matrix scaled with external information
                    (e.g., expression, histology).
            inv_covs (dict): Cached inverse covariance matrices under different model settings
                    (for debugging).
            config (dict): Configurations.

    ### Methods

    `calc_weights_dist(self, coords, scale_coords=True, q_threshold=0.001, band_width=0.1, dist_metric='euclidean', row_scale=False) ‑> None`
    :   Calculate spatial weight matrix using distance band.
        
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

    `calc_weights_knn(self, coords, k=6, symmetric=True, row_scale=False) ‑> None`
    :   Calculate spatial weight matrix using k-nearest neighbours.
        
        Convert spatial coordinate to a spatial weight matrix where non-zero entries represent
        interactions among k-nearest neighbours.
        
        Args:
                coords (2D array): Spatial coordinates, num_spot x 2 (or 3).
                k (int): Number of nearest neighbours to keep.
                symmetric (bool): If True only keep mutual neighbors.
                row_scale (bool): If True scale row sum of the spatial weight matrix to 1.

    `calc_weights_spagcn(self, coords, image, scale_factors: dict, histology_axis_scale=1.0, band_width=1.0)`
    :   Calculate pairwise histology similarity between spots.
        
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

    `get_inv_cov(self, model, l=1, cached=True, standardize=False)`
    :   Calculate or extract cached inverse covariance matrix.
        
        Args:
                model (str): The spatial process model, can be one of 'sma','sar', 'car', 'icar'.
                l (float): Smoothing effect size.

    `scale_by_expr(self, expr, dist_metric='cosine', reduce='pca', dim=10, row_scale=False) ‑> None`
    :   Scale weight matrix using transcriptional similarity.
        
        Args:
                expr (2D array): Spatial gene expression count matrix, num_genes x num_spot.
                dist_metric (str): Distance metric.
                reduce (str): If `PCA`, calculate distance on the reduced PCA space.
                dim (int): Number of dimension of the reduced space.
                row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.

    `scale_by_histology(self, coords, image, scale_factors: dict, dist_metric='euclidean', reduce='pca', dim=10, row_scale=False)`
    :   Calculate pairwise histology similarity between spots.
        
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

    `scale_by_identity(self, spot_ids, boundary_connectivity=0, row_scale=False, return_swm=False)`
    :   Scale spatial weight matrix by spot identity.
        
        Args:
                spot_ids (1D array): Spot identity of length num_spot.
                boundary_connectivity (float): Connectivity of spots with different identities.
                        If 0 (default), no interaction across identities.

    `scale_by_similarity(self, pairwise_sim: torch.Tensor, row_scale=False, return_swm=False)`
    :   Scale spatial weight matrix by external pairwise similarity.
        
        Args:
                pairwise_sim (2D tensor): External pairwise similarity, num_spot x num_spot.
                row_scale (bool): If True, scale rowsums of spatial weight matrix to be 1.
                return_swm (bool): Whether to output the scaled spatial weight matrix.