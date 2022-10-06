Module smoother.simulation.histology
====================================
Extract spatial patterns from a given histology image

Functions
---------

    
`cal_membership_value(membership_mat, feature_df, cluster_centers, m)`
:   Calculate fuzzy-c-means membership values
    
    Args:
            membership_mat: membership matrix n_location x n_cluster
            feature_df: features matrix n_location x n_feature
            cluster_centers: center point for clusters n_cluster x n_feature
            m: m value required by FCM
    
    Returns:
            membership_mat: updated membership matrix n_location x n_cluster

    
`extract_features_from_image(image: squidpy.im._container.ImageContainer, crop_size, layer)`
:   Extract features from a given image
    
    Args:
            image: squidpy ImageContainer
            crop_size: the # of crops wanted [n_row_crops, n_col_crops]
            layer: Image layer in image that should be processed
    
    Returns:
            features_df: the extracted features at all crops

    
`get_features_matrix(crop: squidpy.im._container.ImageContainer, layer)`
:   Extract features from a given crop
    
    Args:
            crop: crop of a image
            layer: Image layer in crop that should be processed
    
    Returns:
            df_1: the extracted features at this crop