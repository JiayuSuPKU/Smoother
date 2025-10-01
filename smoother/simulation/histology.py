"""
Extract spatial patterns from a given histology image
"""

import warnings
import math
import operator
from tqdm import tqdm
import pandas as pd
import numpy as np

try:
    from squidpy.im import ImageContainer
except ImportError:
    warnings.warn("Package 'Squidpy' not installed.", category=ImportWarning)
    ImageContainer = None

def get_features_matrix(crop : ImageContainer, layer):
    '''
    Extract features from a given crop

    Args:
        crop: crop of a image
        layer: Image layer in crop that should be processed

    Returns:
        df_1: the extracted features at this crop
    '''
    text = crop.features_texture(layer = layer)
    hist = crop.features_histogram(layer = layer)
    summ = crop.features_summary(layer = layer)
    dict_1 = text | hist| summ
    df_1 = pd.DataFrame(dict_1.items()).transpose()
    df_1.columns = df_1.iloc[0]
    return df_1.drop(df_1.index[0])

def extract_features_from_image(image : ImageContainer, crop_size, layer):
    '''
    Extract features from a given image

    Args:
        image: squidpy ImageContainer
        crop_size: the # of crops wanted [n_row_crops, n_col_crops]
        layer: Image layer in image that should be processed

    Returns:
        features_df: the extracted features at all crops
    '''
    features_df = pd.DataFrame()
    total = int(image.shape[0]/crop_size[0]) * int((image.shape[1]/crop_size[1]))
    for crop in tqdm(image.generate_equal_crops(size = crop_size), total = total):
        features_df_1 = get_features_matrix(crop, layer)
        features_df = pd.concat((features_df, features_df_1), axis=0)
    features_df.index = [f"location_{i}" for i in range(len(features_df))]
    return features_df

def cal_membership_value(membership_mat, # membership matrix n_location x n_cluster
                         feature_df, # features matrix n_location x n_feature
                         cluster_centers, # center point for clusters n_cluster x n_feature
                         m): # Calculating the membership value
    '''
    Calculate fuzzy-c-means membership values

    Args:
        membership_mat: membership matrix n_location x n_cluster
        feature_df: features matrix n_location x n_feature
        cluster_centers: center point for clusters n_cluster x n_feature
        m: m value required by FCM

    Returns:
        membership_mat: updated membership matrix n_location x n_cluster
    '''
    p = float(2/(m-1))
    n = membership_mat.shape[0]
    k = cluster_centers.shape[0]
    for i in range(n):
        x = list(feature_df.iloc[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_centers[j]))))
                        for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat
