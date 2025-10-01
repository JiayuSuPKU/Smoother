"""
Visualization functions for smoother.models.deconv
"""

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
from skbio.stats.composition import clr, ilr
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotnine import *
from smoother.utils import normalize_minmax

def plot_celltype_props(p_inf, coords, cell_type_names = None, n_col = 4, figsize = None):
    """Plot the deconvolution results.

    Args:
        p_inf: n_spots x n_groups. The inferred cell-type proportions.
        coords: n_spots x 2. The coordinates of the spots.
        cell_type_names: list of str. The names of the cell types.
        n_col: int. The number of columns in the figure.
        figsize: tuple. The size of the figure.
    """
    if cell_type_names is None:
        cell_type_names = [f"Cell type {i}" for i in range(p_inf.shape[1])]

    if isinstance(p_inf, pd.DataFrame):
        p_inf = p_inf.to_numpy()

    # set figure configurations
    assert len(cell_type_names) == p_inf.shape[1]
    n_row = int(np.ceil(p_inf.shape[1] / n_col))
    if n_col > len(cell_type_names):
        n_col = p_inf.shape[1]

    if figsize is None:
        figsize = (4 * n_col, 4 * n_row)

    # plot the results
    fig, axes = plt.subplots(n_row, n_col, figsize = figsize)

    # iterate through each cell type
    for ind, (name, ax) in enumerate(zip(cell_type_names, axes.flatten())):
        ax.set_title(f"{name}")
        p = ax.scatter(coords.iloc[:, 0], coords.iloc[:,1], c = p_inf[:, ind], s = 0.5)

        # add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(p, cax=cax, orientation = 'vertical')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')

    plt.show()


def clr_stable(props, epsilon = 1e-8):
    """Apply centre log ratio transform (clr) to transform proportions to the real space.

    Args:
        props: n_spots x n_groups. Rowsum equals to 1 or 0. If 0, the transformed vector
            will also be the zero vector.
    """
    return clr(props + epsilon)

def ilr_stable(props, epsilon = 1e-8):
    """Apply isometric log ratio transformation (ilr) to transform proportions to the real space."""
    return ilr(props + epsilon)

def cluster_features(features, transform = 'pca',
                     n_neighbors = 15, res = 1) -> pd.Series:
    """Leiden clustering on the input features."""

    # create temporary adata to calculate the clustering
    # features: n_spots x n_groups
    if isinstance(features, pd.DataFrame):
        features = features.to_numpy()
    elif isinstance(features, torch.Tensor):
        features = features.numpy()

    features = features.copy()


    assert transform in ['pca', 'clr', 'ilr']
    if transform == 'clr':
        adata = ad.AnnData(clr_stable(features), dtype = np.float64)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    elif transform == 'ilr':
        adata = ad.AnnData(ilr_stable(features), dtype = np.float64)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    else:
        adata = ad.AnnData(features, dtype = np.float64)
        # run pca
        sc.pp.scale(adata)
        sc.pp.pca(adata, n_comps=min(20, features.shape[1] - 1))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # perform leiden clustering
    sc.tl.leiden(adata, resolution=res)

    return adata.obs["leiden"]


def _get_cost_matrix(clu1, clu2):
    """Helper function to get the cost matrix for aligning two clusterings."""
    n_clu1 = len(clu1.unique())
    n_clu2 = len(clu2.unique())
    cm = np.zeros((n_clu1, n_clu2))

    # calculate cost as the number of spots that were not assigned
    # to the same cluster in the two clustering results
    for i in range(n_clu1):
        clu_ind = (clu1 == str(i))
        for j in range(n_clu2):
            cm[i, j] = -(clu2[clu_ind] == str(j)).sum()
    return cm


def align_clusters(clu_list, ref_ind = None):
    """Align the clusterings in clu_list to the reference clustering.

    Args:
        clu_list: list of clustering result.
        ref_ind: int. The index of the reference clustering. If None,
            use the clustering with the largest number of clusters.
    """
    n_clu_list = [len(clu.unique()) for clu in clu_list]
    if ref_ind == None:
        ref_ind = np.argmax(n_clu_list)

    clu_aligned_list = []
    for i, clu in enumerate(clu_list):
        if i == ref_ind:
            clu_aligned_list.append(clu)
        else:
            # set random seed for reproducibility
            np.random.seed(0)

            # align the current clustering to the reference clustering
            cm = _get_cost_matrix(clu, clu_list[ref_ind])
            _, col_ind = linear_sum_assignment(cm)

            # rename the categories
            clu_aligned_list.append(
                clu.cat.rename_categories(col_ind).astype(str)
            )

    return clu_aligned_list, ref_ind


def plot_spatial_clusters(clu_aligned_list, coords, names = None, n_col = 4):
    """Plot clusters.

    Args:
        clu_aligned_list: list of aligned clusterings.
        coords: n_spots x 2. Coordinates of the spots.
        names: list of str. Names of the deconvolution model.
        n_col: int. Number of columns in the plot.
    """
    if names is None:
        names = [f'Cluster {i}' for i in range(len(clu_aligned_list))]

    if isinstance(coords, pd.DataFrame):
        coords = coords.values

    clu_aligned_arr = np.stack(clu_aligned_list, axis = 1)
    df_clu_aligned = pd.DataFrame(clu_aligned_arr, columns = names)
    df_clu_aligned = pd.concat([pd.DataFrame(coords, columns = ['x', 'y']), df_clu_aligned], axis = 1)
    df_clu_aligned = pd.melt(df_clu_aligned, id_vars=['x', 'y'], var_name = 'Method', value_name = 'Cluster')
    df_clu_aligned['Method'] = df_clu_aligned['Method'].astype('category')
    df_clu_aligned['Method'] = df_clu_aligned['Method'].cat.reorder_categories(names)

    # plot the results
    p = (
        ggplot(df_clu_aligned, aes(x = 'x', y = 'y', fill = 'Cluster')) +
        facet_wrap('Method', ncol=n_col) +
        geom_tile() +
        theme_void()
    )

    return p


def cluster_and_plot_celltype_props(p_inf_list, coords, names = None, n_col = 4,
                                    transform = 'pca', n_neighbors = 15, res = 1,
                                    return_clu = False):
    """Cluster the cell-type proportions and visualize the results.

    Args:
        p_inf_list: list of cell-type proportions.
        coords: n_spots x 2. Coordinates of the spots.
        names: list of str. Names of the deconvolution model.
        transform: str. Transformation to apply to the cell-type proportions.
            'pca', 'clr', 'ilr'.
        n_neighbors: int. Number of neighbors to use for clustering.
        res: float. Resolution for leiden clustering.
        return_clu: bool. Whether to return the clustering results.
    """

    if names is None:
        names = [f'Cluster {i}' for i in range(len(p_inf_list))]

    if isinstance(coords, pd.DataFrame):
        coords = coords.values

    # scale the coordinates for geom_tile
    coords = normalize_minmax(coords)

    # cluster cell type proportions
    clu_inf_list = [
        cluster_features(p_inf, transform = transform, n_neighbors=n_neighbors, res = res)
        for p_inf in p_inf_list
    ]
    # align the clustering results
    clu_aligned_list, _ = align_clusters(clu_inf_list)

    # plot the results
    p = plot_spatial_clusters(clu_aligned_list, coords, names = names, n_col = n_col)

    if return_clu:
        return p, clu_aligned_list

    return p