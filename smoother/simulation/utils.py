"""
Generate ST count matrix based on the assigned pattern.
Some functions adapted from the cell2location paper:
    Kleshchevnikov, Vitalii, et al. "Cell2location maps fine-grained cell types in spatial transcriptomics."
    Nature biotechnology 40.5 (2022): 661-671.
    https://www.nature.com/articles/s41587-021-01139-4#
"""
from re import sub
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,lil_matrix
import anndata
from tqdm import tqdm
from smoother.weights import coordinate_to_weights_knn_dense


def sample_cell_indices(generation_snrna, annot_label, cell_count_df, cell_capture_eff_df):
    '''
    Sample cell indices function adapted from cell2location paper
    (https://github.com/vitkl/cell2location_paper/blob/master/notebooks/benchmarking/synthetic_data_construction_improved_real_mg.ipynb)

    Args:
        generation_snrna: reference single-cell adata.
        cell_count_df, cell_capture_eff_df: Array of n_spots x n_cell_types

    Returns:
        cell index matrix (n_spots x n_cells).
    '''
    # num_spots x num_cells_in_train_set
    locations2cells = np.zeros((cell_count_df.shape[0], generation_snrna.shape[0]))

    for i, l in enumerate(cell_count_df.index): # locations
        for ct in cell_count_df.columns: # celltypes
            cell_ind_all = generation_snrna.obs['cell_ind']
            cell_ind_all = cell_ind_all[generation_snrna.obs[annot_label] == ct]
            # assign single cells to each location
            cell_ind = np.random.choice(cell_ind_all, int(cell_count_df.loc[l, ct]), replace=False)

            # scale by capture efficiency
            locations2cells[i,cell_ind] = cell_capture_eff_df.loc[l, ct]

    return csr_matrix(locations2cells)

def diag_block_mat_boolindex(L):
    '''
    Helper function to generate diagonal matrix across n_experiments
    '''
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=float)
    out[mask] = np.concatenate(L).ravel()
    return csr_matrix(out)

def smooth_by_neighbors(shared_prop, locations2cells_matrix, coords, n_experiments):
    '''
    Smoothing the spot x cell-index matrix by neighoring spots

    Args:
        shared_prop: proportion of cells that come from neighboring spots.
        locations2cells_matrix: binary spot x cell-index matrix,
            ((n_spots * n_experiments) x n_cells).
        coords: n_spots x 2 (x and y coordinates) grids, the same for all experiments.
        n_experiments: number of experiments.

    Returns:
        smoothed cell index matrix (n_spots x n_cells).
    '''
    # compute adjacency matrix
    swm = coordinate_to_weights_knn_dense(coords, k=4, row_scale=True) * shared_prop
    swm = swm.fill_diagonal_(1 - shared_prop)

    # compute smoothed cell index matrix
    swm_all_exp = diag_block_mat_boolindex([swm for i in range(n_experiments)])
    locations2cells_smoothed = swm_all_exp.dot(locations2cells_matrix)
    return csr_matrix(locations2cells_smoothed)

def create_spatial_adata(adata, locations2cells_matrix,
                         cell_counts, cell_abundances, capture_eff,
                         locations, cell_types2zones_df):
    '''
    Construct count matrix using locations x cell_indice matrix and reference anndata

    Args:
        adata: refrence adata where the raw counts are stored in adata.X
        locations2cells_matrix: cell index matrix (n_spots x n_cells)
        locations: (n_experiments * n_spots) x 2 (x and y coordinates) grids
        cell_abundances,cell_count_df, cell_capture_eff_df: Array of n_spots x n_cell_types
        cell_types2zones_df: n_cell_types x n_zones matrix

    Returns:
        synthetic anndata object
    '''
    print("Computing synthetic counts" )
    synthetic_counts = locations2cells_matrix.dot(adata.X)

    print("Constructing adata object......")
    # Create adata object
    synth_adata = anndata.AnnData(synthetic_counts, dtype = np.dtype('float'))
    synth_adata.obs_names = cell_counts.index
    synth_adata.var_names = adata.var_names
    synth_adata.obs[[f'cell_count_{ct}' for ct in cell_counts.columns]] = cell_counts
    synth_adata.obs[[f'cell_abundances_{ct}' for ct in cell_abundances.columns]] = cell_abundances
    synth_adata.obs[[f'cell_capture_eff_{ct}' for ct in capture_eff.columns]] = capture_eff
    synth_adata.obsm['X_spatial'] = locations
    synth_adata.uns['design'] = {'cell_types2zones': cell_types2zones_df}
    synth_adata.obs['sample'] = [sub('_location.+$','',i) for i in synth_adata.obs_names]
    return synth_adata

def cal_counts_ct(synth_adata,
                  orig_adata,
                  annot_label,
                  locations2cells_matrix):
    '''
    Compute synthetic counts per cell type

    Args:
        synth_adata: synthetic adata
        orig_adata: reference adata
        annot_label: key to access celltype information
        locations2cells_matrix: cell index matrix (n_spots x n_cells)

    Returns:
        updated synth_adata with UMI per cell_type stored in obs

    '''
    cell_types = np.unique(orig_adata.obs[annot_label])
    for _, ct in tqdm(enumerate(cell_types), total = len(cell_types)):
        select = np.where(orig_adata.obs[annot_label]==ct)[0]
        locations2cells_ct = lil_matrix(np.zeros(locations2cells_matrix.shape))
        locations2cells_ct[:,select] = locations2cells_matrix[:,select]
        synthetic_counts_ct = locations2cells_ct.tocsr().dot(orig_adata.X)
        synth_adata.obs[f'UMI_count_{ct}'] = np.array(synthetic_counts_ct.sum(1)).flatten()

    return synth_adata

def add_gamma_noises(synth_adata,
                     shape = 0.4586,
                     scale=1/0.6992
                    ):
    '''
    Fuction for adding noises adapted from cell2location paper
    (https://github.com/vitkl/cell2location_paper/blob/master/notebooks/benchmarking/synthetic_data_construction_improved_real_mg.ipynb)

    Args:
        synth_adata: synthetic adata
        shape: shape of gamma distribution
        scale: scale of gamma distribution

    Returns:
        synthetic adata
    '''
    # Sample detection rates
    gene_level = np.random.gamma(shape,scale=scale,
                                 size=(1, synth_adata.shape[1])) / 2
    synth_adata.var['gene_level'] = gene_level.flatten()
    synth_adata.X = synth_adata.X.toarray() * gene_level

    # Sample poisson integers
    synth_adata.layers['expression_levels'] = synth_adata.X
    synth_adata.X = np.random.poisson(synth_adata.X)
    synth_adata.X = csr_matrix(synth_adata.X)

    return synth_adata

def grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None):
    '''
    Group referenceobs by celltypes and output group mean for deconvolution

    Args:
        adata: reference anndata
        group_key: key to access celltype information
        layer: anndata layer that stores the feature matrix to calculate group mean
        gene_symbols: genes to keep; keep all genes when "none"

    Returns:
        gene x cell_type count matrix
    '''
    if layer is not None: # if layer is provided, use it
        get_x = lambda x: x.layers[layer]
    else: # load data stored in adata.X
        get_x = lambda x: x.X
    if gene_symbols is not None:
        feature_idx = gene_symbols
    else:
        feature_idx = adata.var_names

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((len(feature_idx), len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys())
    )

    for group, idx in grouped.indices.items():
        ref_counts = get_x(adata[idx, feature_idx])
        out[group] = np.ravel(ref_counts.mean(axis=0, dtype=np.float64))

    out.index = feature_idx
    return out
