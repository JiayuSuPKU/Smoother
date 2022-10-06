Module smoother.simulation.utils
================================
Generate ST count matrix based on the assigned pattern.
Some functions adapted from the cell2location paper:
        Kleshchevnikov, Vitalii, et al. "Cell2location maps fine-grained cell types in spatial transcriptomics."
        Nature biotechnology 40.5 (2022): 661-671.
        https://www.nature.com/articles/s41587-021-01139-4#

Functions
---------

    
`add_gamma_noises(synth_adata, shape=0.4586, scale=1.4302059496567505)`
:   Fuction for adding noises adapted from cell2location paper
    (https://github.com/vitkl/cell2location_paper/blob/master/notebooks/benchmarking/synthetic_data_construction_improved_real_mg.ipynb)
    
    Args:
            synth_adata: synthetic adata
            shape: shape of gamma distribution
            scale: scale of gamma distribution
    
    Returns:
            synthetic adata

    
`cal_counts_ct(synth_adata, orig_adata, annot_label, locations2cells_matrix)`
:   Compute synthetic counts per cell type
    
    Args:
            synth_adata: synthetic adata
            orig_adata: reference adata
            annot_label: key to access celltype information
            locations2cells_matrix: cell index matrix (n_spots x n_cells)
    
    Returns:
            updated synth_adata with UMI per cell_type stored in obs

    
`create_spatial_adata(adata, locations2cells_matrix, cell_counts, cell_abundances, capture_eff, locations, cell_types2zones_df)`
:   Construct count matrix using locations x cell_indice matrix and reference anndata
    
    Args:
            adata: refrence adata where the raw counts are stored in adata.X
            locations2cells_matrix: cell index matrix (n_spots x n_cells)
            locations: (n_experiments * n_spots) x 2 (x and y coordinates) grids
            cell_abundances,cell_count_df, cell_capture_eff_df: Array of n_spots x n_cell_types
            cell_types2zones_df: n_cell_types x n_zones matrix
    
    Returns:
            synthetic anndata object

    
`diag_block_mat_boolindex(L)`
:   Helper function to generate diagonal matrix across n_experiments

    
`grouped_obs_mean(adata, group_key, layer=None, gene_symbols=None)`
:   Group referenceobs by celltypes and output group mean for deconvolution
    
    Args:
            adata: reference anndata
            group_key: key to access celltype information
            layer: anndata layer that stores the feature matrix to calculate group mean
            gene_symbols: genes to keep; keep all genes when "none"
    
    Returns:
            gene x cell_type count matrix

    
`sample_cell_indices(generation_snrna, annot_label, cell_count_df, cell_capture_eff_df)`
:   Sample cell indices function adapted from cell2location paper
    (https://github.com/vitkl/cell2location_paper/blob/master/notebooks/benchmarking/synthetic_data_construction_improved_real_mg.ipynb)
    
    Args:
            generation_snrna: reference single-cell adata.
            cell_count_df, cell_capture_eff_df: Array of n_spots x n_cell_types
    
    Returns:
            cell index matrix (n_spots x n_cells).

    
`smooth_by_neighbors(shared_prop, locations2cells_matrix, coords, n_experiments)`
:   Smoothing the spot x cell-index matrix by neighoring spots
    
    Args:
            shared_prop: proportion of cells that come from neighboring spots.
            locations2cells_matrix: binary spot x cell-index matrix,
                    ((n_spots * n_experiments) x n_cells).
            coords: n_spots x 2 (x and y coordinates) grids, the same for all experiments.
            n_experiments: number of experiments.
    
    Returns:
            smoothed cell index matrix (n_spots x n_cells).