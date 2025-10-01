import os
import argparse
import numpy as np
import scipy.io as sio
import pandas as pd
import scanpy as sc

from sklearn.model_selection import train_test_split

from smoother.simulation.gp import generate_grid
from smoother.simulation.utils import add_gamma_noises, cal_counts_ct, \
    create_spatial_adata, grouped_obs_mean, sample_cell_indices, smooth_by_neighbors


def main():

    prs = argparse.ArgumentParser()

    # IO arguments
    prs.add_argument('-i','--abundance_dir',
                     type = str,
                     required = True,
                     help = 'Path to zone abundance matrix (.csv).')
    prs.add_argument('-d','--ref_dir',
                     type = str,
                     required = True,
                     help = 'Path to reference single cell dataset (anndata).')
    prs.add_argument('-l','--ann_label',
                     type = str,
                     required = True,
                     help = 'Key to access celltype information in the reference anndata.')
    prs.add_argument('-o','--out_dir',
                     type = str,
                     default = None,
                     help = 'Output directory. Default: None to use the same directory as the input.')
    prs.add_argument('-oa','--output-anndata', action='store_true',
                     help = 'Whether to output entire anndata objects for synthetic spatial data ' \
                            'and paired single cell data. Default: True.' )

    # Simulation arguments
    prs.add_argument('-ne','--n_experiments',
                     type = int,
                     default = 1,
                     help = 'Number of experiments to simulate. Default: 1.')
    prs.add_argument('-ns','--grid_size',
                     nargs=2,
                     type=int,
                     default=[50,50],
                     help = 'Grid size, needs to be 2 int n_row, n_col. ' \
                            'Default: 50 50.')
    prs.add_argument('-s','--seed',
                     default = 253286,
                     type = int,
                     help = 'Random seed. Default: 253286.')
    prs.add_argument('-sp','--split', action=argparse.BooleanOptionalAction, default=True,
                     help = 'Split training and testing set to seperate single cells used ' \
                            'in data simulation and deconvolution. Default: True.' )
    prs.add_argument('-ph','--p_high_density',
                     default = 0.4,
                     type = float,
                     help = 'Proportion of high-density cell types in each zone. Default: 0.4.')
    prs.add_argument('-muh','--mu_high_density',
                     default = 4.0,
                     type = float,
                     help = 'Average abundance for high-density cell types. Default: 4.0.')
    prs.add_argument('-mul','--mu_low_density',
                     default = 0.4,
                     type = float,
                     help = 'Average abundance for low-density cell types. Default: 0.4.')
    prs.add_argument('-f','--multi-pattern', action=argparse.BooleanOptionalAction, default=False,
                     help = 'Flag specifies whether one celltype can have several spatial patterns ' \
                            'number of patterns per celltype is sampled with a Gamma distribution. ' \
                            'Default: False.')

    # Spatial noise arguments
    prs.add_argument('-sm','--smooth_scale',
                     default = 0.0,
                     type = float,
                     help = 'Scale for smoothing from neighboring spots specifying the ' \
                            'proportion of cells that are shared with neighboring spots. '\
                            'Default: 0.')
    prs.add_argument('-gn','--gamma_noises', action=argparse.BooleanOptionalAction, default=True,
                     help = 'Add gamma noises to the synthetic count matrix. Default: True.' )

    # Marker selection arguments
    prs.add_argument('-lfcmin', '--log2fc_min',
                     type=float,
                     default=1.0,
                     help = 'Minimum log2fc for marker gene selection. Default: 1.0.')
    prs.add_argument('-pv', '--p_value',
                     type=float,
                     default=0.01,
                     help = 'P-value threshold for marker gene selection. Default: 0.01.')
    prs.add_argument('-nm','--n_markers',
                     type=int,
                     default=0,
                     help = 'Number of marker genes per celltype in the simulated count matrix. ' \
                            'Default: 0 to return all genes passed log2fc_min and p_value_threshold.')
    # extract arguments
    args = prs.parse_args()

    scref_dir = args.ref_dir
    zone_ab_dir = args.abundance_dir
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(zone_ab_dir)
        if out_dir == '':
            out_dir = '.'
    abundances_df = pd.read_csv(zone_ab_dir, index_col=0) # n_spot x n_zone

    n_zones = abundances_df.shape[1]
    n_ubiquitous_zones = abundances_df.columns.str.contains('uniform').sum()
    n_regional_zones = n_zones - n_ubiquitous_zones
    annot_label = args.ann_label
    output_anndata = bool(args.output_anndata)

    n_experiments = args.n_experiments
    grid_size = args.grid_size
    seed = args.seed
    flag_split = bool(args.split)
    p_high_density = args.p_high_density
    mu_high_density = args.mu_high_density
    mu_low_density = args.mu_low_density
    flag_multi_pattern = bool(args.multi_pattern)
    smooth_scale = args.smooth_scale
    gamma_noises = bool(args.gamma_noises)

    log2fc_min = args.log2fc_min
    p_value_threshold = args.p_value
    n_markers = args.n_markers

    # create output directories
    if out_dir != '.':
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + '/deconv_inputs', exist_ok=True)

    ###############################################################################
    # 1. Prepare single-cell reference data
    ###############################################################################

    # load reference dataset
    adata_snrna_raw = sc.read(scref_dir)
    sc.pp.filter_cells(adata_snrna_raw, min_genes=200)
    sc.pp.filter_genes(adata_snrna_raw, min_cells=10)
    cell_types = np.unique(adata_snrna_raw.obs[annot_label])

    # preprocess single cell data
    sc._settings.ScanpyConfig(verbosity="error")
    np.random.seed(seed)
    # store raw counts data
    adata_snrna_raw.layers['counts'] = adata_snrna_raw.X.copy()
    # normalize by total counts per cell
    sc.pp.normalize_total(adata_snrna_raw, target_sum=1e4)
    adata_snrna_raw.layers['norm_counts'] = adata_snrna_raw.X.copy()
    # log transform
    sc.pp.log1p(adata_snrna_raw)
    adata_snrna_raw.layers['log1p_norm_counts'] = adata_snrna_raw.X.copy()

    # split the reference into training and validation set so that the simulated spatial data
    # does not use exactly the same cells as in the paired reference)
    if flag_split:
        np.random.seed(seed)
        idx = np.arange(len(adata_snrna_raw))
        train_idx, val_idx = train_test_split(
            idx, train_size=0.5,
            shuffle=True, stratify=adata_snrna_raw.obs[annot_label]
        )
        # make sure all cell types are included in training set
        assert adata_snrna_raw[train_idx].obs[annot_label].value_counts().shape[0] == len(cell_types), \
            "Not enough cells for splitting training set in at least one cell type!"
    else:
        train_idx = idx
        val_idx = idx

    generation_scref = adata_snrna_raw[train_idx] # sc ref used for spatial data simulation
    paired_scref = adata_snrna_raw[val_idx] # sc ref used as paired reference in deconvolution

    # find marker genes from the single cell reference
    sc.tl.rank_genes_groups(generation_scref, annot_label, method='wilcoxon')
    markers_df = sc.get.rank_genes_groups_df(generation_scref, group = None,
                                             pval_cutoff = p_value_threshold, log2fc_min = log2fc_min)

    if n_markers > 0: # select n_markers genes for each cell type (overlap allowed)
        markers_df = markers_df.sort_values(['group', 'logfoldchanges'],
                                            ascending=False).groupby('group').head(n_markers)
        markers_df = markers_df.pivot_table(index='names', columns='group',
                                            values='logfoldchanges', fill_value=0)
    else: # keep highly specific markers only
        markers_df = markers_df.pivot_table(index='names', columns='group',
                                    values='logfoldchanges', fill_value=0)
        markers_df = markers_df.loc[((markers_df >= log2fc_min).sum(1) == 1),:]

    markers_df.to_csv(out_dir + "/markers_lfc.csv")

    marker_names = markers_df.index

    # save ref count matrix
    ref_exp_avg_raw = grouped_obs_mean(paired_scref, group_key=annot_label,
                                       layer='counts', gene_symbols=marker_names)
    ref_exp_avg_norm = grouped_obs_mean(paired_scref, group_key=annot_label,
                                        layer='norm_counts', gene_symbols=marker_names)
    ref_exp_avg_raw.to_csv(out_dir + "/deconv_inputs/ref_avg_raw_count_markers.csv")
    ref_exp_avg_norm.to_csv(out_dir + "/deconv_inputs/ref_avg_norm_count_markers.csv")

    ###############################################################################
    # 2. Assign spatial abundances to cell types
    ###############################################################################

    # assign celltype to zones
    np.random.seed(seed)
    n_cell_types = len(cell_types)

    # assign celltypes to different distribution patterns
    # (uniform vs regional, high vs low density etc.)
    n_uniform_celltypes = n_ubiquitous_zones
    n_sparse_celltypes = n_cell_types - n_uniform_celltypes # number of regional celltypes

    uniform_cell_types = np.random.choice(cell_types, n_ubiquitous_zones, replace=False)
    sparse_cell_types = cell_types[~np.isin(cell_types, uniform_cell_types)]
    cell_types = np.concatenate([sparse_cell_types, uniform_cell_types])

    np.random.seed(seed)
    # assign some regional celltypes to multiple spatial patterns
    if flag_multi_pattern:
        n_zones_per_cell_type = np.random.binomial(n_regional_zones, 0.02, size=n_sparse_celltypes) + 1
    else:
        n_zones_per_cell_type = np.repeat(1, n_sparse_celltypes)

    # generate matrix of which cell types are in which zones (n_cell_types x n_zones)
    cell_types2zones = pd.DataFrame(
        0, index=cell_types,
        columns=[f'tissue_zone_{i}' for i in range(n_regional_zones)] \
        + [f'uniform_{i}' for i in range(n_ubiquitous_zones)])

    # regional celltypes
    for i, n in enumerate(n_zones_per_cell_type):
        pos = np.random.randint(n_regional_zones, size=n)
        cell_types2zones.iloc[i,pos] = 1

    # ubiquitous celltypes
    for i in range(n_ubiquitous_zones):
        cell_types2zones.iloc[n_sparse_celltypes + i, n_regional_zones + i] = 1

    # split celltype high/low abundances in each zone
    np.random.seed(seed)
    # assign cell types to either high or low density, balanced by uniform / tissue zone
    high_density_cell_types = list(np.random.choice(uniform_cell_types,
                                                    int(np.round(n_ubiquitous_zones * p_high_density)),
                                                    replace=False))
    for zone, n_ct in cell_types2zones.sum(0).items():

        ct = list(np.random.choice(cell_types2zones.index[cell_types2zones[zone] > 0],
                                                    int(np.round(n_ct * p_high_density)),
                                                    replace=False))
        high_density_cell_types = high_density_cell_types + ct

    low_density_cell_types = cell_types[~np.isin(cell_types, high_density_cell_types)]

    # generate average abundance for low and high density cell types
    mean_var_ratio = 5
    np.random.seed(seed)
    # low density cell types
    cell_types2zones.loc[low_density_cell_types] = cell_types2zones.loc[low_density_cell_types] \
     * np.random.gamma(mu_low_density * mean_var_ratio,
                       1/mean_var_ratio,
                       size=(len(low_density_cell_types), 1))
    # high density cell types
    cell_types2zones.loc[high_density_cell_types] = cell_types2zones.loc[high_density_cell_types] \
     * np.random.gamma(mu_high_density * mean_var_ratio,
                       1/mean_var_ratio,
                       size=(len(high_density_cell_types), 1))

    # calculate cell type abundance at each spot
    np.random.seed(seed)
    cell_abundances = np.dot(abundances_df, cell_types2zones.T) # n_spots x n_cell_types

    # add random noise
    cell_abundances = cell_abundances * np.random.lognormal(0, 0.35, size=cell_abundances.shape)

    cell_abundances_df = pd.DataFrame(cell_abundances,
                                 index=abundances_df.index,
                                 columns=cell_types2zones.index
                                )

    # make cell type abundance discrete
    # cells at each location will be sampled according to the discrete count values,
    # and the pooled expression will be downscaled to reflect imperfect capture efficiency,
     # thus stick with the simulated abundance
    cell_count_df = np.ceil(cell_abundances_df) # n_spots x n_cell_types
    cell_capture_eff_df = cell_abundances_df / cell_count_df # n_spots x n_cell_types
    cell_capture_eff_df[cell_capture_eff_df.isna()] = 0

    # add cell type abundance group to the dataframe
    cell_types2zones['is_uniform'] = 0
    cell_types2zones.loc[uniform_cell_types, 'is_uniform'] = 1
    cell_types2zones['is_high_density'] = 0
    cell_types2zones.loc[high_density_cell_types, 'is_high_density'] = 1

    # save cell_abundances matrix
    cell_abundances_df.to_csv(out_dir + "/celltype_abundances.csv")
    cell_count_df.to_csv(out_dir + "/celltype_counts.csv")
    cell_capture_eff_df.to_csv(out_dir + "/celltype_capture_eff.csv")
    cell_types2zones.to_csv(out_dir + "/celltype_zone_assignment.csv")

    ###############################################################################
    # 3. Generate spatial data
    ###############################################################################

    # set the X to be raw counts
    generation_scref.X = generation_scref.layers['counts']
    # add index to each cell
    generation_scref.obs['cell_ind'] = np.arange(generation_scref.shape[0])

    np.random.seed(seed)
    locations2cells = sample_cell_indices(generation_scref, annot_label,
                                          cell_count_df, cell_capture_eff_df)

    # generate (n_experiments * n_spots) x 2 (x and y coordinates) grids
    locations_1, _, _ = generate_grid(grid_size=grid_size) # one location
    locations = np.concatenate([locations_1 for _ in range(n_experiments)], axis=0)

    # add lateral diffusion (spatial covariance not explained by cell type abundance)
    if smooth_scale == 0: # percentage of expression profiles contributed by neighbor locations
        locations2cells_smoothed = locations2cells
        cell_abundances_smoothed_df = cell_abundances_df
    else:
        locations2cells_smoothed = smooth_by_neighbors(
            shared_prop = smooth_scale, locations2cells_matrix = locations2cells,
            coords = locations_1.reshape(-1, 2), n_experiments = n_experiments)

        # calculate abundances per spot after smoothing
        cell_abundances_smoothed = np.zeros_like(cell_abundances)
        for ict, ct in enumerate(cell_abundances_df.columns): # celltypes
            cell_ind_all = generation_scref.obs['cell_ind']
            cell_ind_all = cell_ind_all[generation_scref.obs[annot_label] == ct]
            cell_abundances_smoothed[:,ict] = \
                locations2cells_smoothed[:, cell_ind_all].sum(1).reshape(-1)
        cell_abundances_smoothed_df = pd.DataFrame(
            cell_abundances_smoothed, index=abundances_df.index, columns=cell_types2zones.index)
        cell_abundances_smoothed_df.to_csv(out_dir + "/celltype_abundances_smoothed.csv")

    # construct anndata object of spatial data
    synth_adata = create_spatial_adata(generation_scref,
                                   locations2cells_smoothed,
                                   cell_count_df,
                                   cell_abundances_smoothed_df,
                                   cell_capture_eff_df,
                                   locations,
                                   cell_types2zones)
    # calculate and store per-cell-type summary statistics
    synth_adata = cal_counts_ct(synth_adata,
                                generation_scref,
                                annot_label,
                                locations2cells_smoothed)

    # add gene-level random noises
    if gamma_noises:
        np.random.seed(seed)
        synth_adata = add_gamma_noises(synth_adata)

    # save synthetic data and paired single-cell reference
    if output_anndata:
        synth_adata.write(out_dir + "/synthetic_sp_adata.h5ad", compression = 'gzip')
        paired_scref.write(out_dir + "/paired_sc_adata.h5ad", compression = 'gzip')

    for i in range(n_experiments):
        ## only keep marker genes for each experiment
        adata = synth_adata[
            (grid_size[0] * grid_size[1] * i):(grid_size[0] * grid_size[1] * (i + 1)), marker_names]
        coords_exp = adata.obsm['X_spatial'][:,:,0]
        df_coords = pd.DataFrame(coords_exp, index = adata.obs_names, columns = ['x', 'y'])
        df_coords.to_csv(out_dir + f"/deconv_inputs/coords_exp{i}.csv")

        sio.mmwrite(out_dir + f"/deconv_inputs/syn_sp_count_markers_exp{i}.mtx", adata.X)
        # y_exp = adata.X.toarray()
        # df_c_y = pd.DataFrame(y_exp, index = adata.obs_names, columns = marker_names)
        # df_c_y.to_csv(out_dir + f"/deconv_inputs/syn_sp_count_markers_exp{i}.csv")


if __name__ == '__main__':
    main()
