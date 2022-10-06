Module smoother.simulation.gp
=============================
Simulate spatial patterns from 2D Gaussian process.
Some functions adapted from the cell2location paper:
        Kleshchevnikov, Vitalii, et al. "Cell2location maps fine-grained cell types in spatial transcriptomics."
        Nature biotechnology 40.5 (2022): 661-671.
        https://www.nature.com/articles/s41587-021-01139-4#

Functions
---------

    
`conc_cir(x1, x2, mu_r=0, sd=0.1)`
:   Sample concentric circlar patterns

    
`generate_grid(grid_size=[50, 50])`
:   Generating a grid containing n_spots
    for each spot, return the x1 and x2 pos

    
`get_abundances(locations, x1, x2, n_regional_zones, n_ubiquitous_zones, band_width, n_experiments, n_locations, plot_abundances=True, eta_rg=1.5, eta_ub=0.3)`
:   Get the abundances matrix as presented in the cell2location notebook
    
    Args:
            locations, x1, x2: # location coordinates for 1 experiment
            n_regional_zones: number of regional zones
            n_ubiquitous_zones: number of ubiquitous zones
            band_width, eta_rg, eta_ub: parameters used to calculate the 2D-gaussian kernel
            n_experiments: number of experiments required
            n_locations: number of locations[n_row, n_col] per experiment
            plot_abundances: whether to plot the abundance patterns of each zone in the first experiment
    
    Returns:
            abundances_df: (np.prod(n_locations) * n_exp) x (n_regional_zones + n_ubiquitous_zones)

    
`get_abundances_circ(locations, x1, x2, n_regional_zones, n_ubiquitous_zones, r_circ, sd_circ, band_width, n_experiments, n_locations, plot_abundances=True, eta=0.3)`
:   Get the abundances matrix using donut patterns
    
    Args:
            locations, x1, x2: # location coordinates for 1 experiment
            n_regional_zones: number of regional zones
            n_ubiquitous_zones: number of ubiquitous zones
            r_circ, sd_circ: the mean and std of a gaussian distribution used to sample circular ring patterns
            band_width, eta: parameters used to calculate the 2D-gaussian kernel
            n_experiments: number of experiments required
            n_locations: number of locations[n_row, n_col] per experiment
            plot_abundances: whether to plot the abundance patterns of each zone in the first experiment
    
    Returns:
            abundances_df: (np.prod(n_locations) * n_exp) x (n_regional_zones + n_ubiquitous_zones)

    
`get_abundances_hist(locations, x1, x2, n_regional_zones, n_ubiquitous_zones, hist_abundances, band_width, n_experiments, n_locations, plot_abundances=True, eta=0.3)`
:   Get the abundances matrix using histological patterns
    
    Args:
            locations, x1, x2: # location coordinates for 1 experiment
            n_regional_zones: number of regional zones
            n_ubiquitous_zones: number of ubiquitous zones
            hist_abundances: membership matrix from fuzzy-c-means clustering,
                    np.prod(n_locations) x n_regional_zones
            band_width, eta: parameters used to calculate the 2D-gaussian kernel
            n_experiments: number of experiments required
            n_locations: number of locations[n_row, n_col] per experiment
            plot_abundances: whether to plot the abundance patterns of each zone in the first experiment
    
    Returns:
            abundances_df: (np.prod(n_locations) * n_exp) x (n_regional_zones + n_ubiquitous_zones)

    
`kernel(X1, X2, band_width=1.0, eta=1.0)`
:   Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.
    
    Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
            band_width: Bandwidth of the Gaussian kernel.
    
    Returns:
            Covariance matrix (m x n).

    
`plot_spatial(values, grid_size=[50, 50], nrows=5, names=['cell type'], vmin=0, vmax=1, cmap='magma')`
:   Plot the input spatial value in 2D

    
`random_gp(coords, x1=1, x2=1, n_patterns=3, eta_true=5, l1_true=[8, 10, 15], l2_true=[8, 10, 15])`
:   Generating random 2D gaussian based on an isotropic squared exponential kernel.
    
    Args:
            coords (2D array): spatial coordinates for each spot (n_spots x 2).
            x1 (1D array): x-coordinates of the grid (x_size x 1).
            x2 (1D array): y-coordinates of the grid (y_size x 1).

    
`sample_circ(locations, x1, x2, n_circ, r_circ, sd_circ, band_width, eta=0.5)`
:   Sample abundances with GP based on a prior concentric circlar pattern

    
`sample_gp(locations, x1, x2, n_zones, band_width, eta=1.5)`
:   Sample abundances with 2D GP

    
`sample_hist(locations, x1, x2, n_hist, hist_abundances, band_width, eta)`
:   Sample abundances with GP based on a prior histological pattern