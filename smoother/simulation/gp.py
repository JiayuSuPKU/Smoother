"""
Simulate spatial patterns from 2D Gaussian process.
Some functions adapted from the cell2location paper:
    Kleshchevnikov, Vitalii, et al. "Cell2location maps fine-grained cell types in spatial transcriptomics."
    Nature biotechnology 40.5 (2022): 661-671.
    https://www.nature.com/articles/s41587-021-01139-4#
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


def kernel(X1, X2, band_width=1.0, eta=1.0):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        band_width: Bandwidth of the Gaussian kernel.

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return eta**2 * np.exp(-0.5 / band_width**2 * sqdist)

def generate_grid(grid_size=[50, 50]):
    '''
    Generating a grid containing n_spots
    for each spot, return the x1 and x2 pos
    '''
    n1, n2 = grid_size
    x1 = np.linspace(0, 100, n1)[:,None]
    x2 = np.linspace(0, 100, n2)[:,None]

    # make cartesian grid out of each dimension x1 and x2
    grid = np.array(list(itertools.product(x1, x2)))
    return (grid, x1, x2)

def random_gp(coords, x1 = 1, x2 = 1, #coordinates
              n_patterns = 3, # number of zones to simulate
              eta_true = 5, #variance, defines overlapping
              l1_true = [8, 10, 15], #bw parameter
              l2_true = [8, 10, 15]
             ):
    '''
    Generating random 2D gaussian based on an isotropic squared exponential kernel.

    Args:
        coords (2D array): spatial coordinates for each spot (n_spots x 2).
        x1 (1D array): x-coordinates of the grid (x_size x 1).
        x2 (1D array): y-coordinates of the grid (y_size x 1).
    '''
    assert len(l1_true) == len(l2_true) == n_patterns

    # generate spatial covariance matrix
    #cov1, cov2 = kernel(x1, x1, l1_true), kernel(x2, x2, l2_true)
    sp_cov_list = [np.kron(kernel(x1, x1, band_width=l1_true[i], eta=eta_true),
                           kernel(x2, x2, band_width=l2_true[i], eta=eta_true))
                   for i in range(n_patterns)]

    # sample abundances from GP
    gaus_true = np.stack(
        [np.random.multivariate_normal(np.zeros(coords.shape[0]), # mean
                                       2 * sp_cov_list[i])  # cov
         for i in range(n_patterns)]
    ).T

    # softmax transformation
    props_true = (np.exp(gaus_true).T / np.exp(gaus_true).sum(axis=1)).T
    return props_true


def plot_spatial(values, grid_size=[50,50], nrows=5, names=['cell type'],
                 vmin=0, vmax=1, cmap="magma"):
    '''
    Plot the input spatial value in 2D
    '''
    if len(values.shape) == 1:
        values = np.array(values).reshape(-1,1)
    n_cell_types = values.shape[1]
    n1, n2 = grid_size
    ncols = np.ceil((n_cell_types+1) / nrows).astype('int')
    for ct in range(n_cell_types):
        plt.subplot(nrows, ncols, ct+1)
        plt.imshow(values[:,ct].reshape(n1,n2).T,
                   cmap=plt.cm.get_cmap(cmap),
                   vmin=vmin, vmax=vmax
                  )
        plt.colorbar()
        if len(names) > 1:
            plt.title(names[ct])
        else:
            plt.title(f'{names[0]} {ct+1}')

    plt.subplot(nrows, ncols, n_cell_types+1)
    plt.imshow(values.sum(axis=1).reshape(n1,n2).T,
               cmap=plt.cm.get_cmap('Greys'))
    plt.colorbar()
    plt.title('total')


def sample_gp(locations, x1, x2,
              n_zones, # number of zones
              band_width,
              eta = 1.5):
    '''
    Sample abundances with 2D GP
    '''

    assert len(band_width) == n_zones
    if n_zones == 0:
        return np.empty([locations.shape[0],0])

    sparse_abundances = random_gp(coords=locations, x1=x1, x2=x2, n_patterns=n_zones,
                                  eta_true=eta,l1_true=band_width,l2_true=band_width)

    sparse_abundances = sparse_abundances / sparse_abundances.max(0)
    sparse_abundances[sparse_abundances < 0.1] = 0

    return sparse_abundances


def conc_cir(x1,x2, mu_r=0, sd=0.1):
    '''
    Sample concentric circlar patterns
    '''
    X1 = 2*((x1-np.mean(x1))/np.max(x1))**2
    X2 = 2*((x2-np.mean(x2))/np.max(x2))**2

    d = np.stack([X1 for i in range(len(x2))]) \
    + np.stack([X2 for i in range(len(x1))], axis = 1)

    return norm(mu_r, sd).pdf(np.sqrt(d)).reshape(-1,1)

def sample_circ(locations, x1, x2,
                n_circ, # number of circular (regional) zones
                r_circ, # list of means of radius
                sd_circ, # list of sds of radius
                band_width,
                eta = 0.5):
    '''
    Sample abundances with GP based on a prior concentric circlar pattern
    '''
    ## regional regions
    assert len(r_circ) == n_circ
    assert len(sd_circ) == n_circ
    assert len(band_width) == n_circ

    # looping through all circular zones
    cir_abundances = np.stack([conc_cir(x1, x2, r_circ[i], sd_circ[i])
                               for i in range(n_circ)], axis = 1)


    gp_abundances = random_gp(coords=locations, x1=x1, x2=x2, n_patterns=n_circ,
                                                    eta_true=eta,
                                                    l1_true=band_width,
                                                    l2_true=band_width)

    sparse_abundances = cir_abundances.reshape(cir_abundances.shape[0:2]) * gp_abundances

    sparse_abundances = sparse_abundances / sparse_abundances.max(0)
    sparse_abundances[sparse_abundances < 0.1] = 0

    return sparse_abundances

def sample_hist(locations, x1, x2,
                n_hist, # number of histology clusters (zones)
                hist_abundances, # membership matrix
                band_width, eta # GP parameters
                ):
    '''
    Sample abundances with GP based on a prior histological pattern
    '''
    gp_abundances = random_gp(coords=locations, x1=x1, x2=x2, n_patterns=n_hist,
                                                    eta_true=eta,
                                                    l1_true=band_width,
                                                    l2_true=band_width)

    sparse_abundances = hist_abundances * gp_abundances

    sparse_abundances = sparse_abundances / sparse_abundances.max(0)
    sparse_abundances[sparse_abundances < 0.1] = 0

    return sparse_abundances

def get_abundances(locations, x1, x2, # location coordinates for 1 experiment
                   n_regional_zones, n_ubiquitous_zones, # zone information
                   band_width, # GP parameters
                   n_experiments, n_locations,
                   plot_abundances = True,
                   eta_rg = 1.5, # GP parameters
                   eta_ub = 0.3
                   ):
    '''
    Get the abundances matrix as presented in the cell2location notebook

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
    '''
    abundances_df = pd.DataFrame()
    for exp in tqdm(range(n_experiments)):
        # abundance df (n_locations, n_zones)
        sparse_abundances = sample_gp(locations, x1, x2,
                    n_regional_zones, # number of circular (regional) zones
                    band_width[0:n_regional_zones],
                    eta = eta_rg)

        uniform_abundances = sample_gp(locations, x1, x2, n_ubiquitous_zones,
                                       band_width[n_regional_zones:(n_regional_zones+n_ubiquitous_zones)], eta = eta_ub)

        abundances_df_1 = np.concatenate([sparse_abundances, uniform_abundances], axis=1)
        abundances_df_1 = pd.DataFrame(abundances_df_1,
                                       index=[f'exper{exp}_location_{i}' for i in range(abundances_df_1.shape[0])],
                                       columns=[f'tissue_zone_{i}' for i in range(n_regional_zones)] \
                                       + [f'uniform_{i}' for i in range(n_ubiquitous_zones)])

        abundances_df = pd.concat((abundances_df, abundances_df_1), axis=0)

        if (exp == 0) & (plot_abundances) :
            plt.figure(figsize=(3*5+5,3*5+5))
            plot_spatial(abundances_df_1.values, grid_size=n_locations, nrows=5, names=abundances_df.columns)
            plt.show()
    return abundances_df

def get_abundances_circ(locations, x1, x2, # location coordinates for 1 experiment
                        n_regional_zones, n_ubiquitous_zones, # zone information
                        r_circ, sd_circ, band_width, # GP parameters
                        n_experiments, n_locations,
                        plot_abundances = True,
                        eta = 0.3 # GP parameters
                        ):
    '''
    Get the abundances matrix using donut patterns

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
    '''
    abundances_df = pd.DataFrame()
    for exp in tqdm(range(n_experiments)):
        # abundance df (n_locations, n_zones)
        sparse_abundances = sample_circ(locations, x1, x2,
                    n_regional_zones, # number of circular (regional) zones
                    r_circ, # list of means of radius
                    sd_circ, # list of sds of radius
                    band_width[0:n_regional_zones],
                    eta = eta)

        uniform_abundances = sample_gp(locations, x1, x2, n_ubiquitous_zones,
                                       band_width[n_regional_zones:(n_regional_zones+n_ubiquitous_zones)], eta = eta)

        abundances_df_1 = np.concatenate([sparse_abundances, uniform_abundances], axis=1)
        abundances_df_1 = pd.DataFrame(abundances_df_1,
                                       index=[f'exper{exp}_location_{i}' for i in range(abundances_df_1.shape[0])],
                                       columns=[f'tissue_zone_{i}' for i in range(n_regional_zones)] \
                                       + [f'uniform_{i}' for i in range(n_ubiquitous_zones)])

        abundances_df = pd.concat((abundances_df, abundances_df_1), axis=0)

        if (exp == 0) & (plot_abundances) :
            plt.figure(figsize=(3*5+5,3*5+5))
            plot_spatial(abundances_df_1.values, grid_size=n_locations, nrows=5, names=abundances_df.columns)
            plt.show()
    return abundances_df

def get_abundances_hist(locations, x1, x2, # location coordinates for 1 experiment
                        n_regional_zones, n_ubiquitous_zones, # zone information
                        hist_abundances,
                        band_width, # GP parameters
                        n_experiments, n_locations,
                        plot_abundances = True,
                        eta = 0.3 # GP parameters
                        ):
    '''
    Get the abundances matrix using histological patterns

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
    '''
    abundances_df = pd.DataFrame()
    for exp in tqdm(range(n_experiments)):
        # abundance df (n_locations, n_zones)
        sparse_abundances = sample_hist(locations, x1, x2,
                                        n_regional_zones, # number of histology clusters (zones)
                                        hist_abundances, # membership matrix
                                        band_width[0:n_regional_zones],eta # GP parameters
                                       )

        uniform_abundances = sample_gp(locations, x1, x2, n_ubiquitous_zones,
                                       band_width[n_regional_zones:(n_regional_zones+n_ubiquitous_zones)], eta = eta)

        abundances_df_1 = np.concatenate([sparse_abundances, uniform_abundances], axis=1)
        abundances_df_1 = pd.DataFrame(abundances_df_1,
                                       index=[f'exper{exp}_location_{i}' for i in range(abundances_df_1.shape[0])],
                                       columns=[f'tissue_zone_{i}' for i in range(n_regional_zones)] \
                                       + [f'uniform_{i}' for i in range(n_ubiquitous_zones)])

        abundances_df = pd.concat((abundances_df, abundances_df_1), axis=0)

        if (exp == 0) & (plot_abundances) :
            plt.figure(figsize=(3*5+5,3*5+5))
            plot_spatial(abundances_df_1.values, grid_size=n_locations, nrows=5, names=abundances_df.columns)
            plt.show()
    return abundances_df
