import os
import argparse
import numpy as np

from smoother.simulation.gp import generate_grid, get_abundances

def main():

	prs = argparse.ArgumentParser()
	prs.add_argument('-ne','--n_experiments',
					 type = int,
					 default = 1,
					 help = ('Number of simulated experiments wanted')
					 )
	prs.add_argument('-ns','--n_spots',
					 nargs=2,
					 type=int,  # any type/callable can be used here
					 default=[50,50],
					 help = ( 'Number of spots, needs to be 2 int n_row n_col'
							 'Example: --n_spots 50 50')
					)
	prs.add_argument('-rz','--n_regional_zones',
					 type = int,
					 required = True,
					 help = ('Number of regional zones with sparse spatial patterns')
					)
	prs.add_argument('-uz','--n_ubiquitous_zones',
					 type = int,
					 required = True,
					 help = ('Number of ubiquitous zones with uniform spatial patterns')
					)
	prs.add_argument('-o','--out_dir',
					 required = True,
					 type = str,
					 help = 'Output directory for abundance matrix',
					)
	prs.add_argument('-vr','--variance_rg',
					 type = float,
					 default = 1.5,
					 help = ('Variance required for GP (regional zones)'
							'Increase this will increase the sparsity of patterns')
					)
	prs.add_argument('-vu','--variance_ub',
					 type = float,
					 default = 0.3,
					 help = ('Variance required for GP (ubiquitous zones)'
							'Increase this will increase the sparsity of patterns')
					)
	prs.add_argument('-s','--seed',
					 default = 253286,
					 type = int,
					 help = 'Random seed'
					)


	args = prs.parse_args()

	n_experiments = args.n_experiments
	n_locations = args.n_spots
	n_regional_zones = args.n_regional_zones
	n_ubiquitous_zones = args.n_ubiquitous_zones
	OUT_DIR = args.out_dir
	eta_rg = args.variance_rg
	eta_ub = args.variance_ub
	seed = args.seed

	# make out directories
	os.makedirs(OUT_DIR, exist_ok=True)

	# sample GP parameters
	np.random.seed(seed)
	mean_var_ratio = 5
	mean_l = 10
	l = np.random.gamma(mean_l * mean_var_ratio, 1 / mean_var_ratio,
									   size=(n_regional_zones + n_ubiquitous_zones))


	# generate (n_experiments * n_spots) x 2 (x and y coordinates) grids
	locations_1, x1, x2 = generate_grid(grid_size=n_locations) # one location
	# all experiments will have the same location grid
	# locations = np.concatenate([locations_1 for _ in range(n_experiments)], axis=0)

	zone_abundances_df = get_abundances(locations=locations_1,
										x1=x1,
										x2=x2, # location coordinates for 1 experiment
										n_regional_zones=n_regional_zones,
										n_ubiquitous_zones=n_ubiquitous_zones, # zone information
										band_width=l,
										n_experiments=n_experiments,
										plot_abundances = False,
										n_locations=n_locations,
										eta_rg = eta_rg,
										eta_ub = eta_ub
										)
	zone_abundances_df.to_csv(OUT_DIR + "/zone_abundances.csv")

if __name__ == '__main__':
	main()
