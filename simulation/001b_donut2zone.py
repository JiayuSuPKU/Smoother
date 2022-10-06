import os
import argparse
import numpy as np

from smoother.simulation.gp import generate_grid, get_abundances_circ

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
					 default = 0,
					 help = ('Number of ubiquitous zones with uniform spatial patterns')
					)
	prs.add_argument('-o','--out_dir',
					 required = True,
					 type = str,
					 help = 'Output directory for abundance matrix',
					)
	prs.add_argument('-w','--width',
					 type = float,
					 default = 0.1,
					 help = ('Width of each concentric circular patterns'
							'Increase this will increase the overlap between each zone.')
					)
	prs.add_argument('-v','--variance',
					 type = float,
					 default = 0.3,
					 help = ('Variance required for GP (all zones)'
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
	eta = args.variance
	_mean_sd = args.width
	seed = args.seed

	# make out directories
	os.makedirs(OUT_DIR, exist_ok=True)

	# sample GP parameters
	np.random.seed(seed)
	mean_var_ratio = 5
	mean_l = 10
	l = np.random.gamma(mean_l * mean_var_ratio, 1 / mean_var_ratio,
									   size=(n_regional_zones + n_ubiquitous_zones))

	# Sample means and sds for concentric radii from a gamma distribution

	#  mean_r = np.arange(1,n_regional_zones+1)
	# r_circ = np.random.gamma(mean_r * mean_var_ratio, 1 / mean_var_ratio)

	r_circ =  np.arange(1,n_regional_zones+1)
	r_circ = (r_circ-r_circ.min())/(r_circ.max()-r_circ.min()) # min max norm

	mean_sd = _mean_sd*100
	sd_circ = np.random.gamma(mean_sd * mean_var_ratio, 1 / mean_var_ratio,
									   size=n_regional_zones)/100



	# generate (n_experiments * n_spots) x 2 (x and y coordinates) grids
	locations_1, x1, x2 = generate_grid(grid_size=n_locations) # one location
	# all experiments will have the same location grid
	# locations = np.concatenate([locations_1 for _ in range(n_experiments)], axis=0)

	zone_abundances_df = get_abundances_circ(
		locations=locations_1,
		x1=x1,
		x2=x2,
		n_regional_zones=n_regional_zones,
		n_ubiquitous_zones=n_ubiquitous_zones,
		r_circ=r_circ,
		sd_circ=sd_circ,
		band_width=l,
		n_experiments=n_experiments,
		n_locations=n_locations,
		plot_abundances = False,
		eta = eta
	)

	zone_abundances_df.to_csv(OUT_DIR + "/zone_abundances.csv")

if __name__ == '__main__':
	main()
