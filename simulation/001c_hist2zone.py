import os
import argparse
import numpy as np
from matplotlib.image import imread

from fcmeans import FCM
import squidpy as sq

from smoother.simulation.gp import generate_grid, get_abundances_hist
from smoother.simulation.histology import cal_membership_value, extract_features_from_image

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
    prs.add_argument('-i','--img_dir',
                     type = str,
                     default = "./example_image/glioblastoma_1000x1000.png",
                     help = 'Path to histological image',
                    )
    prs.add_argument('-m','--m_fuzzy',
                     type = float,
                     default = 2,
                     help = ('m for fuzzy c-means clustring'
                            'Increase this will increase the fuzziness of patterns.'
                            'm should be an int >=1 (default 2)')
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
    IMG_DIR = args.img_dir
    OUT_DIR = args.out_dir
    eta = args.variance
    m_fuzzy = args.m_fuzzy
    seed = args.seed

    # make out directories
    os.makedirs(OUT_DIR, exist_ok=True)

    # read histological image
    layer = "image"
    img = imread(IMG_DIR)
    img = sq.im.ImageContainer(img, layer=layer)

    # crop the image to get n_locations grid
    size = (int(img.shape[0]/n_locations[0]),
        int(img.shape[1]/n_locations[1]))
    img = img.crop_corner(0, 0, size = (n_locations[0]*size[0], n_locations[1]*size[1]))

    # feature extraction
    features_df = extract_features_from_image(img, size, layer)

    # fuzzy clustering
    n_clusters = n_regional_zones
    n = len(features_df)
    m = m_fuzzy # Fuzzy parameter: select a value greater than 1 else it will be knn

    my_model = FCM(m=m, n_clusters=n_clusters)
    X = features_df.to_numpy(dtype="float32")
    my_model.fit(X)
    # outputs
    fcm_centers = my_model.centers

    # hist prior
    locations2zone = np.zeros((n,n_clusters))
    locations2zone = cal_membership_value(locations2zone, features_df, fcm_centers, m)

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

    zone_abundances_df = get_abundances_hist(
        locations = locations_1,
        x1 = x1,
        x2 = x2,
        n_regional_zones = n_clusters,
        n_ubiquitous_zones = n_ubiquitous_zones,
        hist_abundances = locations2zone,
        band_width = l,
        n_experiments = n_experiments,
        n_locations=n_locations,
        plot_abundances = False,
        eta = eta
    )

    zone_abundances_df.to_csv(OUT_DIR + "/zone_abundances.csv")

if __name__ == '__main__':
    main()
