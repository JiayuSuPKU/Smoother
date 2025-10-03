import unittest
import torch
import numpy as np
from smoother.weights import SpatialWeightMatrix

class TestSpatialWeightMatrix(unittest.TestCase):
    def setUp(self):
        # Create simple coordinates for 5 spots in 2D
        self.coords = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2]
        ])
        self.spot_ids = np.array([0, 0, 1, 1, 2])
        np.random.seed(42)
        self.expr = np.random.rand(10, 5)  # 10 genes x 5 spots
        # Dummy image and scale_factors for histology
        self.image = np.random.rand(10, 10, 3)
        self.scale_factors = {
            'spot_diameter_fullres': 5.0,
            'tissue_hires_scalef': 0.5
        }

        self.swm = SpatialWeightMatrix()

    def test_calc_weights_knn(self):
        self.swm.calc_weights_knn(self.coords, k=2, symmetric=True, row_scale=True)
        self.assertIsNotNone(self.swm.swm)
        self.assertTrue(self.swm.swm.is_sparse)
        self.assertEqual(self.swm.swm.shape[0], 5)
        self.assertTrue(torch.is_tensor(self.swm.swm.values()))

    def test_calc_weights_dist(self):
        self.swm.calc_weights_dist(self.coords, scale_coords=True, radius_cutoff=1.0, band_width=0.1)
        self.assertIsNotNone(self.swm.swm)
        self.assertTrue(self.swm.swm.is_sparse)
        self.assertEqual(self.swm.swm.shape[0], 5)
        self.assertTrue(torch.is_tensor(self.swm.swm.values()))

    def test_scale_by_similarity(self):
        self.swm.calc_weights_knn(self.coords, k=2)
        sim = torch.eye(5)
        scaled = self.swm.scale_by_similarity(sim, row_scale=True, return_swm=True)
        self.assertIsNotNone(scaled)
        self.assertEqual(scaled.shape[0], 5)

    def test_scale_by_identity(self):
        self.swm.calc_weights_knn(self.coords, k=2)
        scaled = self.swm.scale_by_identity(self.spot_ids, boundary_connectivity=0.5, row_scale=True, return_swm=True)
        self.assertIsNotNone(scaled)
        self.assertEqual(scaled.shape[0], 5)

    def test_scale_by_expr(self):
        self.swm.calc_weights_knn(self.coords, k=2)
        # Should not raise error
        self.swm.scale_by_expr(self.expr, dist_metric='cosine', reduce='pca', dim=3, row_scale=True)
        self.assertIsNotNone(self.swm.swm_scaled)

    def test_scale_by_histology(self):
        self.swm.calc_weights_knn(self.coords, k=2)
        # Should not raise error
        self.swm.scale_by_histology(self.coords, self.image, self.scale_factors, dist_metric='euclidean', reduce='pca', dim=3, row_scale=True)
        self.assertIsNotNone(self.swm.swm_scaled)

    def test_get_inv_cov(self):
        self.swm.calc_weights_knn(self.coords, k=2)
        inv_cov = self.swm.get_inv_cov(model='car', rho=0.5, cached=False, standardize=False, return_sparse=True)
        self.assertIsNotNone(inv_cov)
        self.assertTrue(inv_cov.shape[0] == 5)

if __name__ == '__main__':
    unittest.main()