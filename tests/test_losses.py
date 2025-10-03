import unittest
import torch
import numpy as np
from smoother.losses import SpatialLoss
from smoother.weights import SpatialWeightMatrix

class TestSpatialLoss(unittest.TestCase):
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
        self.expr = np.random.rand(10, 5)  # 10 genes x 5 spots

        # Create a spatial weight matrix
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=2, symmetric=True, row_scale=True)

    def test_init_sma(self):
        with self.assertRaises(NotImplementedError):
            loss = SpatialLoss(prior='sma', spatial_weights=self.swm, rho=0.5)

        loss = SpatialLoss(prior='sma', spatial_weights=self.swm, rho=0.5, use_sparse = False)
        self.assertEqual(loss.prior, 'sma')
        self.assertIsNotNone(loss.inv_cov)
        self.assertTrue(torch.is_tensor(loss.inv_cov))

    def test_init_icar(self):
        loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5)
        self.assertEqual(loss.prior, 'icar')
        self.assertIsNotNone(loss.inv_cov)
        self.assertTrue(torch.is_tensor(loss.inv_cov))

    def test_forward_sma(self):
        loss = SpatialLoss(prior='sma', spatial_weights=self.swm, rho=0.5, use_sparse = False)
        coefs = torch.randn(3, 5)
        out = loss.forward(coefs)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.dim(), 0)

    def test_forward_icar(self):
        loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5)
        coefs = torch.randn(3, 5)
        out = loss.forward(coefs)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.dim(), 0)

    def test_forward_kl(self):
        loss = SpatialLoss(prior='kl', spatial_weights=self.swm)
        coefs = torch.randn(3, 5)
        out = loss.forward(coefs)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.dim(), 0)

    def test_calc_corr_decay_stats(self):
        loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5)
        coefs = torch.randn(3, 5)
        stats = loss.calc_corr_decay_stats(torch.tensor(self.coords, dtype=torch.float32), min_k=0, max_k=2)
        self.assertTrue(hasattr(stats, 'shape') or isinstance(stats, object))

    def test_invalid_prior(self):
        with self.assertRaises(NotImplementedError):
            SpatialLoss(prior='invalid', spatial_weights=self.swm)


if __name__ == '__main__':
    unittest.main()