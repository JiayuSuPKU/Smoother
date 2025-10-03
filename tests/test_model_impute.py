import unittest
import numpy as np
import torch
from smoother.losses import SpatialWeightMatrix, SpatialLoss
from smoother.models.impute import ImputeTorch, ImputeConvex

class TestImputeModel(unittest.TestCase):
    def setUp(self):
        # 5 spots, 3 features
        self.n_obs = 3
        self.n_missing = 2
        self.n_all = self.n_obs + self.n_missing
        self.n_feature = 3
        np.random.seed(42)
        self.y_obs = torch.rand(self.n_obs, self.n_feature)

        self.coords = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 2]
        ])
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(
            self.coords, k=2, symmetric=True, row_scale=False, verbose=False
        )
        self.spatial_loss = SpatialLoss(
            prior='icar', spatial_weights=self.swm, rho=0.9, standardize_cov=True
        )

    def test_impute_torch_basic(self):
        model = ImputeTorch(self.y_obs, self.spatial_loss, fixed_obs=True, nonneg=False, max_epochs=1000, quiet=True)
        results = model.get_results()
        self.assertEqual(results.shape, (self.n_all, self.n_feature))
        self.assertTrue(torch.is_tensor(results))
        self.assertTrue(model.impute_flag)

    def test_impute_torch_nonneg(self):
        # Make y_obs nonnegative
        y_obs = torch.abs(self.y_obs)
        model = ImputeTorch(y_obs, self.spatial_loss, fixed_obs=True, nonneg=True, max_epochs=1000, quiet=True)
        results = model.get_results()
        self.assertTrue(torch.all(results >= 0))

    def test_impute_torch_fixed_obs_false(self):
        model = ImputeTorch(self.y_obs, self.spatial_loss, fixed_obs=False, nonneg=False, max_epochs=1000, quiet=True)
        results = model.get_results()
        self.assertEqual(results.shape, (self.n_all, self.n_feature))
        self.assertTrue(model.impute_flag)

    def test_impute_torch_forward(self):
        model = ImputeTorch(self.y_obs, self.spatial_loss, fixed_obs=True, nonneg=False, max_epochs=10, quiet=True)
        new_y_obs = torch.rand(self.n_obs, self.n_feature)
        out = model.forward(new_y_obs, max_epochs=10, quiet=True)
        self.assertEqual(out.shape, (self.n_all, self.n_feature))

    def test_impute_convex_basic(self):
        try:
            import cvxpy as cp
        except ImportError:
            self.skipTest("cvxpy not installed")
        model = ImputeConvex(self.y_obs.numpy(), self.spatial_loss, fixed_obs=True, nonneg=False, quiet=True)
        results = model.get_results()
        self.assertEqual(results.shape, (self.n_all, self.n_feature))
        self.assertTrue(model.impute_flag)

    def test_impute_convex_nonneg(self):
        try:
            import cvxpy as cp
        except ImportError:
            self.skipTest("cvxpy not installed")
        y_obs = np.abs(self.y_obs.numpy())
        model = ImputeConvex(y_obs, self.spatial_loss, fixed_obs=True, nonneg=True, quiet=True)
        results = model.get_results()
        self.assertTrue(np.all(results >= 0))

    def test_impute_convex_fixed_obs_false(self):
        try:
            import cvxpy as cp
        except ImportError:
            self.skipTest("cvxpy not installed")
        model = ImputeConvex(self.y_obs.numpy(), self.spatial_loss, fixed_obs=False, nonneg=False, quiet=True)
        results = model.get_results()
        self.assertEqual(results.shape, (self.n_all, self.n_feature))
        self.assertTrue(model.impute_flag)

if __name__ == '__main__':
    unittest.main()