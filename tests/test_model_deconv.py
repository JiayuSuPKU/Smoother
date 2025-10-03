import unittest
import numpy as np
from smoother.weights import SpatialWeightMatrix
from smoother.losses import SpatialLoss
from smoother.models.deconv import LinearRegression, NNLS, NuSVR, DWLS

class TestDeconvModel(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_features = 10
        self.n_groups = 3
        self.n_spots = 50
        np.random.seed(42)
        self.x = np.random.rand(self.n_features, self.n_groups)
        self.y = np.random.rand(self.n_features, self.n_spots)
        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.is_cvxpy_installed = self._test_cvxpy_installed()

    def _test_cvxpy_installed(self):
        try:
            import cvxpy
            return True
        except ImportError:
            return False

    def test_lr(self):
        model = LinearRegression(backend='pytorch', bias=True)
        model.deconv(self.x, self.y, spatial_loss=self.spatial_loss, lambda_spatial_loss=1, verbose=False, quiet=True)
        props = model.get_props().numpy()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))
        self.assertTrue(np.all(props >= 0))
        self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))
        
        if self.is_cvxpy_installed:
            model = LinearRegression(backend='cvxpy', bias=True)
            model.deconv(self.x, self.y, spatial_loss=self.spatial_loss, lambda_spatial_loss=1, verbose=False, quiet=True)
            props = model.get_props()
            self.assertEqual(props.shape, (self.n_spots, self.n_groups))
            self.assertTrue(np.all(props >= 0))
            self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))

    def test_nnls(self):
        model = NNLS(backend='pytorch', bias=True)
        model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
        props = model.get_props().numpy()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))
        self.assertTrue(np.all(props >= 0))
        self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))
        
        if self.is_cvxpy_installed:
            model = NNLS(backend='cvxpy', bias=True)
            model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
            props = model.get_props()
            self.assertEqual(props.shape, (self.n_spots, self.n_groups))
            self.assertTrue(np.all(props >= 0))
            self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))

    def test_nusvr(self):
        model = NuSVR(backend='pytorch', bias=True)
        model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
        props = model.get_props().numpy()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))
        self.assertTrue(np.all(props >= 0))
        self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))
        
        if self.is_cvxpy_installed:
            model = NuSVR(backend='cvxpy', bias=True)
            model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
            props = model.get_props()
            self.assertEqual(props.shape, (self.n_spots, self.n_groups))
            self.assertTrue(np.all(props >= 0))
            self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))
    
    def test_dwls(self):
        model = DWLS(backend='pytorch', bias=True)
        model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
        props = model.get_props().numpy()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))
        self.assertTrue(np.all(props >= 0))
        self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))
        
        if self.is_cvxpy_installed:
            model = DWLS(backend='cvxpy', bias=True)
            model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
            props = model.get_props()
            self.assertEqual(props.shape, (self.n_spots, self.n_groups))
            self.assertTrue(np.all(props >= 0))
            self.assertTrue(np.allclose(props.sum(1), np.ones(self.n_spots), atol=1e-2) or np.all(props.sum(1) <= 1.01))

    def test_cvx_warm_start(self):
        # Test warm start (lambda_spatial_loss update)
        model = LinearRegression(backend='cvxpy', bias=True)
        model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.0, verbose=False, quiet=True)
        props = model.get_props()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))

        # Should trigger warm start
        model.deconv(self.x, self.y, spatial_loss=None, lambda_spatial_loss=0.1, verbose=False, quiet=True)
        props = model.get_props()
        self.assertEqual(props.shape, (self.n_spots, self.n_groups))

if __name__ == '__main__':
    unittest.main()