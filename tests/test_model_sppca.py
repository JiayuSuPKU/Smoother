import unittest
import numpy as np
import torch
from anndata import AnnData
from smoother.models.reduction._sppca import SpatialPCA
from smoother.models.reduction._spae import SpatialAutoEncoder
from smoother.losses import SpatialLoss
from smoother.weights import SpatialWeightMatrix

class TestSpatialPCA(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_features = 10
        self.n_spots = 100
        np.random.seed(42)
        self.X = np.random.rand(self.n_features, self.n_spots)
        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X.T) # n_spots x n_features
        U, S, Vh = np.linalg.svd(self.X, full_matrices=False)
        self.adata.varm['PCs'] = U[:, :5]  # first 5 PC projections
        self.adata.obsm['X_pca'] = self.X.T @ self.adata.varm['PCs']  # n_spots x 5

    def test_init(self):
        model = SpatialPCA(self.adata, n_latent=5, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        self.assertEqual(model.n_latent, 5)
        self.assertEqual(model.n_feature, 10)
        self.assertEqual(model.U.shape, (10, 5))

    def test_forward(self):
        model = SpatialPCA(self.adata, n_latent=5, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        x = torch.rand(self.n_features, self.n_spots)  # n_feature x n_sample
        out = model.forward(x)
        self.assertEqual(out.shape, (5, 100))

    def test_get_latent_representation(self):
        model = SpatialPCA(self.adata, n_latent=5, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        latent = model.get_latent_representation()
        self.assertEqual(latent.shape, (100, 5))

    def test_reduce(self):
        model = SpatialPCA(self.adata, n_latent=5, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        model.reduce(max_epochs=100, verbose=False, quite=True)
        self.assertTrue(len(model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(model.dr_logs['spatial_loss']) > 0)

    def test_from_rna_model(self):
        rna_model = SpatialPCA(self.adata, n_latent=5)
        pca_sp = SpatialPCA.from_rna_model(rna_model, self.adata, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        self.assertEqual(pca_sp.U.shape, (10, 5))

    def test_from_scanpy(self):
        pca_sp = SpatialPCA.from_scanpy(self.adata, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5)
        self.assertEqual(pca_sp.U.shape, (10, 5))

class TestSpatialAutoEncoder(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_features = 10
        self.n_spots = 100
        np.random.seed(42)
        self.X = np.random.rand(self.n_features, self.n_spots)
        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X.T) # n_spots x n_features

    def test_init(self):
        model = SpatialAutoEncoder(
            self.adata, n_latent=5, spatial_loss=self.spatial_loss, 
            lambda_spatial_loss=1.0, lambda_orth_loss=1.0,
            n_layers = 1, n_hidden = 128, dropout_rate = 0.1
        )
        self.assertEqual(model.n_latent, 5)
        self.assertEqual(model.n_feature, 10)

    def test_forward(self):
        model = SpatialAutoEncoder(
            self.adata, n_latent=5, spatial_loss=self.spatial_loss, 
            lambda_spatial_loss=1.0, lambda_orth_loss=1.0
        )
        recon_loss = model.get_recon_loss()
        orth_loss = model.get_orth_loss()
        sp_loss = model.get_sp_loss()
        self.assertTrue(recon_loss >= 0)
        self.assertTrue(orth_loss >= 0)
        self.assertTrue(sp_loss >= 0)

        x = torch.rand(self.n_features, self.n_spots)  # n_feature x n_sample
        z = model.encode(x)
        recon = model.forward(x)
        self.assertEqual(z.shape, (100, 5))
        self.assertEqual(recon.shape, (10, 100))
        
        z = model.get_latent_representation()
        self.assertEqual(z.shape, (100, 5))

    def test_reduce(self):
        model = SpatialAutoEncoder(
            self.adata, n_latent=5, spatial_loss=self.spatial_loss, 
            lambda_spatial_loss=1.0, lambda_orth_loss=1.0
        )
        model.reduce(max_epochs=1000, verbose=False, quite=True)
        self.assertTrue(len(model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(model.dr_logs['spatial_loss']) > 0)

if __name__ == '__main__':
    unittest.main()