import unittest
import numpy as np
import torch
from copy import deepcopy
from anndata import AnnData
from smoother.models.reduction._spvae import SPVAE, SpatialVAE
from smoother.weights import SpatialWeightMatrix
from smoother.losses import SpatialLoss
from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module import VAE
from scvi.model import SCVI

class TestSPVAE(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input = 10
        self.n_spots = 100
        self.n_latent = 5
        np.random.seed(42)
        self.X = np.random.rand(self.n_input, self.n_spots)

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X.T) # n_spots x n_input

        # Dummy tensors for loss
        self.tensors = {
            REGISTRY_KEYS.X_KEY: torch.randn(self.n_spots, self.n_input),
            REGISTRY_KEYS.BATCH_KEY: torch.zeros(self.n_spots, dtype=torch.long),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros(self.n_spots, dtype=torch.long),
        }
        self.inference_outputs = {
            MODULE_KEYS.QZ_KEY: torch.distributions.Normal(torch.zeros(self.n_spots, self.n_latent), torch.ones(self.n_spots, self.n_latent)),
            MODULE_KEYS.QZM_KEY: torch.randn(self.n_spots, self.n_latent),
            MODULE_KEYS.QZV_KEY: torch.ones(self.n_spots, self.n_latent),
            MODULE_KEYS.QL_KEY: torch.distributions.Normal(torch.zeros(self.n_spots, 1), torch.ones(self.n_spots, 1)),
            MODULE_KEYS.Z_KEY: torch.randn(self.n_spots, self.n_latent),
        }
        self.generative_outputs = {
            MODULE_KEYS.PZ_KEY: torch.distributions.Normal(torch.zeros(self.n_spots, self.n_latent), torch.ones(self.n_spots, self.n_latent)),
            MODULE_KEYS.PL_KEY: torch.distributions.Normal(torch.zeros(self.n_spots, 1), torch.ones(self.n_spots, 1)),
            MODULE_KEYS.PX_KEY: torch.distributions.Normal(torch.zeros(self.n_spots, self.n_input), torch.ones(self.n_spots, self.n_input)),
        }

    def test_spatial_loss(self):
        # Test with spatial loss as an separate loss term
        self.model = SPVAE(
            n_input=self.n_input,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.2,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
        )
        out = self.model.loss(self.tensors, self.inference_outputs, self.generative_outputs)
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))

        # Test with spatial loss as KL divergence
        self.model = SPVAE(
            n_input=self.n_input,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.2,
            sp_loss_as_kl=True,
            n_latent=self.n_latent,
        )
        out = self.model.loss(self.tensors, self.inference_outputs, self.generative_outputs)
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))
        self.assertTrue(out.kl_global["kl_global"] == out.kl_local[MODULE_KEYS.KL_Z_KEY])

    def test_from_vae(self):
        vae = VAE(n_input=self.n_input)
        spvae = SPVAE.from_vae(vae, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5, sp_loss_as_kl=True)
        self.assertIsInstance(spvae, SPVAE)
        self.assertEqual(spvae.l_sp_loss, 0.5)
        self.assertTrue(spvae.sp_loss_as_kl)

class TestSpatialVAE(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input = 10
        self.n_spots = 100
        self.n_latent = 5
        np.random.seed(42)
        self.X = np.random.poisson(5, (self.n_spots, self.n_input))

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X) # n_spots x n_input
        self.adata.layers["counts"] = self.X

        SpatialVAE.setup_anndata(self.adata, layer="counts")


    def test_init(self):
        model = SpatialVAE(
            st_adata=self.adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.1,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
        )
        self.assertIsInstance(model.module, SPVAE)
        self.assertEqual(model.module.l_sp_loss, 0.1)

    def test_train(self):
        model = SpatialVAE(
            st_adata=self.adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=1.0,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
        )
        model.train(max_epochs=10, lr = 0.01, accelerator = "cpu")
        self.assertIsNotNone(model.history)
        self.assertIsNotNone(model.dr_logs['elapsed_time'])
        self.assertTrue(len(model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(model.dr_logs['recon_loss']) > 0)
        self.assertTrue(len(model.dr_logs['spatial_loss']) > 0)

    def test_from_rna_model(self):
        # initialize and train a scVI model on the same data
        adata = deepcopy(self.adata)
        SCVI.setup_anndata(adata, layer="counts")
        rna_model = SCVI(adata, n_latent=self.n_latent)
        rna_model.train(max_epochs=10, accelerator = "cpu")
        self.assertIsNotNone(rna_model.history)

        # then convert to SpatialVAE
        sp_model = SpatialVAE.from_rna_model(
            sc_model=rna_model,
            st_adata=self.adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=1.0,
            sp_loss_as_kl=True,
        )
        sp_model.train(max_epochs=10, lr = 0.01, accelerator = "cpu")
        self.assertIsNotNone(sp_model.history)
        self.assertIsNotNone(sp_model.dr_logs['elapsed_time'])
        self.assertTrue(len(sp_model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(sp_model.dr_logs['recon_loss']) > 0)
        self.assertTrue(len(sp_model.dr_logs['spatial_loss']) > 0)

if __name__ == "__main__":
    unittest.main()