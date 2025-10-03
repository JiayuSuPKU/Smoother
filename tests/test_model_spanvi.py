import unittest
import numpy as np
import torch
from copy import deepcopy
from anndata import AnnData
from smoother.models.reduction._spanvi import SPANVAE, SpatialANVI
from smoother.weights import SpatialWeightMatrix
from smoother.losses import SpatialLoss

from scvi.module import SCANVAE
from scvi.model import SCVI, SCANVI
from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS

class TestSPANVAE(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input = 10
        self.n_spots = 100
        self.n_latent = 5
        self.n_batch = 2
        self.n_labels = 3

        np.random.seed(42)
        self.X = np.random.rand(self.n_input, self.n_spots)
        self.labels = np.random.choice(self.n_labels, self.n_spots)
        self.batch = np.random.choice(self.n_batch, self.n_spots)

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X.T) # n_spots x n_input
        self.adata.obs['batch'] = self.batch
        self.adata.obs['labels'] = self.labels

        # Dummy tensors for loss
        self.tensors = {
            REGISTRY_KEYS.X_KEY: torch.from_numpy(self.X.T).float(),
            REGISTRY_KEYS.BATCH_KEY: torch.from_numpy(self.batch).long(),
            REGISTRY_KEYS.LABELS_KEY: torch.from_numpy(self.labels).long(),
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
        # Test with spatial loss as an additional loss term
        model = SPANVAE(
            n_input=self.n_input,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.5,
            sp_loss_as_kl=False,
            n_batch=self.n_batch,
            n_labels=self.n_labels,
            n_latent=self.n_latent,
        )

        out = model.loss(
            self.tensors,
            self.inference_outputs,
            self.generative_outputs,
        )
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))

        # Test with spatial loss as KL divergence
        model = SPANVAE(
            n_input=self.n_input,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.5,
            sp_loss_as_kl=True,
            n_batch=self.n_batch,
            n_labels=self.n_labels,
            n_latent=self.n_latent,
        )
        out = model.loss(
            self.tensors,
            self.inference_outputs,
            self.generative_outputs,
        )
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))

    def test_from_vae(self):
        vae = SCANVAE(
            n_input=self.n_input,
            n_batch=self.n_batch,
            n_labels=self.n_labels,
            n_latent=self.n_latent,
        )
        spanvae = SPANVAE.from_vae(
            vae, spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.2, sp_loss_as_kl=False
        )
        self.assertIsInstance(spanvae, SPANVAE)
        self.assertEqual(spanvae.l_sp_loss, 0.2)
        self.assertFalse(spanvae.sp_loss_as_kl)


class TestSpatialANVI(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input = 10
        self.n_spots = 100
        self.n_latent = 5
        self.n_batch = 2
        self.n_labels = 3

        np.random.seed(42)
        self.X = np.random.rand(self.n_input, self.n_spots)
        self.labels = np.random.choice(self.n_labels, self.n_spots)
        self.batch = np.random.choice(self.n_batch, self.n_spots)

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata = AnnData(self.X.T) # n_spots x n_input
        self.adata.obs['batch'] = self.batch
        self.adata.obs['labels'] = self.labels

        SpatialANVI.setup_anndata(self.adata, batch_key="batch", labels_key="labels", unlabeled_category=0)

    def test_init(self):
        model = SpatialANVI(
            st_adata=self.adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.1,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
            n_layers=1,
            dropout_rate=0.1,
        )
        self.assertIsInstance(model, SpatialANVI)
        self.assertIsInstance(model.module, SPANVAE)
        self.assertEqual(model.module.l_sp_loss, 0.1)

    def test_train(self):
        model = SpatialANVI(
            st_adata=self.adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=1.0,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
            n_layers=1,
            dropout_rate=0.1,
        )
        model.train(max_epochs=10, accelerator = "cpu")
        self.assertIsNotNone(model.history)
        self.assertIsNotNone(model.dr_logs['elapsed_time'])
        self.assertTrue(len(model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(model.dr_logs['recon_loss']) > 0)
        self.assertTrue(len(model.dr_logs['spatial_loss']) > 0)

    def test_from_rna_model(self):
        # initialize and train a SCANVI model on the same data
        adata = deepcopy(self.adata)
        SCANVI.setup_anndata(adata, batch_key="batch", labels_key="labels", unlabeled_category=0)
        rna_model = SCANVI(adata, n_latent=self.n_latent)
        rna_model.train(max_epochs=10, accelerator = "cpu")
        self.assertIsNotNone(rna_model.history)

        # then convert to SpatialANVI
        spanvi = SpatialANVI.from_rna_model(
            sc_model=rna_model,
            st_adata=adata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.1,
            sp_loss_as_kl=True
        )
        spanvi.train(max_epochs=10, accelerator = "cpu")
        self.assertIsNotNone(spanvi.history)
        self.assertIsNotNone(spanvi.dr_logs['elapsed_time'])
        self.assertTrue(len(spanvi.dr_logs['total_loss']) > 0)
        self.assertTrue(len(spanvi.dr_logs['recon_loss']) > 0)
        self.assertTrue(len(spanvi.dr_logs['spatial_loss']) > 0)

if __name__ == "__main__":
    unittest.main()