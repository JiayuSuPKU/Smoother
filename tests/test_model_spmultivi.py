import unittest
from copy import deepcopy
import torch
import numpy as np
from anndata import AnnData
from mudata import MuData
from smoother.models.reduction._spmultivi import SPMULTIVAE, SpatialMULTIVI
from smoother.weights import SpatialWeightMatrix
from smoother.losses import SpatialLoss

from scvi import REGISTRY_KEYS
from scvi.module import MULTIVAE
from scvi.model import MULTIVI

class TestSPMULTIVAE(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input_rna = 10
        self.n_input_atac = 20
        self.n_spots = 100
        self.n_latent = 5
        np.random.seed(42)

        self.X_rna = np.abs(np.random.rand(self.n_input_rna, self.n_spots))
        self.X_atac = np.random.rand(self.n_input_atac, self.n_spots)
        self.X_atac[self.X_atac < 0.7] = 0.0  # make it sparse

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata_rna = AnnData(self.X_rna.T) # n_spots x n_input_rna
        self.adata_rna.var_names = [f'gene_{i}' for i in range(self.n_input_rna)]
        self.adata_atac = AnnData(self.X_atac.T) # n_spots x n_input_atac
        self.adata_atac.var_names = [f'peak_{i}' for i in range(self.n_input_atac)]
        self.mudata = MuData({"rna": self.adata_rna, "atac": self.adata_atac})

        # Dummy tensors for loss
        self.tensors = {
            REGISTRY_KEYS.X_KEY: torch.randn(self.n_spots, self.n_input_rna + self.n_input_atac),
            REGISTRY_KEYS.BATCH_KEY: torch.zeros(self.n_spots, dtype=torch.long),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros(self.n_spots, dtype=torch.long),
        }
        self.inference_outputs = {
            'x': torch.randn(self.n_spots, self.n_input_rna + self.n_input_atac),
            'qz_m': torch.randn(self.n_spots, self.n_latent),
            'qz_v': torch.ones(self.n_spots, self.n_latent),
            'ql': torch.distributions.Normal(torch.zeros(self.n_spots, 1), torch.ones(self.n_spots, 1)),
            'z': torch.randn(self.n_spots, self.n_latent),
            'qzm_expr': torch.randn(self.n_spots, self.n_latent),
            'qzv_expr': torch.ones(self.n_spots, self.n_latent),
            'qzm_acc': torch.randn(self.n_spots, self.n_latent),
            'qzv_acc': torch.ones(self.n_spots, self.n_latent),
            'qzm_pro': torch.zeros(self.n_spots, self.n_latent),
            'qzv_pro': torch.ones(self.n_spots, self.n_latent),
            'libsize_acc': torch.ones(self.n_spots, 1),
        }
        self.generative_outputs = {
            'p': torch.sigmoid(torch.randn(self.n_spots, self.n_input_atac)),
            'px_rate': torch.randn(self.n_spots, self.n_input_rna),
            'px_r': torch.randn(self.n_spots, self.n_input_rna),
            'px_dropout': torch.sigmoid(torch.randn(self.n_spots, self.n_input_rna)),
        }

    def test_spatial_loss(self):
        # Test with spatial loss as an separate loss term
        self.model = SPMULTIVAE(
            n_input_regions=self.n_input_atac,
            n_input_genes=self.n_input_rna,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.2,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
        )
        out = self.model.loss(self.tensors, self.inference_outputs, self.generative_outputs)
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))

        # Test with spatial loss as KL divergence
        self.model = SPMULTIVAE(
            n_input_regions=self.n_input_atac,
            n_input_genes=self.n_input_rna,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.2,
            sp_loss_as_kl=True,
            n_latent=self.n_latent,
        )
        out = self.model.loss(self.tensors, self.inference_outputs, self.generative_outputs)
        self.assertTrue(hasattr(out, "kl_global"))
        self.assertTrue(torch.is_tensor(out.kl_global["kl_global"]))
        self.assertTrue(out.kl_global["kl_global"] == out.kl_local['kl_divergence_z'])

    def test_from_vae(self):
        mvae = MULTIVAE(n_input_regions=self.n_input_atac, n_input_genes=self.n_input_rna)
        spmvae = SPMULTIVAE.from_vae(mvae, spatial_loss=self.spatial_loss, lambda_spatial_loss=0.5, sp_loss_as_kl=True)
        self.assertIsInstance(spmvae, SPMULTIVAE)
        self.assertEqual(spmvae.l_sp_loss, 0.5)
        self.assertTrue(spmvae.sp_loss_as_kl)

class TestSpatialMULTIVI(unittest.TestCase):
    def setUp(self):
        # Small synthetic dataset
        self.n_input_rna = 10
        self.n_input_atac = 20
        self.n_spots = 100
        self.n_latent = 5
        np.random.seed(42)

        self.X_rna = np.abs(np.random.rand(self.n_input_rna, self.n_spots))
        self.X_atac = np.random.rand(self.n_input_atac, self.n_spots)
        self.X_atac[self.X_atac < 0.7] = 0.0  # make it sparse

        self.coords = np.random.rand(self.n_spots, 2)
        self.swm = SpatialWeightMatrix()
        self.swm.calc_weights_knn(self.coords, k=4, symmetric=True, row_scale=False, verbose=False)
        self.spatial_loss = SpatialLoss(prior='icar', spatial_weights=self.swm, rho=0.5, standardize_cov=True)

        self.adata_rna = AnnData(self.X_rna.T) # n_spots x n_input_rna
        self.adata_rna.var_names = [f'gene_{i}' for i in range(self.n_input_rna)]
        self.adata_atac = AnnData(self.X_atac.T) # n_spots x n_input_atac
        self.adata_atac.var_names = [f'peak_{i}' for i in range(self.n_input_atac)]
        self.mudata = MuData({"rna": self.adata_rna, "atac": self.adata_atac})

        SpatialMULTIVI.setup_mudata(self.mudata, modalities = {"rna_layer": "rna", "atac_layer": "atac"})

    def test_init(self):
        model = SpatialMULTIVI(
            st_adata=self.mudata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=0.1,
            sp_loss_as_kl=False,
            n_latent=self.n_latent,
        )
        self.assertIsInstance(model.module, SPMULTIVAE)
        self.assertEqual(model.module.l_sp_loss, 0.1)

    def test_train_separate_sp_loss(self):
        model = SpatialMULTIVI(
            st_adata=self.mudata,
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

    def test_train_kl_sp_loss(self):
        model = SpatialMULTIVI(
            st_adata=self.mudata,
            spatial_loss=self.spatial_loss,
            lambda_spatial_loss=1.0,
            sp_loss_as_kl=True,
            n_latent=self.n_latent,
        )
        model.train(max_epochs=10, lr = 0.01, accelerator = "cpu")
        self.assertIsNotNone(model.history)
        self.assertIsNotNone(model.dr_logs['elapsed_time'])
        self.assertTrue(len(model.dr_logs['total_loss']) > 0)
        self.assertTrue(len(model.dr_logs['recon_loss']) > 0)
        self.assertTrue(len(model.dr_logs['spatial_loss']) > 0)

    def test_from_sc_model(self):
        # initialize and train a MULTIVI model on the same data
        MULTIVI.setup_mudata(self.mudata, modalities = {"rna_layer": "rna", "atac_layer": "atac"})
        sc_model = MULTIVI(self.mudata, n_latent=self.n_latent)
        sc_model.train(max_epochs=10, accelerator = "cpu")
        self.assertIsNotNone(sc_model.history)

        # then convert to SpatialMULTIVI
        sp_model = SpatialMULTIVI.from_sc_model(
            sc_model=sc_model,
            st_adata=self.mudata,
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

if __name__ == '__main__':
    unittest.main()