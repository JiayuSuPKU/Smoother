from typing import List, Literal, Optional, Union
from timeit import default_timer as timer
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from anndata import AnnData
from smoother import SpatialLoss

class SpatialPCA(nn.Module):
    """Solving PCA using stochastic gradient descent."""
    def __init__(
        self,
        adata: AnnData,
        layer: Optional[str] = None,
        n_latent: int = 10,
        spatial_loss :Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1
    ) -> None:
        # initialize the PCA model.
        super().__init__()

        self.n_latent = n_latent # num_pc
        self.n_feature = adata.shape[1] # num_feature to project

        # store data to project
        self._data = None
        self.load_adata(adata, layer)

        # store spatial loss
        self.spatial_loss = spatial_loss
        self.l_sp_loss = lambda_spatial_loss

        # initialize PC basis matrix (num_feature x num_pc)
        U_init = self._gram_schmidt(nn.init.uniform_(torch.zeros(self.n_feature, self.n_latent)))
        self.U = nn.parameter.Parameter(U_init, requires_grad = True)

        # configs and logs for dimension reduction
        self.dr_configs = {}
        self.dr_logs = {'elapsed_time': None, 'total_loss': [],
                     'recon_loss': [], 'spatial_loss': []}

    def load_adata(self, adata: AnnData, layer: Optional[str] = None):
        """Load data to project.

        Args:
            adata (AnnData): data to project.
            layer (str): layer to project. If None, use adata.X.
        """
        assert adata.shape[1] == self.n_feature, "The new adata has different number of features!"

        if layer is None:
            data = adata.X.T # n_feature x n_sample
        else:
            data = adata.layers[layer].T # n_feature x n_sample

        if not isinstance(data, np.ndarray): # convert sparse matrix to dense
            data = data.toarray()

        self._data = torch.tensor(data).float() # n_feature x n_sample

    def _gram_schmidt(self, U):
        """Project the PC basis matrix to the feasible space.

        Args:
            U (2D tensor): PC basis matrix, num_feature x num_pc.
        """
        # U will be orthogonal after transformation
        return torch.linalg.qr(U, mode = 'reduced')[0]

    def forward(self, x):
        """Project x to the lower PC space.

        Args:
            x (2D tensor): data to project, num_feature x num_sample.
        """
        return self.U.T @ x # num_pc x num_sample

    def _init_with_svd(self, x):
        """Initialize model with svd solution.

        Args:
            x (2D tensor): data to project, num_feature x num_sample.
        """
        with torch.no_grad():
            _, _, V = torch.svd_lowrank(x.T, self.n_latent)
            self.U.copy_(V) # num_feature x num_pc

    def get_latent_representation(self):
        """Project loaded data to the lower PC space and return the latent representation.
        """
        return self(self._data).T.detach().numpy() # num_sample x num_pc

    def reduce(self, lr = 1.0, max_epochs = 1000, patience = 10, tol = 1e-5,
               init_with_svd = False, verbose = True, quite = False, clear_logs = True):
        """Reduce the dimension of the expression matrix.

        Args:
            lr (float): The learning rate.
            max_epochs (int): The maximum number of epochs.
            patience (int): The patience for early stopping.
            tol (float): The tolerated convergence error.
            init_with_svd (bool): Whether to initialize with analytical solution calculated
                using `torch.svd_lowrank()`.
            verbose (bool): If True, print out loss while training.
            quite (bool): If True, no output printed.
            clear_logs (bool): If True, clear logs before training.
        """
        self.dr_configs = {
            'lr':lr, 'max_epochs':max_epochs, 'patience':patience, 'tol':tol,
            'init_with_svd':init_with_svd
        }

        # start timer
        t_start = timer()

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience = int(patience / 4) + 1,
            factor = 0.1
        )

        if init_with_svd:
            self._init_with_svd(self._data)

        if clear_logs:
            self.dr_logs = {'elapsed_time': None, 'total_loss': [],
                            'recon_loss': [], 'spatial_loss': []}

        # start training
        self.train()

        # set iteration limits
        epoch, prev_loss, d_loss = 0, 100000.0, 100000.0
        max_epochs = 10000 if max_epochs == -1 else max_epochs
        patience = patience if patience > 0 else 1

        while epoch < max_epochs and d_loss >= tol or (d_loss < 0 and patience > 0):

            # set loss to 0
            recon_loss, sp_loss = torch.tensor(0), torch.tensor(0)

            optimizer.zero_grad()

            loss = 0

            # project data to hidden subspace
            hidden = self(self._data) # k x num_spot

            # calculate reconstruction error
            # here we want to maximize the variance in the subspace
            recon_loss = - torch.norm(hidden, dim = 0).pow(2).sum() / (hidden.shape[0] * hidden.shape[1])
            loss += recon_loss

            # add spatial loss on U.T @ x
            if self.spatial_loss is not None and self.l_sp_loss > 0:
                sp_loss = self.l_sp_loss * self.spatial_loss(hidden, normalize = True)
                loss += sp_loss

            # backpropagate and update weights
            loss.backward()
            optimizer.step()

            # project U to the feasible space
            with torch.no_grad():
                self.U.copy_(self._gram_schmidt(self.U))

            # check convergence
            epoch += 1
            d_loss = prev_loss - loss.detach().item()
            if d_loss < 0: # if loss increases
                patience -= 1

            prev_loss = loss.detach().item()
            recon_loss = recon_loss.detach().item()
            sp_loss = sp_loss.detach().item()

            # update learning rate
            scheduler.step(prev_loss)

            # store losses to the log
            self.dr_logs['total_loss'].append(prev_loss)
            self.dr_logs['recon_loss'].append(recon_loss)
            self.dr_logs['spatial_loss'].append(sp_loss)

            if (verbose and not quite) and epoch % 10 == 0:
                print(f'Epoch {epoch}. Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
                      f'(spatial) {sp_loss:.4f}.')

        t_end = timer()
        self.dr_logs['elapsed_time'] = t_end - t_start

        if not quite: # print final message
            print(f'=== Time {t_end - t_start : .2f}s. Total epoch {epoch}. '
                  f'Final Loss: (total) {prev_loss:.4f}. (recon) {recon_loss:.4f}. '
                      f'(spatial) {sp_loss:.4f}.')

        if d_loss >= tol:
            warnings.warn("Fail to converge. Try to increase 'max_epochs'.")

    @classmethod
    def from_rna_model(
        cls,
        rna_model,
        st_adata: AnnData,
        layer: Optional[str] = None,
        spatial_loss :Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1
    ):
        """Initialize a spatial model from a pre-trained RNA model."""
        pca_sp = cls(
            adata = st_adata,
            layer = layer,
            n_latent = rna_model.n_latent,
            spatial_loss = spatial_loss,
            lambda_spatial_loss = lambda_spatial_loss
        )

        # copy the PCA loadings from the RNA model
        with torch.no_grad():
            pca_sp.U.copy_(rna_model.U)

        return pca_sp

    @classmethod
    def from_scanpy(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        spatial_loss :Optional[SpatialLoss] = None,
        lambda_spatial_loss = 0.1
    ):
        """Initialize a spatial model from a pre-trained RNA model."""
        assert 'PCs' in adata.varm.keys(), "Please run 'sc.tl.pca()' first."
        n_latent = adata.varm['PCs'].shape[1]
        pca_sp = cls(
            adata = adata,
            layer = layer,
            n_latent = n_latent,
            spatial_loss = spatial_loss,
            lambda_spatial_loss = lambda_spatial_loss
        )

        # copy the PCA loadings from the RNA model
        with torch.no_grad():
            pca_sp.U.copy_(torch.tensor(adata.varm['PCs']))

        return pca_sp

