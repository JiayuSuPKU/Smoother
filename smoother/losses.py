"""
Transform spatial covariance into spatial loss
"""
from typing import List
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats
from smoother.utils import get_neighbor_quantile_value_by_k

# optional imports for the topological loss
try:
    from topologylayer.nn import LevelSetLayer2D, PartialSumBarcodeLengths
except ImportError:
    warnings.warn("Package 'TopologyLayer' not installed. "
                  "The topological loss is not supported.",
                  category=ImportWarning)
    LevelSetLayer2D, PartialSumBarcodeLengths = None, None

from smoother.weights import SpatialWeightMatrix, normalize_minmax

def kl_loss_scipy(p, weights = None):
    """Calculate KL divergence using scipy.

    Args:
        p (2D array): Probability distributions, num_group x num_spot.
            Each column is one discrete distribution over num_group groups.
        weights (2D array): Spatial weight matrix, num_spot x num_spot.
            If not None, will scale pairwise KL divergence by the weight matrix.

    Returns:
        kl (float): The mean pairwise KL divergence between spots, num_spot x num_spot.
    """
    p = p + 1e-20 # for stability

    # calculate kl divergence
    kl = stats.entropy(p[:,:,None], p[:,None,:])

    # return loss
    return kl.mean() if weights is None else np.multiply(kl, weights).mean()


def kl_loss(p, weights = None):
    """Calculate kl divergence using pytorch.

    Args:
        p (2D array): Probability distributions, num_group x num_spot.
            Each column is one discrete distribution over num_group groups.
        weights (2D array): Spatial weight matrix, num_spot x num_spot.
            If not None, will scale pairwise KL divergence by the weight matrix.

    Returns:
        kl (float): The mean pairwise KL divergence between spots, num_spot x num_spot.
    """
    p = p + 1e-20 # for stability
    #p = 1.0*p / torch.sum(p, axis=0, keepdims=True)

    # calculate kl divergence
    ce = torch.matmul(p.T, torch.log(p)) # cross entropy, n x n
    kl = torch.diag(ce).reshape(-1,1) - ce # kl divergence, n x n

    # return loss
    return kl.mean() if weights is None else torch.mul(kl, weights).sum() / (weights != 0).sum()


def quadratic_loss(beta, inv_cov, group_scales, normalize = True):
    """Calculate quadratic smoothing loss using torch.

    This is the main loss that transforms spatial covariance into spatial loss. It is equivalent to
    adding a multivariate normal prior with precision matrix `inv_cov` on `beta` in a regression
    setting. See model description for more details.

    Args:
        beta (2D tensor): Columns of regression coefficients, num_group x num_spot.
        inv_cov (3D tensor): Inverse covariance (precision) matrix of spatial variables of each group,
            (num_group or 1) x n x n. If the first dimension has length 1, all groups will have the
            same covariance structure.
        group_scales (1D tensor or float): Relative prior confidence of each group. The higher the
            confidence, the stronger the smoothing will be. If float, all groups will have
            the same confidence.
        normalize (bool): If True, normalize the likelihood over the size of beta for comparability.

    Returns:
        neg_loglik (float): Sum of quadratic loss, i.e., the negative likelihood of the corresponding
            multivariate normal prior on `beta`.
    """
    if inv_cov.shape[0] == 1: # use the same spatial weight matrix for all celltypes
        neg_loglik = (torch.diag(beta @ inv_cov[0,:,:] @ beta.T) * group_scales).sum()
    else: # use different spatial weight matrices for different cell types
        neg_loglik = ((beta.unsqueeze(1) @ inv_cov @ beta.unsqueeze(2)).squeeze() * group_scales).sum()

    if normalize: # normalize the likelihood over the size of beta for better comparability
        neg_loglik /= (beta.shape[0] * beta.shape[1])

    return neg_loglik

def sparse_quadratic_loss(beta, inv_cov_2d_sp, group_scales, normalize = True):
    """Calculate quadratic smoothing loss using torch.sparse.

    This function uses the sparsity of the inverse covariance matrix to speed up the calculation.

    Args:
        beta (2D tensor): Columns of regression coefficients, num_group x num_spot.
        inv_cov_2d_sp (2D sparse tensor): Sparse block diagonal inverse covariance (precision) matrix
            of each group. There are in total (num_group or 1) num_spot x num_spot blocks.
            If 1, all groups will have the same covariance structure.
        group_scales (1D tensor or float): Relative prior confidence of each group. The higher the
            confidence, the stronger the smoothing will be. If float, all groups will have
            the same confidence.
        normalize (bool): If True, normalize the likelihood over the size of beta for comparability.

    Returns:
        neg_loglik (float): Sum of quadratic loss, i.e., the negative likelihood of the corresponding
            multivariate normal prior on `beta`.
    """
    if inv_cov_2d_sp.shape[0] == beta.shape[1]: # use the same spatial weight matrix for all celltypes
        neg_loglik = (torch.diag(beta @ torch.mm(inv_cov_2d_sp, beta.T)) * group_scales).sum()
    else: # use different spatial weight matrices for different cell types
        neg_loglik = ((beta * group_scales[:,None]).reshape(1, -1) @ \
            torch.mm(inv_cov_2d_sp, beta.reshape(-1, 1))).squeeze()

    if normalize: # normalize the likelihood over the size of beta for better comparability
        neg_loglik /= (beta.shape[0] * beta.shape[1])

    return neg_loglik


class SpatialLoss(nn.Module):
    """Spatial loss.

    The spatial smoothing loss on a spatial random variable (num_group x num_spot).

    Attributes:
        prior (str): The prior spatial process model, can be one of 'sma','sar', 'car', 'icar'.
        spatial_weights (List[SpatialWeightMatrix]): Spatial weight matrix collection of
            length num_group or 1. If 1 then all groups will be subject to the same covariance.
        rho (float): The spatial autocorrelation parameter (for SpatialWeightMatrix.get_inv_cov).
        use_sparse (bool): Whether to use sparse inverse covariance matrix in the calculation.
        standardize_cov (bool): Whether to standardize the covariance matrix to have same variance (1)
              across locations. Only proper covariance can be standardized.
        inv_cov (3D tensor): Inverse covariance (precision) matrix of spatial variables of each group,
            (num_group or 1) x n x n. If the first dimension has length 1, all groups will have the
            same covariance structure.
        inv_cov_2d_sp (2D sparse tensor): Sparse block diagonal inverse covariance (precision) matrix.
        confidences (1D tensor or float): Relative prior confidence of each group. The higher the
            confidence, the stronger the smoothing will be. If float, all groups will have
            the same confidence.
    """
    def __init__(self, prior, spatial_weights : List[SpatialWeightMatrix] = None,
                 rho = 1, use_sparse = True, standardize_cov = False) -> None:
        super(SpatialLoss, self).__init__()

        # store configs
        self.prior = prior
        self.spatial_weights = spatial_weights
        self.rho = rho
        self.use_sparse = use_sparse
        self.standardize_cov = standardize_cov
        self._sanity_check()

        # calculate inverse covariance matrix
        if self.prior != 'kl':
            if isinstance(self.spatial_weights, SpatialWeightMatrix):
                # will use the same spatial weight matrix for all celltypes
                # self.inv_cov: 1 x n_spot x n_spot
                self.inv_cov = self.spatial_weights.get_inv_cov(
                    self.prior, rho, cached=False,
                    standardize=self.standardize_cov,
                    return_sparse=use_sparse
                ).unsqueeze(0)
                self.confidences = torch.ones(1) # group-specific variance
            else: # use different spatial weight matrix per celltype
                # self.inv_cov: n_group x n_spot x n_spot
                self.inv_cov = torch.stack(
                    [swm.get_inv_cov(
                        self.prior, self.rho, cached=False,
                        standardize=self.standardize_cov,
                        return_sparse=use_sparse)
                    for swm in self.spatial_weights], dim=0)
                self.confidences = torch.ones(len(self.spatial_weights)) # group-specific variance

            # concatenate the inverse covariance matrix into a sparse diag matrix for efficient computation
            # self.inv_cov_2d_sp.shape == n_group/1 * n_spot x n_group/1 * n_spot
            if self.use_sparse:
                if self.inv_cov.shape[0] == 1:
                    self.inv_cov_2d_sp = self.inv_cov[0]
                else:
                    num_group = self.inv_cov.shape[0]
                    num_spot = self.inv_cov.shape[1]
                    indices = torch.concat(
                        [self.inv_cov[i].coalesce().indices() + i * num_spot for i in range(num_group)],
                        dim=1
                    )
                    values = torch.concat(
                        [self.inv_cov[i].coalesce().values() for i in range(num_group)],
                        dim=0
                    )
                    self.inv_cov_2d_sp = torch.sparse_coo_tensor(
                        indices, values, (num_group * num_spot, num_group * num_spot)
                    ).coalesce()
        else: # prior == 'kl'
            if (self.spatial_weights is not None) and (not isinstance(self.spatial_weights, SpatialWeightMatrix)):
                # use only the first spatial weight matrix if multiple are provided
                self.spatial_weights = self.spatial_weights[0]

    def _sanity_check(self) -> None:
        """Check whether the spatial loss is defined properly."""
        # check spatial prior model
        valid_priors = ['kl', 'sma', 'sar', 'isar', 'car', 'icar']
        if self.prior not in valid_priors:
            raise NotImplementedError(f"Spatial prior currently must be one of {valid_priors}")

        # check spatial weight matrix
        if self.spatial_weights is None and self.prior != 'kl':
            raise ValueError("A spatial weight matrix must be supplied "
                             "if you intend to apply spatial smoothing!")

    def estimate_confidence(self, ref_exp, st_exp, method = 'lr') -> None:
        """Estimate the relative confidence for each group.

        The covariance matrix will be scaled accordingly.

        Args:
            ref_exp (2D tensor): Bulk expression signiture matrix, num_gene x num_group.
            st_exp (2D tensor): Spatial expression matrix, num_gene x num_spot.
            method (str): Method used to estimate variance.
        """
        if method != 'lr':
            raise NotImplementedError

        # calculate linear regression solution
        solution = torch.linalg.lstsq(ref_exp, st_exp)[0] # num_group x num_spot

        # calculate variances, which indicate the relative confidence
        # the higher the variance, the stronger the smoothing will be
        self.confidences = torch.std(solution, dim = 1) / torch.std(solution, dim=1).max()


    def forward(self, coefs, normalize = True):
        """Calculate spatial loss.

        Args:
            coefs (2D tensor): Columns of regression coefficients, num_group x num_spot.
        """
        if self.prior == 'kl':
            # calculate weighted mean of spot-spot pairwise KL divergence
            p = torch.nn.functional.softmax(coefs, dim=0) # num_group x num_spot
            weights = self.spatial_weights # None or SpatialWeightMatrix

            if isinstance(weights, SpatialWeightMatrix):
                if self.use_sparse: # convert sparse matrix to dense
                    weights = weights.swm_scaled.to_dense()
                else:
                    weights = weights.swm_scaled

            return kl_loss(p = p, weights = weights)

        if self.prior != 'sma' and self.use_sparse: # use sparse matrix
            return sparse_quadratic_loss(
                beta = coefs, inv_cov_2d_sp=self.inv_cov_2d_sp,
                group_scales=self.confidences, normalize=normalize
            )

        return quadratic_loss(
            beta = coefs, inv_cov=self.inv_cov,
            group_scales=self.confidences, normalize=normalize
        )


    def calc_corr_decay_stats(self, coords, min_k = 0, max_k = 50, cov_ind = 0, return_var = False):
        """Calculate spatial covariance decay over degree of neighborhoods.

        Covariance measured between k-nearest neighbors. If the number of covariance
        matrices (i.e. self.inv_cov.shape[0]) is larger than 1, use 'cov_ind' to select
        the covariance matrix to use.

        Args:
            coords (2D tensor): Coordinates of spots, num_spot x 2.
            min_k: Minimum number of k in k-nearest neighbors. k = 0: self.
            max_k: Maximum number of k in k-nearest neighbors.
            cov_ind (int): Index of the covariance matrix to use.
            return_var (bool): Whether to return variance stats.

        Returns:
            corr_decay_quantiles_df (pd.DataFrame): Correlation decay quantiles.
            var_quantiles_df (pd.DataFrame): Per-spot variance quantiles.
        """
        # each element in self.inv_cov specifies a different covariance matrix
        # if self.inv_cov.shape[0] > 1, use cov_id to select the covariance matrix
        assert cov_ind < self.inv_cov.shape[0] # 1 or n_group

        num_spot = coords.shape[0]

        # calculate covariance according to the spatial loss setting
        if self.prior == 'kl':
            raise ValueError("Covariance is not supported for KL prior.")
        elif self.prior == 'sma':
            # calculate covariance matrix directly to avoid numeric error
            if isinstance(self.spatial_weights, SpatialWeightMatrix):
                weights = self.spatial_weights.swm_scaled
            else:
                weights = self.spatial_weights[cov_ind].swm_scaled

            # covariance
            cov = torch.sparse_coo_tensor(
                torch.arange(num_spot).repeat(2,1),
                torch.ones(num_spot),
                weights.shape, dtype=torch.float32
            ) + self.rho * (weights + weights.transpose(0,1)) + \
                (self.rho ** 2) * torch.matmul(weights, weights.transpose(0,1))
            cov = cov.to_dense()

        else:
            if num_spot > 10000:
                warnings.warn(
                    f"Caution: The covariance of {num_spot} spots requires large dense matrix inversion."
                )

            try:
                cov = torch.cholesky_inverse(torch.linalg.cholesky(self.inv_cov[cov_ind].to_dense()))
            except RuntimeError as exc:
                raise RuntimeError(f"The current loss ({self.prior}, rho={self.rho}) "
                                    "contains an improper spatial covariance structure. "
                                    "Please use a different spatial prior or scale "
                                    "if you intend to calculate covariance decay.") from exc

        # separate correlation and variance
        var = torch.diagonal(cov)
        inv_sd = torch.diagflat(var ** -0.5)
        corr = inv_sd @ cov @ inv_sd

        # calculate correlation decay
        corr_decay_quantiles_df = get_neighbor_quantile_value_by_k(corr, coords, min_k, max_k)

        if not return_var:
            return corr_decay_quantiles_df

        # calculate variance quantile distribution
        q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
        var_quantiles = torch.nanquantile(var, q).numpy().reshape(1,-1)
        var_quantiles_df = pd.DataFrame(var_quantiles, columns=['Q10', 'Q25', 'Q50', 'Q75', 'Q90'])

        return corr_decay_quantiles_df, var_quantiles_df


class TopLoss(nn.Module):
    """Topological loss.

    The topological smoothing loss on a spatial random variable (num_group x num_spot).

    Attributes:
        betti_prior (dict): A nested dictionary that specifies topological priors for each group.
            {group_id: {betti_k : expectation}}.
        coords_xy (2D array): Spatial coordinates (2D), num_spot x 2.
        pdfn (topologylayer.nn.LevelSetLayer2D): Super-level set layer.
        topfn (dict): Topological features for each group.
            {group_id: {betti_k: topologylayer.nn.PartialSumBarcodeLengths}}.
    """
    def __init__(self, betti_priors : dict, coords_xy):
        super(TopLoss, self).__init__()

        # store configs
        self.betti_priors = betti_priors
        self.coords_xy = np.array(coords_xy)
        self._num_spot = self.coords_xy.shape[0]
        self._dim_x, self._dim_y = self.coords_xy.max(0) + 1

        # no topological prior applied
        if self.betti_priors is None:
            return None

        if LevelSetLayer2D is None:
            print("Please install the package 'TopologyLayer' before "
                  "using the topological loss.")
            return None

        # initialize topological layers
        self.pdfn = LevelSetLayer2D(size=(self._dim_x, self._dim_y),  sublevel=False)
        self.topfn = {} # featurization layers for each group
        for group_id in self.betti_priors:
            self.topfn[group_id] = {
                # should always skip the first one barcode that
                # represents captured vs uncaptured regions
                betti_k : PartialSumBarcodeLengths(dim = betti_k, skip=prior)
                for betti_k, prior in betti_priors[group_id].items()
            }
        return None

    def reshape_coefs(self, beta, pad_value : float = None):
        """Reshape 2D coefficients into 3D and pad with zeros.

        The original coefficient matrix has 1 location axis (the first dimension)
        while the transformed matrix will have 2 location axes (x and y, the first two).
        Spatial arrangement specified in `self.coords_xy`.

        Args:
            beta (2D array): Columns of regression coefficients, num_group x num_spot.
            pad_value (float): Value of coefficients at unobserved regions.
        """
        # pad with the smallesy coef
        if pad_value is None:
            pad_value = beta.min()

        num_group = beta.shape[0]
        beta_3d = torch.ones((self._dim_x, self._dim_y, num_group)) * pad_value
        beta_3d[self.coords_xy[:,0], self.coords_xy[:,1]] = beta.T
        return beta_3d

    def forward(self, coefs):
        """Calculate topological loss.

        Args:
            coefs (2D tensor): Columns of regression coefficients, num_group x num_spot.
        """
        if self.betti_priors is None:
            return 0
        # reshape coefficients
        beta_3d = self.reshape_coefs(coefs)

        # calculate topological loss
        loss = 0
        for group_id in self.betti_priors:
            dgms_g = self.pdfn(beta_3d[:,:,group_id])
            for betti_k in self.topfn[group_id]:
                loss += self.topfn[group_id][betti_k](dgms_g)

        return loss / (self._num_spot ** 0.5 * len(self.betti_priors))


class ContrastiveSpatialLoss(SpatialLoss):
    """Spatial loss for contrastive learning.

    The spatial loss that maximizes similarity of a spatial random variable (num_group x num_spot)
    over true neighbors while minimizing similarity over randomly generated neighbors. See
    Zhu, Hao, Ke Sun, and Peter Koniusz. "Contrastive laplacian eigenmaps."
     Advances in Neural Information Processing Systems 34 (2021).
    https://arxiv.org/abs/2201.05493

    By default, the inverse covariance matrix generated from `prior = 'icar'` and
     `rho = 1` is the graph laplacian. Set `standardize_cov = True` to normalize
    the graph laplacian.

    Attributes: See `SpatialLoss`.
        num_perm (int): Number of negative graphs (generated by randomly shuffling spots).
        neg2pos_ratio (float): Relative importance of negative samples to positive samples.
        lower_bound (float): Lower bound of the loss in case that cov is not positive semi-definite.
        check_neg_samples (bool): Whether to check if resulting cov is positive definite.
    """
    def __init__(self, prior = 'icar', spatial_weights: List[SpatialWeightMatrix] = None,
                 rho=1, use_sparse=True, standardize_cov=True,
                 num_perm = 10, neg2pos_ratio = 0.5, lower_bound = -1,
                 check_positive_definite = False) -> None:
        # recommend using the defaul settings, i.e., `prior = 'icar'`, `rho = 1`
        # and standardize_cov = True
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            super().__init__(prior, spatial_weights, rho, use_sparse, standardize_cov)

        self.num_perm = num_perm
        self.neg2pos_ratio = neg2pos_ratio
        self.lower_bound = lower_bound

        if check_positive_definite:
            self.check_positive_definite()

    def check_positive_definite(self, n_tests = 10):
        """Check if the covariance matrix is positive semi-definite."""
        for _ in range(n_tests):
            inv_cov_perm = torch.zeros_like(self.inv_cov)
            for _ in range(self.num_perm): # randomly permute spots
                idx_perm = torch.randperm(self.inv_cov.shape[-1])
                inv_cov_perm += self.inv_cov[:,:,idx_perm][:,idx_perm,:]
            inv_cov = self.inv_cov - self.neg2pos_ratio / self.num_perm * inv_cov_perm

            try: # check if the covariance matrix is positive definite
                torch.linalg.cholesky(inv_cov)
            except RuntimeError:
                eigvals = np.sort(torch.linalg.eigvals(inv_cov))
                if not (eigvals[0].real > -1e-7).all():
                    warnings.warn(
                        "The constrastive covariance is not positive semi-definite. "
                        f"you may want to use a smaller neg2pos_ratio than {self.neg2pos_ratio}."
                    )
                    break

    def forward(self, coefs, normalize = True):
        """Calculate contrastive spatial loss."""
        # positive loss to maximize similarity with true neighbors
        pos_loss = super().forward(coefs, normalize = normalize)

        # negative loss to minimize similarity with randomly generated neighbors
        neg_loss = 0
        for _ in range(self.num_perm): # randomly permute spots
            idx_perm = torch.randperm(coefs.shape[1])
            neg_loss += super().forward(coefs[:, idx_perm], normalize = normalize)
        neg_loss /= self.num_perm

        # total contrastive loss clamped to lower bound
        total_loss = pos_loss - self.neg2pos_ratio * neg_loss

        # clamp to lower bound
        if not normalize:
            lower_bound = self.lower_bound * coefs.shape[0] * coefs.shape[1]
        else:
            lower_bound = self.lower_bound

        return torch.clamp(total_loss, min=lower_bound)
