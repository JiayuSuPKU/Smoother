Basic usage
============

Smoother is a two-step framework that (1) extracts prior dependency structures from positional information and (2) integrates them into non-spatial models for downstream tasks to encourage local smoothness.
Here we briefly outline key components and steps of the Smoother pipeline. 

.. code-block:: python

   # functionalities for creating spatial graphs and losses
   from smoother import SpatialWeightMatrix, SpatialLoss

   # models for imputation, deconvolution, and dimension reduction
   from smoother.models.deconv import NNLS
   from smoother.models.reduction import SpatialPCA, SpatialVAE

Spatial graph construction
----------------------------------

Our first step is to construct a weighted spatial graph using physical positions, histology, and additional features, which serves to represent spatial dependencies `a priori`. 
The neighborhood graph is represented as an :math:`n \times n` adjacency matrix :math:`W`, referred to as `SpatialWeightMatrix`, such that 
each entry :math:`W_{ij}` indicates the connectivity between sample :math:`i` and :math:`j`.

.. code-block:: python

   import numpy as np
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss

   # simulated spatial coordinates
   coords = np.random.rand(1000, 2) # n_spot x 2, spatial coordinates

   # build spatial weight matrix from spatial coordinates
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(
      coords, 
      k=4, 
      symmetric=True, # only mutual kNN edges are kept
      row_scale=False # rows do not need to sum to 1
   )

   # or alternatively, connect spots by a hard-coded distance threshold and 
   # compute edge weights using pairwise distance
   weights.calc_weights_dist(
      coords, 
      scale_coords=True, # scale coordinates to [0, 1]
      radius_cutoff = 1, # pairwise distance cutoff for edge creation, in scaled units
      band_width = 0.1 # Gaussian kernel bandwidth for edge weight computation
   ) 

By default, the resulting spatial weight matrix is stored as a sparse matrix in COO format (accessible at `weights.swm`).

.. note::

   Optionally, the weight matrix can be scaled to encode information on region boundary using additional information. 
   See :class:`smoother.SpatialWeightMatrix` for details.

This includes scaling by expression,

.. code-block:: python

   # scale weights by transcriptomics similarity
   y = torch.from_numpy(
      np.random.poisson(1, (200, 1000)) # simulated expression count matrix, n_gene x n_spot
   ).float()
   weights.scale_by_expr(
      y,
      dist_metric='cosine', # scale edge weights by pairwise cosine similarity
      row_scale=True, # rows sum to 1 after scaling 
   )

by histology images (10X Visium as an example),

.. code-block:: python

   from matplotlib.image import imread
   import json

   # load 10X visium histology image and scale factors
   img = imread('path_to_data/tissue_hires_image.png') # n_pixel x n_pixel x 3
   scale_factors = json.load('path_to_data/scalefactors_json.json')
   # scale_factors['spot_diameter_fullres']: Spot size (fullres)
	# scale_factors['tissue_hires_scalef']: Scale factor that transforms fullres image to the given image.

   # scale weights by histological similarity
   weights.scale_by_histology(
      coords, img, scale_factors,
      dist_metric='euclidean', # scale edge weights by pairwise euclidean distance
      reduce='pca', # per-spot histology features are extracted using PCA from raw pixel values
      dim=10
   )

and by external annotations on spatial regions and clusters (no interaction between regions).

.. code-block:: python

   # hard-prune the graph to remove all unwanted interactions between regions
   class_anno = np.random.choice(5, 1000) # simulated region annotations, n_spot
   weights.scale_by_identity(
      class_anno, 
      boundary_connectivity=0 # no connectivity between different regions
   )

Spatial loss construction
----------------------------------

Subsequently, Smoother translates the spatial weight matrix into a covariance structure, assuming certain underlying stochastic processes. 
The covariance is then converted into a modular sparse loss function, referred to as `SpatialLoss`, through a multivariate normal (MVN, aka Gaussian random field) prior. 
When applied to a random variable of interest, the spatial loss regularizes incoherence in the variable, therefore improving performance in downstream tasks. 

.. code-block:: python

   # import spatial losses and models
   import numpy as np
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss

   # simulated data and coordinates
   data = np.random.rand(1000, 5) # n_spot x n_gene, the spatial count matrix
   coords = np.random.rand(1000, 2) # n_spot x 2, spatial coordinates

   # build spatial weight matrix from spatial coordinates
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(coords, k=4, symmetric=True, row_scale=False) # mutual kNN graph weights

   # (optional) scale weights by transcriptomics similarity
   weights.scale_by_expr(data.T, dist_metric='cosine', reduce='none')

   # transform it into spatial loss
   sp_loss_fn = SpatialLoss(prior='icar', spatial_weights=weights, rho=0.99, use_sparse=True, standardize_cov=False)
   # by default the inverse spatial covariance matrix is stored as a sparse matrix in sp_loss_fn.inv_cov
   sp_loss_fn.inv_cov.shape # -> (1, 1000, 1000)

   # regularize any spatial random variable of interest
   variable_of_interest = torch.randn(2, 1000) # n_dim x n_spot
   loss = sp_loss_fn(variable_of_interest) # losses are summed over all dimensions, returning a scalar

.. note::
   The inverse of covariance :math:`\Sigma_{n \times n}` is stored at `sp_loss_fn.inv_cov` as a sparse COO matrix. 
   During downstream optimizations, the covariance is always fixed since it represents prior belief.
   The loss is proportional to the negative log likelihood of the prior :math:`L_{sp}(X; \Sigma) = \frac{1}{2}X^T \Sigma^{-1}X`.

Smoother offers five different yet related spatial processes: CAR (conditional autoregressive), SAR (simultaneous autoregressive), ICAR, ISAR, and 
SMA (spatial moving average). Specifically, CAR and SAR are equivalent upon transformation, and ICAR and ISAR are the weights-scaled versions so that 
the autocorrelation parameter :math:`\rho` falls in [0, 1]. By adjusting :math:`\rho`, these models can achieve parallel regularization effects. 
Based on numerical considerations, we typically recommend using ICAR with varying :math:`\rho` (or ISAR with smaller :math:`\rho`) 
to accommodate data with diverse neighborhood structures, for instance, “ICAR (:math:`\rho = 0.99`)” for data with clear anatomy and 
“ICAR (:math:`\rho = 0.9`)” for tumor data. SMA is generally not recommended since the resulting inverse covariance matrix tends to be less sparse, potentially slowing down computation.

.. code-block:: python

   sp_loss_fn = SpatialLoss('icar', weights, rho=0.99)
   sp_loss_fn = SpatialLoss('isar', weights, rho=0.9)

In addition, we implement a contrastive extension of the spatial loss :class:`smoother.losses.ContrastiveSpatialLoss` to increase the penalty for pulling distant spots too close, ensuring that the inference does not collapse into trivial solutions. 
This is done by shuffling spot locations and producing corrupted covariance structures as negative samples.

.. code-block:: python

   from smoother import ContrastiveSpatialLoss

   sp_loss_fn = ContrastiveSpatialLoss(
      prior='icar', spatial_weights=weights, rho=0.99,
      num_perm=20, neg2pos_ratio=0.1, lower_bound = -1
   )
   loss = sp_loss_fn(x)

.. note::

   The corresponding covariance of the contrastive loss may not be positive semi-definite because of the negative sampling.
   To avoid exploding loss, the contrastive loss function has an intrinsic lower bound.

Downstream task applications
--------------------------------------------
Given the `SpatialLoss`, it is essentially possible to morph any model with a loss objective :math:`L_m` into a spatially aware version 
by minimizing a new joint loss function :math:`L_{joint} = L_m(X, ...) + \lambda L_{sp}(X; \Sigma)`. 
Optimization is generally performed using gradient-based methods. 
We have implemented a collection of models in the tasks of data imputation, cell-type deconvolution, and dimensionality reduction, 
which will be introduced with more details in the next section.

Example 1: Deconvolution using NNLS with spatial regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss
   from smoother.models.deconv import NNLS

   # Example 1: deconvolution using NNLS with spatial regularization
   x = np.random.rand(5, 10) # n_celltype x n_gene, the reference cell-type signatures
   y = np.random.rand(1000, 10) # n_spot x n_gene, the spatial count matrix
   coords = np.random.rand(1000, 2) # n_spot x 2, spatial coordinates

   # build spatial loss
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(coords, k=4, symmetric=True, row_scale=False)
   spatial_loss = SpatialLoss(prior='icar', spatial_weights=weights, rho=0.99, use_sparse=True, standardize_cov=False)

   # run deconvolution with spatial regularization
   model = NNLS(backend='pytorch')
   model.deconv(x.T, y.T, spatial_loss=spatial_loss, lambda_spatial_loss=1.0)
   ct_props = model.get_props() # n_spot x n_celltype


Example 2: Dimension reduction using spatially-aware VAE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss
   from smoother.models.reduction import SpatialVAE
   from anndata import AnnData

   # Example 2: dimension reduction with spatial regularization
   adata = AnnData(X=np.random.poisson(1.0, (1000, 100)).astype(float)) # n_spot x n_gene
   coords = np.random.rand(1000, 2) # n_spot x 2, spatial coordinates

   # build spatial loss
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(coords, k=4, symmetric=True, row_scale=False)
   spatial_loss = SpatialLoss(prior='icar', spatial_weights=weights, rho=0.99, use_sparse=True, standardize_cov=True)

   # train spatial VAE (a wrapper of SCVI with spatial loss)
   SpatialVAE.setup_anndata(adata)
   model = SpatialVAE(st_adata=adata, n_latent=10, spatial_loss=spatial_loss, lambda_spatial_loss=0.1, sp_loss_as_kl=True)
   model.train(max_epochs = 100, lr = 0.01, accelerator='cpu')
   sp_rep = model.get_latent_representation() # n_spot x n_latent


Example 3: Joint dimension reduction of spatial and non-spatial data with spatial regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss
   from smoother.models.reduction import SpatialVAE
   from anndata import AnnData
   from scvi.model import SCVI

   # Example 3: joint dimension reduction of spatial and non-spatial data with spatial regularization
   # first train a non-spatial model on scRNA-seq data
   sc_adata = AnnData(X=np.random.poisson(1.0, (2000, 100)).astype(float)) # n_cell x n_gene
   SCVI.setup_anndata(sc_adata)
   rna_scvi_model = SCVI(sc_adata)
   rna_scvi_model.train(max_epochs=50, accelerator='cpu')

   # then transfer the learned parameters to spatial model, and fine-tune with spatial loss
   st_data = AnnData(X=np.random.poisson(1.0, (1000, 100)).astype(float)) # n_spot x n_gene
   coords = np.random.rand(1000, 2) # n_spot x 2

   spvae_model = SpatialVAE.from_rna_model(
      st_adata=st_data, sc_model=rna_scvi_model,
      spatial_loss=spatial_loss, lambda_spatial_loss=0.1,
      sp_loss_as_kl=True,
      unfrozen=True,
   )
   spvae_model.train(max_epochs=100, lr=0.01, accelerator='cpu')
   st_rep = spvae_model.get_latent_representation() # n_spot x n_latent
