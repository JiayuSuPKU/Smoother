Basic usage
============

Smoother is a two-step framework that (1) extracts prior dependency structures from positional information and (2) integrates them into non-spatial models for downstream tasks to encourage local smoothness.
Here we briefly outline key components and steps of the Smoother pipeline. 

.. code-block:: python

   import torch
   import smoother
   from smoother import SpatialWeightMatrix, SpatialLoss, ContrastiveSpatialLoss
   from smoother.models.deconv import NNLS
   from smoother.models.reduction import SpatialPCA, SpatialVAE

**Spatial graph construction**
----------------------------------

We first need to construct a weighted spatial graph using physical positions, histology, and additional features, which serves to represent spatial dependencies a priori. 
The neighborhood graph is represented as an :math:`n \times n` adjacency matrix :math:`W` such that 
each entry :math:`W_{ij}` indicates the connectivity between sample :math:`i` and :math:`j` (referred to as `SpatialWeightMatrix`).

.. code-block:: python

   from smoother import SpatialWeightMatrix

   # spatial coordinates of samples
   coords = pd.read_csv(...) # n_spot x 2

   # construct the spatial graph and the corresponding weight matrix
   weights = SpatialWeightMatrix()

   # connect the k-nearest neighbors for each spot and
   # use binary edge weights for connectivity
   weights.calc_weights_knn(coords, k = 6)

   # or alternatively, connect spots by a hard-coded distance threshold and 
   # compute edge weights using pairwise distance
   weights.calc_weights_dist(coords, radius_cutoff = 1.0, band_width = 0.1) 

.. note::

   To encode information on region boundary, the weight matrix can be optionally scaled using additional information. 

This includes scaling by expression,

.. code-block:: python

   # scale weights by transcriptomics similarity
   y = torch.tensor(...) # the expression count matrix, n_gene x n_spot
   weights.scale_by_expr(y)

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
   weights.scale_by_histology(coords, img, scale_factors)

and by external annotations on spatial regions and clusters (no interaction between regions).

.. code-block:: python

   # hard-prune the graph to remove all unwanted interactions between regions
   class_anno = adata.obs['region'].to_numpy() # per-spot region/cluster annotation, n_spot x 1
   weights.scale_by_identity(class_anno, boundary_connectivity=0)

**Spatial loss construction**
----------------------------------

Subsequently, Smoother translates the spatial weights matrix into a covariance structure according to assumptions on the underlying stochastic process. 
The covariance is then converted into a modular sparse loss function (referred to as `SpatialLoss`) through a multivariate normal (MVN) prior. 
When applied to a random variable of interest, the spatial loss regularizes incoherence in the variable and thus improve performance in downstream tasks. 

.. code-block:: python

   from smoother import SpatialLoss

   # construct the spatial graph and the corresponding weight matrix
   coords = pd.read_csv(...) # n_spot x 2
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(coords)

   # optional weights scaling using weights.scale_by_
   ...

   # transform it into a spatial loss function using the ICAR model
   sp_loss_fn = SpatialLoss('icar', weights, rho=0.99)

   # regularize any spatial random variable of interest
   x = torch.tensor(...) # n_dim x n_spot
   loss = sp_loss_fn(x)

.. note::
   The inverse of covariance :math:`\Sigma_{n \times n}` is stored at `sp_loss_fn.inv_cov`. 
   During downstream optimizations the covariance is always fixed since it represents prior belief.
   The loss is proportional to the negative log likelihood of the prior :math:`L_{sp}(X; \Sigma) = \frac{1}{2}X^T \Sigma^{-1}X`.

Smoother offers five different yet related spatial processes: CAR, SAR (simultaneous autoregressive), ICAR, ISAR, and SMA (spatial moving average). 
Specifically, CAR and SAR are equivalent upon transformation, and ICAR and ISAR are the weights-scaled versions so that 
the autocorrelation parameter :math:`\rho` falls in [0, 1]. By adjusting :math:`\rho`, these models can achieve parallel regularization effects. 
Based on numerical considerations, we typically recommend using ICAR with varying :math:`\rho` (or ISAR with smaller :math:`\rho`) 
to accommodate data with diverse neighborhood structures, for instance, “ICAR (:math:`\rho = 0.99`)” for data with clear anatomy and 
“ICAR (:math:`\rho = 0.9`)” for tumor data. SMA is generally not recommended since the resulting inverse covariance matrix tends to be less sparse, potentially slowing down computation.

.. code-block:: python

   sp_loss_fn = SpatialLoss('icar', weights, rho=0.99)
   sp_loss_fn = SpatialLoss('isar', weights, rho=0.9)

In addition, we implement a contrastive extension of the spatial loss to increase the penalty for pulling distant spots too close, ensuring that the inference does not collapse into trivial solutions. 
This is done by shuffling spot locations and producing corrupted covariance structures as negative samples.

.. code-block:: python

   from smoother import ContrastiveSpatialLoss

   sp_loss_fn = ContrastiveSpatialLoss(
      spatial_weights=weights, num_perm=20, neg2pos_ratio=0.1, lower_bound = -1
   )
   loss = sp_loss_fn(x)

.. note::

   The corresponding covariance of the contrastive loss may not be positive semi-definite because of the negative sampling.
   To avoid exploding loss, the contrastive loss function has an intrinsic lower bound.

**Incorpotation into downstream tasks**
----------------------------------
Given the `SpatialLoss`, it is essentially possible to morph any model with a loss objective :math:`L_m` into a spatially aware version 
by minimizing a new joint loss function :math:`L_{joint} = L_m + \lambda L_{sp}(X_s; \Sigma)`. 

.. code-block:: python

   # choose model and solve the problem
   # deconvolution
   model = NNLS()
   model.deconv(x, y, spatial_loss=sp_loss_fn, lambda_spatial_loss=1, ...)

   # dimension reduction de novo from spatial data
   SpatialVAE.setup_anndata(adata, layer="raw")
   model = SpatialVAE(st_adata=adata, spatial_loss=sp_loss_fn)
   model.train(max_epochs = 400, lr = 0.01, accelerator='cpu')

   # dimension reduction from single-cell models
   baseline = SpatialPCA(rna_adata, layer='scaled', n_latent=30)
   baseline.reduce(...)
   model_sp = SpatialPCA.from_rna_model(
       rna_model=baseline, st_adata=sp_data, layer='scaled',
       spatial_loss=sp_loss_fn, lambda_spatial_loss=0.1
   )

   model_sp = SpatialVAE.from_rna_model(
       st_adata = sp_data, sc_model = rna_scvi_model, 
       spatial_loss=sp_loss, lambda_spatial_loss=0.01,
       unfrozen=True,
   )

['shells', 'gorgonzola', 'parsley']

