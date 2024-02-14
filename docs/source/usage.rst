Usage
=====

.. _installation:

**Installation**
-----------------

If you only want to use the core functionalities, namely `SpatialWeightMatrix` and `SpatialLoss`, Smoother can be directly installed using `pip`:

.. code-block:: zsh

   pip install git+https://github.com/JiayuSuPKU/Smoother.git#egg=smoother

The dimensionality reduction module (`SpatialAE`, `SpatialVAE`) is built upon `scvi-tools`. Here we refer to the `original repository for installation instructions on different systems <https://docs.scvi-tools.org/en/stable/installation.html>`_.

.. code-block:: zsh

   pip install scvi-tools

Note that `scvi-tools` doesn't officially support Apple's M chips yet. To run `SCVI` and the corresponding `SpatialVAE` on Macs with Apple silicon, a temporary solution is to compile both Pytorch and PyG on M1 chips using compatible wheel files.

To solve data imputation and deconvolution models using convex optimization, you need to also install the 'cvxpy' package.

.. code-block:: zsh

   conda install -c conda-forge cvxpy

To run other functions, e.g., the simulation scripts, we recommend using the conda environment provided in the repo. You can create a new conda environment called 'smoother' and install the package in it using the following commands:

.. code-block:: zsh

   # download the repo from github
   git clone git@github.com:JiayuSuPKU/Smoother.git

   # cd into the repo and create a new conda environment called 'smoother'
   conda env create --file environment.yml
   conda activate smoother

   # add the new conda enviroment to Jupyter
   python -m ipykernel install --user --name=smoother

   # install the package
   pip install -e .

.. _example_usage:

**Basic usage:**
-----------------

**Spatial loss construction:**

.. code-block:: python

   # import spatial losses and models
   import torch
   from smoother import SpatialWeightMatrix, SpatialLoss, ContrastiveSpatialLoss
   from smoother.models.deconv import NNLS
   from smoother.models.reduction import SpatialPCA, SpatialVAE

   # load data
   x = torch.tensor(...) # n_gene x n_celltype, the reference signature matrix
   y = torch.tensor(...) # n_gene x n_spot, the spatial count matrix
   coords = pd.read_csv(...) # n_spot x 2, spatial coordinates

   # build spatial weight matrix
   weights = SpatialWeightMatrix()
   weights.calc_weights_knn(coords)

   # scale weights by transcriptomics similarity
   weights.scale_by_expr(y)

   # transform it into spatial loss
   spatial_loss = SpatialLoss('icar', weights, rho=0.99)

   # or contrastive loss
   spatial_loss = ContrastiveSpatialLoss(
       spatial_weights=weights, num_perm=20, neg2pos_ratio=0.1)

   # regularize any spatial random variable of interest
   variable_of_interest = torch.tensor(...) # n_vars x n_spot
   loss = spatial_loss(variable_of_interest)

**Downstream tasks:**

.. code-block:: python

   # choose model and solve the problem
   # deconvolution
   model = NNLS()
   model.deconv(x, y, spatial_loss=spatial_loss, lambda_spatial_loss=1, ...)

   # dimension reduction de novo from spatial data
   SpatialVAE.setup_anndata(adata, layer="raw")
   model = SpatialVAE(st_adata=adata, spatial_loss=spatial_loss)
   model.train(max_epochs = 400, lr = 0.01, accelerator='cpu')

   # dimension reduction from single-cell models
   baseline = SpatialPCA(rna_adata, layer='scaled', n_latent=30)
   baseline.reduce(...)
   model_sp = SpatialPCA.from_rna_model(
       rna_model=baseline, st_adata=sp_data, layer='scaled',
       spatial_loss=spatial_loss, lambda_spatial_loss=0.1
   )

   model_sp = SpatialVAE.from_rna_model(
       st_adata = sp_data, sc_model = rna_scvi_model, 
       spatial_loss=sp_loss, lambda_spatial_loss=0.01,
       unfrozen=True,
   )

['shells', 'gorgonzola', 'parsley']

