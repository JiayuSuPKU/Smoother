# Smoother: A Unified and Modular Framework to Incorporate Structural Dependency in Spatial Omics Data
[![DOI](https://zenodo.org/badge/546425993.svg)](https://zenodo.org/doi/10.5281/zenodo.10242921)
![Overview](/docs/img/Smoother_overview.png)

## Description
Smoother is a Python package that provides a unified framework to incorporate dependency in spatial omics data by formulating spatial priors from neighborhood graph. Implemented in `Pytorch`, Smoother is modular and ultra-efficient, often capable of analyzing samples tens of thousands of spots in seconds. The key innovation of Smoother is the decoupling of the prior belief on spatial structure (i.e., neighboring spots tend to be more similar) from the likelihood of a non-spatial data-generating model. This flexibility allows the same prior to be used in different models, and the same model to accommodate data with varying or even zero spatial structures. In other words, Smoother can be seamlessly integrated into existing non-spatial models and pipelines (e.g. single-cell analyses) and make them spatially aware. In particular, Smoother provides the following functionalities:

1. **Spatial loss**: A quadratic loss equivalent to an MVN prior reflecting the spatial structure of the data. It can be used to regularize any spatial random variable of interest.
2. **Data imputation**: Mitigates technical noise by borrowing information from the neighboring spots. It can also be applied to enhance the resolution of the data to an arbitrary level in seconds.
3. **Cell-type deconvolution**: Infers the spatially coherent cell-type composition of each spot using reference cell-type expression profiles. Smoother is one of the few deconvolution methods that actually enforce spatial coherence by design.
4. **Dimension reduction**: Find the spatially aware latent representations of spatial omics data in a model-agnostic manner, such that single-cell data without spatial structure can be jointly analyzed using the same pipeline.

For method details, check [the Smoother paper (Su Jiayu, et al. 2022)](https://www.biorxiv.org/content/10.1101/2022.10.25.513785v2.full) and [the Supplementary Notes](/docs/Smoother_sup_notes.pdf).

## Installation
If you only want to use the core functionalities, namely `SpatialWeightMatrix` and `SpatialLoss`, Smoother can be directly installed using `pip` 
```zsh
pip install git+https://github.com/JiayuSuPKU/Smoother.git#egg=smoother
```

The dimensionality reduction module (`SpatialAE`, `SpatialVAE`) is built upon [scvi-tools](https://docs.scvi-tools.org/en/stable/index.html). Here we refer to the [original repository for installation instructions on different systems](https://docs.scvi-tools.org/en/stable/installation.html).
```zsh
pip install scvi-tools
```


- Note that `scvi-tools` doesn't officially support Apple's M chips yet. To run `SCVI` and the corresponding `SpatialVAE` on Macs with Apple silicon, a temporary solution is to [compile both Pytorch and PyG on M1 chips using compatible wheel files](https://github.com/rusty1s/pytorch_scatter/issues/241#issuecomment-1086887332). 

To solve data imputation and deconvolution models using convex optimization, you need to also install the ['cvxpy' package](https://www.cvxpy.org/).

```zsh
conda install -c conda-forge cvxpy
```

To run other functions, e.g. the [simulation scripts](/simulation/README.md), we recommend using the conda environment provided in the repo. You can create a new conda environment called 'smoother' and install the package in it using the following commands:
```zsh
# download the repo from github
git clone git@github.com:JiayuSuPKU/Smoother.git

# cd into the repo and create a new conda environment called 'smoother'
conda env create --file environment.yml
conda activate smoother

# add the new conda enviroment to Jupyter
python -m ipykernel install --user --name=smoother

# install the package
pip install -e .
```

## Basic usage:
### Spatial loss construction:
```python
# import spatial losses and models
import torch
from smoother import SpatialWeightMatrix, SpatialLoss, ContrastiveSpatialLoss
from smoother.models.deconv import NNLS
from smoother.models.reduction import SpatialPCA, SpatialVAE

# load data
x = torch.tensor(...) # n_gene x n_celltype, the reference signature matrix
y = torch.tensor(...) # n_gene x n_spot, the spatial count matrix
coords = pd.read_csv(...) # n_spot x 2, tspatial coordinates

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
```

### Downstream tasks:
```python
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
```

## Smoother tutorials:
Under construction. Check back soon!
1. ~~[Smoother-guided data imputation in the DLPFC dataset](/tutorials/tutorial_impute.ipynb)~~
2. ~~[Smoother-guided cell-type deconvolution in the DLPFC dataset](/tutorials/tutorial_deconv.ipynb)~~
3. ~~[Smoother-guided dimension reduction in the DLPFC dataset](/tutorials/tutorial_dr.ipynb)~~
4. [Spatial transcriptomics data simulation](/simulation/README.md)

## References:
Su, Jiayu, et al. "Smoother: A Unified and Modular Framework for Incorporating Structural Dependency in Spatial Omics Data." bioRxiv (2022): 2022-10.
https://www.biorxiv.org/content/10.1101/2022.10.25.513785v2.full