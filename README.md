# Smoother: Modular Spatial Regularization for Spatial Omics
[![PyPI](https://img.shields.io/pypi/v/smoother-omics.svg)](https://pypi.org/project/smoother-omics/)
[![Downloads](https://static.pepy.tech/badge/smoother-omics)](https://pepy.tech/project/smoother-omics)
[![Test status](https://github.com/JiayuSuPKU/Smoother/actions/workflows/tests.yml/badge.svg)](https://github.com/JiayuSuPKU/Smoother/actions/workflows/tests.yml)
[![Build status](https://github.com/JiayuSuPKU/Smoother/actions/workflows/build.yml/badge.svg)](https://github.com/JiayuSuPKU/Smoother/actions/workflows/build.yml)


## A unified and modular framework to incorporate structural dependency
Smoother is a Python/PyTorch toolkit for modeling spatial dependency and enforcing spatial coherence in spatial omics. It decouples spatial priors (neighbor similarity) from non-spatial likelihoods, so you can plug spatial structure into existing models with minimal changes. It is modular and fast, scaling to tens of thousands of spots in seconds.

What it provides:
1. **Spatial loss:** A quadratic loss equivalent to a Gaussian random field (MVN) prior derived from a boundary-aware graph.
2. **Imputation and resolution enhancement:** Denoise by borrowing from neighbors and upscale to arbitrary resolutions in seconds.
3. **Cell-type deconvolution:** Spatially coherent abundance estimates using cell-type references.
4. **Dimensionality reduction:** Spatially aware embeddings that also enable joint analysis with non-spatial single-cell data.


> **_NOTE:_** As of v1.1.0, we have implemented spatially-aware versions of `SCVI`, `SCANVI` and `MULTIVI` ([scvi-tools](https://scvi-tools.org/) v1.4.0), named `SpatialVAE`, `SpatialSCANVI` and `SpatialMULTIVI`, respectively. See implementation details in [smoother/models/reduction/](/smoother/models/reduction/).

![Overview](/docs/img/Smoother_overview.png)

## Resources
* For basic usages, tutorials and examples, see the [documentation page](https://smoother.readthedocs.io/en/latest/index.html). 
* For mathematical details, please refer to the [Smoother paper (Su Jiayu, et al. 2023)](https://link.springer.com/article/10.1186/s13059-023-03138-x) and the [Supplementary Notes](/docs/Smoother_sup_notes.pdf).

## Installation
### Basic installation
To use core functionalities (`SpatialWeightMatrix` and `SpatialLoss`), Smoother can be directly installed using `pip` either from github (latest version)
```zsh
pip install git+https://github.com/JiayuSuPKU/Smoother.git#egg=smoother-omics
```
or from PyPI (stable version)
```zsh
pip install smoother-omics
```

### Optional dependencies
To use convex optimization-based models for data imputation and deconvolution, you need to also install the [CVXPY package](https://www.cvxpy.org/).
```zsh
pip install smoother-omics[cvxpy]
```

To use spatially-aware VAE models (`SpatialVAE`, `SpatialANVI`, `SpatialMULTIVI`), you need to also install the [scvi-tools package](https://scvi-tools.org/) (v1.4.0 or above, which may require extra dependencies including [JAX](https://jax.readthedocs.io/en/latest/)).
```zsh
pip install smoother-omics[scvi]
```

<!-- To run [simulation scripts](/simulation/README.md), we recommend using the Conda environment provided in the repo. You can create a new conda environment called 'smoother' and install the package in it using the following commands:
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
``` -->

## Basic usage
### Spatial loss construction
```python
# import spatial losses and models
import numpy as np
import torch
from smoother import SpatialWeightMatrix, SpatialLoss

# load data
x = np.random.rand(1000, 5) # n_spot x n_gene, the spatial count matrix
coords = np.random.rand(1000, 2) # n_spot x 2, spatial coordinates

# build spatial weight matrix from spatial coordinates
weights = SpatialWeightMatrix()
weights.calc_weights_knn(coords, k=4, symmetric=True, row_scale=False) # mutual kNN graph weights

# (optional) scale weights by transcriptomics similarity
weights.scale_by_expr(x.T, dist_metric='cosine', reduce='none')

# transform it into spatial loss
spatial_loss = SpatialLoss(prior='icar', spatial_weights=weights, rho=0.99, use_sparse=True, standardize_cov=False)
# by default the inverse spatial covariance matrix is stored as a sparse matrix in spatial_loss.inv_cov
spatial_loss.inv_cov.shape # -> (1, 1000, 1000)

# regularize any spatial random variable of interest
variable_of_interest = torch.randn(2, 1000) # n_dim x n_spot
loss = spatial_loss(variable_of_interest) # losses are summed over all dimensions, returning a scalar
```

### Downstream tasks
#### Example 1: Deconvolution using NNLS with spatial regularization
```python
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
```

#### Example 2: Dimension reduction using spatially-aware VAE
```python
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
```

#### Example 3: Joint dimension reduction of spatial and non-spatial data with spatial regularization
```python
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
```

## Reference
If you use `Smoother` in your research, please cite
> Su, Jiayu, et al. "Smoother: a unified and modular framework for incorporating structural dependency in spatial omics data." 
>
> Genome Biology 24.1 (2023): 291.
https://link.springer.com/article/10.1186/s13059-023-03138-x

```bibtex
@article{su2023smoother,
  title={Smoother: a unified and modular framework for incorporating structural dependency in spatial omics data},
  author={Su, Jiayu and Reynier, Jean-Baptiste and Fu, Xi and Zhong, Guojie and Jiang, Jiahao and Escalante, Rydberg Supo and Wang, Yiping and Aparicio, Luis and Izar, Benjamin and Knowles, David A and Rabadan, Raul},
  journal={Genome Biology},
  volume={24},
  number={1},
  pages={291},
  year={2023},
  publisher={Springer}
}
```