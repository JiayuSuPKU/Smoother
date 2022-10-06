# Smoother: A Unified and Modular Framework to Incorporate Structural Dependency in Spatial Omics Data
![Overview](/docs/img/Smoother_overview.png)
Check [the notes](/docs/Smoother_sup_notes.pdf) and [documentations](/docs/smoother/index.md) for method details.

## Installation
We recommend clone the repository and use `conda` to manage dependencies, especially if you want to use the [simulation scripts](/simulation/README.md).
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
Alternatively, Smoother can be directly installed using `pip`
```zsh
#pip install git+https://github.com/
```


(Optional) To solve the deconvolution problem via convex optimization, 
you need to also install the ['cvxpy' package](https://www.cvxpy.org/).

```zsh
conda install -c conda-forge cvxpy
```

(Optional) To use the topological loss, you need to also install the 
['TopologyLayer' package](https://github.com/bruel-gabrielsson/TopologyLayer).

```zsh
pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git
```

## Smoother tutorials:
1. [Smoother-guided data imputation in the DLPFC dataset](/tutorials/tutorial_impute.ipynb)
2. [Smoother-guided cell-type deconvolution in the DLPFC dataset](/tutorials/tutorial_deconv.ipynb)
3. [Smoother-guided dimension reduction in the DLPFC dataset](/tutorials/tutorial_dr.ipynb)
4. [Spatial transcriptomics data simulation](/simulation/README.md)

Sample usage:
```python
# import different components
import torch
from smoother import SpatialWeightMatrix, SpatialLoss
from smoother.models.deconv import NNLS

# load data
x = torch.tensor(...) # n_gene x n_celltype, the reference signature matrix
y = torch.tensor(...) # n_gene x n_spot, the spatial count matrix
coords = pd.read_csv(...) # n_spot x 2, tspatial coordinates

# build spatial weight matrix
weights = SpatialWeightMatrix()
weights.calc_weights_knn(coords)

# transform it into spatial loss
spatial_loss = SpatialLoss('icar', weights, scale_weights=0.99)

# choose deconvolution model and solve the problem
model = NNLS()
model.deconv(x, y, spatial_loss=spatial_loss, lambda_spatial_loss=1, ...)

# get estimated cell type proportions
ct_props = model.get_props()
```