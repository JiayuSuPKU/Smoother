Installation
============

Minimal installation
------------------------

If you only want to use the core functionalities, namely :class:`smoother.SpatialWeightMatrix`
and :class:`smoother.SpatialLoss`, Smoother can be directly installed using `pip`.

.. code-block:: zsh

   # from PyPI (stable version)
   $ pip install smoother-omics

   # or from github (latest version)
   $ pip install git+https://github.com/JiayuSuPKU/Smoother.git#egg=smoother

.. note::

   This will be enough if you do not plan to run spatial dimensionality reduction models and convex optimization solvers.

Minimal software dependencies include

.. code-block:: text

   torch
   scipy
   scikit-learn
   pandas
   tqdm

Full installation
------------------------

Models in the dimensionality reduction module (:class:`smoother.SpatialAE`, :class:`smoother.SpatialVAE`) is built upon `scvi-tools`. 
Here we refer to the `original repository for installation instructions on different systems <https://docs.scvi-tools.org/en/stable/installation.html>`_.

.. code-block:: zsh

   $ pip install scvi-tools

.. note::

   scvi-tools` doesn't officially support Apple's M chips yet. To run `SCVI` and the corresponding `SpatialVAE` on Macs with Apple silicon, 
   a temporary solution is to `compile both Pytorch and PyG on M1 chips using compatible wheel files <https://github.com/rusty1s/pytorch_scatter/issues/241#issuecomment-1086887332>`_.

.. code-block:: zsh

   $ conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64

   $ MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch

   $ MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-geometric

   $ pip install scvi-tools

To solve data imputation and deconvolution models using convex optimization, you need to also install the `'cvxpy' package <https://www.cvxpy.org/>`_.

.. code-block:: zsh

   $ conda install -c conda-forge cvxpy

To run other functions, e.g., the simulation scripts, we recommend using the conda environment provided in the repo. 
You can create a new conda environment called 'smoother' and install the package in it using the following commands:

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

The following software dependencies specified in the `environment.yml` will be installed

.. code-block:: text

   name: smoother
   channels:
   - conda-forge
   dependencies:
   - python<4.0
   - pip
   - scipy
   - pytorch
   - pandas
   - scanpy
   - python-igraph 
   - leidenalg
   - scvi-tools
   - scikit-learn
   - matplotlib==3.5.3
   - plotnine==0.8.0
   - jupyterlab
   - ipywidgets
   - pynndescent
   - cvxpy
   - pip:
      - squidpy
      - fuzzy-c-means
      - scikit-bio==0.5.8
