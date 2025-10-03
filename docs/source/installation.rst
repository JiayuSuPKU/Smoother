Installation
============

Minimal installation
------------------------

To use core functionalities (:class:`smoother.SpatialWeightMatrix` and :class:`smoother.SpatialLoss`), 
Smoother can be directly installed using `pip`

.. code-block:: zsh

   # from PyPI (stable version)
   $ pip install smoother-omics

   # or from github (latest version)
   $ pip install git+https://github.com/JiayuSuPKU/Smoother.git#egg=smoother-omics

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

Models in the dimensionality reduction module (:class:`smoother.models.reduction.SpatialVAE`, :class:`smoother.models.reduction.SpatialANVI`, :class:`smoother.models.reduction.SpatialMULTIVI`)
is built upon `scvi-tools (v1.4.0) <https://scvi-tools.org/>`_. We refer to the `original repository for installation instructions on different systems <https://docs.scvi-tools.org/en/stable/installation.html>`_.

.. code-block:: zsh

   # either separately install scvi-tools
   $ pip install torch jax scvi-tools==1.4.0

   # or install it together with Smoother
   $ pip install smoother-omics[scvi]


To solve data imputation and deconvolution models using convex optimization, you need to also install the `CVXPY package <https://www.cvxpy.org/>`_.

.. code-block:: zsh

   # either separately install cvxpy
   $ pip install cvxpy==1.7.3

   # or install it together with Smoother
   $ pip install smoother-omics[cvxpy]

To run simulation scripts under `/simulation <https://github.com/JiayuSuPKU/Smoother/tree/main/simulation>`_, we recommend using the Conda environment provided in the repo. 
You can create a new Conda environment called 'smoother' and install the package in it using the following commands:

.. code-block:: zsh

   # download the repo from github
   $ git clone git@github.com:JiayuSuPKU/Smoother.git

   # cd into the repo and create a new conda environment called 'smoother'
   $ conda env create --file environment.yml
   $ conda activate smoother

   # add the new conda enviroment to Jupyter
   $ python -m ipykernel install --user --name=smoother

   # install the package
   $ pip install -e .[scvi,cvxpy]

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

Reporting issues
------------------------
If you encounter any issues during installation or usage, please report them on the `GitHub Issues page <https://github.com/JiayuSuPKU/Smoother/issues>`_.
