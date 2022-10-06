# Synthetic Data Generation
![Overview](/docs/img/simulation.png)

## Overview
This tool is designed to facilitate the simulation of 10x-Visium-style (capture-based) spatial transcriptomics data. 
We follow the simulation pipeline described in the [cell2location](https://pubmed.ncbi.nlm.nih.gov/35027729/) paper, with key modifications enabling us to: 
1. assign arbitrary spatial patterns to zones (co-localized cell type groups) and
2. add additional noise to account for cell-type-independent dependencies. 

In brief, we first overlay designated patterns with a two-dimensional Gaussian process to generate the per-zone abundance values in space ([Step 1](#step-1-patterns-to-zones)), then assign cell types to patterns and sample gene expression profiles at each location from a scRNA-seq reference according to the cell type composition ([Step 2](#step-2-zones-to-counts)). The aggregated ST data is further combined with various sources of noise, including a lateral diffusion term to introduce confounding dependencies where each spot shares a certain proportion of mRNA to its neighbors directly, and a multiplicative per-gene gamma noise (by default shape = 0.4586, scale = 1/0.6992) to represent sampling error. 

## Quick Start
Before running the simulation make sure all the prerequisites are installed. The tool requires [scanpy](https://scanpy.readthedocs.io/en/stable/) for processing and storing single-cell and ST data, and [squidpy](https://squidpy.readthedocs.io/en/stable/index.html) and [fuzzy-c-means](https://pypi.org/project/fuzzy-c-means/) for histology image processing (if needed).
```zsh
# installing dependencies
conda install -c conda-forge scanpy
pip install squidpy fuzzy-c-means
```

Generating donut-shaped simulated spatial dataset using a [mouse brain snRNA-seq reference](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-11115/):
```zsh
# cd <this_folder-path>
python 001b_donut2zone.py -ne 10 -ns 50 50 -rz 15 -uz 0 -o <abundance_df-path>

python 002_zone2counts.py -ne 10 -ns 50 50 -sm 0 \
    -d ./example_adata/E-MTAB-11115_subset.h5ad -l annotation_1 \
    -i <abundance_df-path>/zone_abundances.csv -o <output-path> 
```

Generating simulated spatial dataset based on a histological image of [human glioblastoma](https://support.10xgenomics.com/spatial-gene-expression/datasets/1.2.0/Parent_Visium_Human_Glioblastoma): 
```zsh
# cd <this_folder-path>
python 001c_hist2zone.py -ne 10 -ns 50 50 -rz 5 -uz 2 -i ./example_images/glioblastoma_1000x1000.png -o  <abundance_df-path>

python 002_zone2counts.py -ne 10 -ns 50 50 -f False \
    -d ./example_adata/E-MTAB-11115_subset.h5ad -l annotation_1 \
    -i <abundance_df-path>/zone_abundances.csv -o <output-path> 
```


## Descriptions of output files
### Spatial zone abundances (from step 1):
- `zone_abundances.csv` contains information about the abundance of each spatial zone at each location. <br>
    *(n_spots * n_exp) x (n_regional_zones + n_ubiquitous_zones)*

### Cell type abundances (from step 2): 
- `celltype_zone_assignment.csv` is the abundance matrix of cell types in spatial zones. <br>
    *n_cell_types x (n_regional_zones + n_ubiquitous_zones + 2)* <br>
    The last two columns denotes the category of each cell type:
    - `is_uniform`: 1 for being uniformly distributed (thus in ubiquitous zones).
    - `is_high_density`: 1 for being highly abundant (major cell types). The expected densities of high/low-density cell types are set with `-p`/`-muh -mul`.
    
- `celltype_abundances.csv` contains information about the captured abundance of each cell type at each location. <br>
    *(n_spots * n_exp) x n_cell_types* <br>
    **For deconvolution purpose it is used to calculate ground truth cell type proportions.**
- `celltype_counts.csv` is the discrete count of each cell type at each location used to sample single cells before down-scaling. <br>
    *(n_spots * n_exp) x n_cell_types*
- `celltype_capture_eff.csv` is the capture efficiency matrix calculated by ` celltype_abundances / celltype_counts` used to down-sample UMIs.
    Capture efficiency is always ≤1 and is lower at sparse spots which mimics the dropout event of single cell sequencing.

### Marker genes (from step 2):
- `markers_lfc.csv` contains the log2fold changes of marker genes identified using the Wilcoxon test. <br>
    *n_genes x n_cell_types*

### Deconvolution inputs (from step 2): 
Count matrices (keep only marker genes) and spatial coordinates stored in the `/deconv_inputs/` subfolder.
- `ref_avg_raw_count_markers.csv` is the reference count matrix for deconvolution calculated by averaging raw counts per cell type. <br>
    *n_genes x n_cell_types*
- `ref_avg_norm_count_markers.csv` is the normalized reference count matrix for deconvolution calculated by averaging depth-normalized counts (target_sum=1e4) per cell type. <br>
    *n_genes x n_cell_types*
- `syn_sp_count_markers_exp{i}.mtx` is the synthetic raw count matrix in experiment `i`. <br>
    *n_spots x n_genes*
- `coords_exp{i}.csv` contains the spatial coordinates of spots in experiment `i`. <br>
    *n_spots x 2*

### Synthetic anndata (from step 2):
**Optional** set `-oa True` or `--output_anndata True` to save all data generated from simulation
- `synthetic_sp_adata.h5ad`: gzip compressed `h5ad` file containing the synthetic anndata object
    - Cell-type-abundance-related dataframes are saved in the metadata `adata.obs`
    - coordinates for all experiments are saved in `adata.obsm['X_spatial']` 
    - UMI per cell type `ct` are calculated and saved in `adata.obs[f'UMI_count_{ct}']`
- `paired_sc_adata.h5ad`: gzip compressed `h5ad` file containing paired single-cell data (unseen in simulation).
    
## Visualization

Check [this notebook](./visualization.ipynb) for the visualization of synthetic data.


## Simulation details step-by-step:
### Step 1. Patterns to Zones

#### a. Random GP patterns as presented in the [cell2location](https://pubmed.ncbi.nlm.nih.gov/35027729/) paper
```
usage: 001a_c2l_gp.py [-h] [-ne N_EXPERIMENTS] [-ns N_SPOTS N_SPOTS] -rz N_REGIONAL_ZONES -uz N_UBIQUITOUS_ZONES -o
                      OUT_DIR [-vr VARIANCE_RG] [-vu VARIANCE_UB] [-s SEED]

options:
  -h, --help            show this help message and exit
  -ne N_EXPERIMENTS, --n_experiments N_EXPERIMENTS
                        Number of simulated experiments wanted
  -ns N_SPOTS N_SPOTS, --n_spots N_SPOTS N_SPOTS
                        Number of spots, needs to be 2 int n_row n_col
                        Example: --n_spots 50 50
  -rz N_REGIONAL_ZONES, --n_regional_zones N_REGIONAL_ZONES
                        Number of regional zones with sparse spatial patterns
  -uz N_UBIQUITOUS_ZONES, --n_ubiquitous_zones N_UBIQUITOUS_ZONES
                        Number of ubiquitous zones with uniform spatial patterns
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory for abundance matrix
  -vr VARIANCE_RG, --variance_rg VARIANCE_RG
                        Variance required for GP (regional zones)Increase this will increase the sparsity of patterns
  -vu VARIANCE_UB, --variance_ub VARIANCE_UB
                        Variance required for GP (ubiquitous zones)Increase this will increase the sparsity of
                        patterns
  -s SEED, --seed SEED  Random seed
```

#### b. Donut-shaped patterns 
```
usage: 001b_donut2zone.py [-h] [-ne N_EXPERIMENTS] [-ns N_SPOTS N_SPOTS] -rz N_REGIONAL_ZONES
                          [-uz N_UBIQUITOUS_ZONES] -o OUT_DIR [-w WIDTH] [-v VARIANCE] [-s SEED]

options:
  -h, --help            show this help message and exit
  -ne N_EXPERIMENTS, --n_experiments N_EXPERIMENTS
                        Number of simulated experiments wanted
  -ns N_SPOTS N_SPOTS, --n_spots N_SPOTS N_SPOTS
                        Number of spots, needs to be 2 int n_row n_col
                        Example: --n_spots 50 50
  -rz N_REGIONAL_ZONES, --n_regional_zones N_REGIONAL_ZONES
                        Number of regional zones with sparse spatial patterns
  -uz N_UBIQUITOUS_ZONES, --n_ubiquitous_zones N_UBIQUITOUS_ZONES
                        Number of ubiquitous zones with uniform spatial patterns
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory for abundance matrix
  -w WIDTH, --width WIDTH
                        Width of each concentric circular patternsIncrease this will increase the overlap between
                        each zone.
  -v VARIANCE, --variance VARIANCE
                        Variance required for GP (all zones)Increase this will increase the sparsity of patterns
  -s SEED, --seed SEED  Random seed

```

#### c. Histology-based patterns 
```
usage: 001c_hist2zone.py [-h] [-ne N_EXPERIMENTS] [-ns N_SPOTS N_SPOTS] -rz N_REGIONAL_ZONES [-uz N_UBIQUITOUS_ZONES]
                         -o OUT_DIR [-i IMG_DIR] [-m M_FUZZY] [-v VARIANCE] [-s SEED]

options:
  -h, --help            show this help message and exit
  -ne N_EXPERIMENTS, --n_experiments N_EXPERIMENTS
                        Number of simulated experiments wanted
  -ns N_SPOTS N_SPOTS, --n_spots N_SPOTS N_SPOTS
                        Number of spots, needs to be 2 int n_row n_col
                        Example: --n_spots 50 50
  -rz N_REGIONAL_ZONES, --n_regional_zones N_REGIONAL_ZONES
                        Number of regional zones with sparse spatial patterns
  -uz N_UBIQUITOUS_ZONES, --n_ubiquitous_zones N_UBIQUITOUS_ZONES
                        Number of ubiquitous zones with uniform spatial patterns
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory for abundance matrix
  -i IMG_DIR, --img_dir IMG_DIR
                        Path to histological image
  -m M_FUZZY, --m_fuzzy M_FUZZY
                        m for fuzzy c-means clustringIncrease this will increase the fuzziness of patterns.m should
                        be an int >=1 (default 2)
  -v VARIANCE, --variance VARIANCE
                        Variance required for GP (all zones)Increase this will increase the sparsity of patterns
  -s SEED, --seed SEED  Random seed
```

#### d. Custom patterns
To use a custom pattern, user needs to provide a non-negative `n_row` by `n_col` matrix, similar to the generated `zone_abundances.csv`, where:
- `n_row = n_spots[0] * n_spots[1] * n_experiments`
- `n_col = n_regional_zones + n_ubiquitous_zones`

### Step 2. Zones to Counts
**NOTE:** the number of experiments `-ne` and the number of spots `-ns` must be the same as used in **Step 1**. 

```
usage: 002_zone2counts.py [-h] -i ABUNDANCE_DIR -d REF_DIR -l ANN_LABEL [-o OUT_DIR] [-oa | --output-anndata | --no-output-anndata]
                          [-ne N_EXPERIMENTS] [-ns GRID_SIZE GRID_SIZE] [-s SEED] [-sp | --split | --no-split] [-ph P_HIGH_DENSITY]
                          [-muh MU_HIGH_DENSITY] [-mul MU_LOW_DENSITY] [-f | --multi-pattern | --no-multi-pattern] [-sm SMOOTH_SCALE]
                          [-gn | --gamma_noises | --no-gamma_noises] [-lfcmin LOG2FC_MIN] [-pv P_VALUE] [-nm N_MARKERS]

optional arguments:
  -h, --help            show this help message and exit
  -i ABUNDANCE_DIR, --abundance_dir ABUNDANCE_DIR
                        Path to zone abundance matrix (.csv).
  -d REF_DIR, --ref_dir REF_DIR
                        Path to reference single cell dataset (anndata).
  -l ANN_LABEL, --ann_label ANN_LABEL
                        Key to access celltype information in the reference anndata.
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory. Default: None to use the same directory as the input.
  -oa, --output-anndata, --no-output-anndata
                        Whether to output entire anndata objects for synthetic spatial data and paired single cell data. Default: True. 
  -ne N_EXPERIMENTS, --n_experiments N_EXPERIMENTS
                        Number of experiments to simulate. Default: 1.
  -ns GRID_SIZE GRID_SIZE, --grid_size GRID_SIZE GRID_SIZE
                        Grid size, needs to be 2 int n_row, n_col. Default: 50 50.
  -s SEED, --seed SEED  Random seed. Default: 253286.
  -sp, --split, --no-split
                        Split training and testing set to seperate single cells used in data simulation and deconvolution. Default: True.
  -ph P_HIGH_DENSITY, --p_high_density P_HIGH_DENSITY
                        Proportion of high-density cell types in each zone. Default: 0.4.
  -muh MU_HIGH_DENSITY, --mu_high_density MU_HIGH_DENSITY
                        Average abundance for high-density cell types. Default: 4.0.
  -mul MU_LOW_DENSITY, --mu_low_density MU_LOW_DENSITY
                        Average abundance for low-density cell types. Default: 0.4.
  -f, --multi-pattern, --no-multi-pattern
                        Flag specifies whether celltypes have several spatial patterns number of patterns per celltype is sampled with a Gamma distribution. Default: False.
  -sm SMOOTH_SCALE, --smooth_scale SMOOTH_SCALE
                        Scale for smoothing from neighboring spots specifying the proportion of cells that are shared with neighboring spots. Default:
                        0.
  -gn, --gamma_noises, --no-gamma_noises
                        Add gamma noises to the synthetic count matrix. Default: True.
  -lfcmin LOG2FC_MIN, --log2fc_min LOG2FC_MIN
                        Minimum log2fc for marker gene selection. Default: 1.0.
  -pv P_VALUE, --p_value P_VALUE
                        P-value threshold for marker gene selection. Default: 0.01.
  -nm N_MARKERS, --n_markers N_MARKERS
                        Number of marker genes per celltype in the simulated count matrix. Default: 0 to return all genes passed log2fc_min and
                        p_value_threshold.
```
