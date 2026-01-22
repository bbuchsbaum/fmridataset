# Latent Dataset Interface

A specialized dataset interface for working with latent space
representations of fMRI data. Unlike traditional fMRI datasets that work
with voxel-space data, latent datasets operate on compressed
representations using basis functions.

This interface is designed for data that has been decomposed into
temporal components (basis functions) and spatial loadings, such as from
PCA, ICA, or dictionary learning methods.

Creates a dataset object for working with latent space representations
of fMRI data. This is the primary constructor for latent datasets.

## Usage

``` r
latent_dataset(
  source,
  TR,
  run_length,
  event_table = data.frame(),
  base_path = ".",
  censor = NULL,
  preload = FALSE
)
```

## Arguments

- source:

  Character vector of file paths to LatentNeuroVec HDF5 files (.lv.h5)
  or a list of LatentNeuroVec objects from the fmristore package.

- TR:

  The repetition time in seconds.

- run_length:

  Vector of integers indicating the number of scans in each run.

- event_table:

  Optional data.frame containing event onsets and experimental
  variables.

- base_path:

  Base directory for relative file paths.

- censor:

  Optional binary vector indicating which scans to remove.

- preload:

  Logical indicating whether to preload all data into memory.

## Value

A `latent_dataset` object with class
`c("latent_dataset", "fmri_dataset")`.

## Details

### Key Differences from Standard Datasets:

- **Data Access**: Returns latent scores (time × components) instead of
  voxel data

- **Mask**: Represents active components, not spatial voxels

- **Dimensions**: Component space rather than voxel space

- **Reconstruction**: Can optionally reconstruct to voxel space on
  demand

### Data Structure:

Latent representations store data as:

- `basis`: Temporal components (n_timepoints × k_components)

- `loadings`: Spatial components (n_voxels × k_components)

- `offset`: Optional per-voxel offset terms

- Reconstruction: `data = basis %*% t(loadings) + offset`

## See also

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md),
[`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md),
[`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md),
[`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md),
[`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# From LatentNeuroVec files
dataset <- latent_dataset(
  source = c("run1.lv.h5", "run2.lv.h5"),
  TR = 2,
  run_length = c(100, 100)
)

# From pre-loaded objects
lvec1 <- fmristore::read_vec("run1.lv.h5")
lvec2 <- fmristore::read_vec("run2.lv.h5")
dataset <- latent_dataset(
  source = list(lvec1, lvec2),
  TR = 2,
  run_length = c(100, 100)
)

# Access latent scores
scores <- get_latent_scores(dataset)

# Get component metadata
comp_info <- get_component_info(dataset)
} # }
```
