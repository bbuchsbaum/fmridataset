# Get Latent Scores from Dataset

Extract the latent scores (temporal components) from a latent dataset.
This is the primary data access method for latent datasets.

## Usage

``` r
get_latent_scores(x, rows = NULL, cols = NULL, ...)
```

## Arguments

- x:

  A latent_dataset object

- rows:

  Optional row indices (timepoints) to extract

- cols:

  Optional column indices (components) to extract

- ...:

  Additional arguments

## Value

Matrix of latent scores (time × components)

## See also

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md),
[`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md),
[`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)
