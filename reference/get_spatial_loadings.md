# Get Spatial Loadings from Dataset

Extract the spatial loadings (spatial components) from a latent dataset.

## Usage

``` r
get_spatial_loadings(x, components = NULL, ...)
```

## Arguments

- x:

  A latent_dataset object

- components:

  Optional component indices to extract

- ...:

  Additional arguments

## Value

Matrix or sparse matrix of spatial loadings (voxels × components)

## See also

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md),
[`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md),
[`reconstruct_voxels()`](https://bbuchsbaum.github.io/fmridataset/reference/reconstruct_voxels.md)
