# Reconstruct Voxel Data from Latent Representation

Reconstruct the full voxel-space data from the latent representation.
This is computationally expensive and should be used sparingly.

## Usage

``` r
reconstruct_voxels(x, rows = NULL, voxels = NULL, ...)
```

## Arguments

- x:

  A latent_dataset object

- rows:

  Optional row indices (timepoints) to reconstruct

- voxels:

  Optional voxel indices to reconstruct

- ...:

  Additional arguments

## Value

Matrix of reconstructed voxel data (time × voxels)

## See also

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md),
[`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md),
[`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md)
