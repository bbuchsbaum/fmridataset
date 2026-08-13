# Reconstruct Voxel-Space Data from a Latent-Mode BIDS H5 Dataset

Reconstructs full voxel-space time series for a scan by computing
`basis %*% t(loadings) + offset`. Only available when
`compression_mode = "latent"`.

Reconstruct the full voxel-space data from the latent representation.
This is computationally expensive and should be used sparingly.

## Usage

``` r
reconstruct_voxels(x, scan_name = NULL, rows = NULL, voxels = NULL, ...)

# S3 method for class 'bids_h5_study_dataset'
reconstruct_voxels(x, scan_name, rows = NULL, voxels = NULL, ...)

# S3 method for class 'latent_dataset'
reconstruct_voxels(x, scan_name = NULL, rows = NULL, voxels = NULL, ...)
```

## Arguments

- x:

  A `bids_h5_study_dataset` object opened in latent mode.

- scan_name:

  Character. Name of the scan to reconstruct.

- rows:

  Integer vector of timepoint indices to return, or `NULL` for all.

- voxels:

  Integer vector of voxel indices to return, or `NULL` for all.

- ...:

  Additional arguments passed to methods.

## Value

A numeric matrix `[T, V]` (or subset thereof).

## See also

Other latent_data:
[`get_component_info()`](https://bbuchsbaum.github.io/fmridataset/reference/get_component_info.md),
[`get_latent_scores()`](https://bbuchsbaum.github.io/fmridataset/reference/get_latent_scores.md),
[`get_spatial_loadings()`](https://bbuchsbaum.github.io/fmridataset/reference/get_spatial_loadings.md),
[`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md)
