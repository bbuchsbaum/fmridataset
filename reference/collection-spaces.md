# Summarize collection feature spaces

Summarize collection feature spaces

## Usage

``` r
collection_space_data(x)

collection_common_space(x)
```

## Arguments

- x:

  An `fmri_collection`.

## Value

`collection_space_data()` returns one metadata row per frame;
`collection_common_space()` returns whether every feature space is
exactly compatible with the first.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
collection <- fmri_collection(list(sub01 = frame, sub02 = frame))
collection_space_data(collection)
#> # A tibble: 2 × 5
#>   .frame_id n_observation n_feature space_type   space_digest                   
#>   <chr>             <int>     <int> <chr>        <chr>                          
#> 1 sub01                 3         4 volume_space ca8461c12469d60a05dc7ee4dd058c…
#> 2 sub02                 3         4 volume_space ca8461c12469d60a05dc7ee4dd058c…
collection_common_space(collection)
#> [1] TRUE
```
