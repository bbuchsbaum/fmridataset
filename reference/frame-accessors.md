# Frame accessors

Generic accessors for the components of an `fmri_frame` or `fmri_view`:
its assays, active assay, observation and feature axes and IDs, block
registries, and dimensions.

## Usage

``` r
assays(x, ...)

assay(x, name = active_assay(x), ...)

active_assay(x, ...)

observation_axis(x, ...)

observations(x, resolve = FALSE, ...)

features(x, ...)

observation_ids(x, ...)

obs_blocks(x, resolve = FALSE, ...)

feature_blocks(x, ...)

# S3 method for class 'fmri_frame'
dim(x)

nrow.fmri_frame(x)

ncol.fmri_frame(x)

# S3 method for class 'fmri_view'
dim(x)

nrow.fmri_view(x)

ncol.fmri_view(x)
```

## Arguments

- x:

  An `fmri_frame` or `fmri_view`.

- ...:

  Additional method arguments.

- name:

  Assay name.

- resolve:

  Whether to append reachable, namespaced entity annotations or lazily
  lifted entity blocks.

## Value

`assays()` returns a named `aligned_assay_set` list and `assay()` one
`aligned_assay` from it. `active_assay()` returns a single assay name.
`observation_axis()` returns an `axis_frame`; `observations()` and
`features()` return a data frame of the corresponding metadata.
`observation_ids()` and
[`feature_ids()`](https://bbuchsbaum.github.io/fmridataset/reference/feature-space.md)
return character vectors of stable IDs. `obs_blocks()` and
`feature_blocks()` return named lists of `axis_block`s.
[`dim()`](https://rdrr.io/r/base/dim.html) returns a length-2 integer
vector, and [`nrow()`](https://rdrr.io/r/base/nrow.html) and
[`ncol()`](https://rdrr.io/r/base/nrow.html) return single integers.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
  observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
  space = sp
)
names(assays(frame))
#> [1] "bold"
active_assay(frame)
#> [1] "bold"
dim(frame)
#> [1] 4 8
```
