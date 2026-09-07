# Study link and table accessors

Study link and table accessors

## Usage

``` r
study_links(x)

study_link(x, name)

study_tables(x)

study_table(x, name)

events(x, name = "events")
```

## Arguments

- x:

  An `fmri_study`.

- name:

  Stable link or table name.

## Value

A registry or one descriptor/table.

## Examples

``` r
voxels <- volume_space(dim = c(2, 2, 1), affine = diag(4), template = "toy")
frame <- fmri_frame(
  assays = list(bold = matrix(rnorm(3 * n_features(voxels)), nrow = 3)),
  observations = data.frame(.obs_id = paste0("vol-", 1:3)),
  space = voxels
)
study <- fmri_study(list(main = frame))
study_links(study)
#> list()
study_tables(study)
#> list()
```
