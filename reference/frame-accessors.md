# Frame accessors

Frame accessors

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
