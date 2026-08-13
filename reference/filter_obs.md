# Filter frame observations using scalar metadata

Filter frame observations using scalar metadata

## Usage

``` r
filter_obs(x, predicate, resolve = TRUE)
```

## Arguments

- x:

  An `fmri_frame` or view.

- predicate:

  A metadata expression returning one logical value per observation.

- resolve:

  Whether the predicate may use namespaced entity metadata.

## Value

An `fmri_view`.
