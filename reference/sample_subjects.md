# Sample subjects from an fmri_group

Sample subjects from an fmri_group

## Usage

``` r
sample_subjects(gd, n, replace = FALSE, strata = NULL)
```

## Arguments

- gd:

  An `fmri_group`.

- n:

  Number of subjects to sample. When `strata` is supplied and `n` has
  length 1, the same number is drawn from each stratum. Provide a named
  vector to request different counts per stratum.

- replace:

  Logical indicating whether to sample with replacement.

- strata:

  Optional column name used to stratify the sampling.

## Value

A sampled `fmri_group`.
