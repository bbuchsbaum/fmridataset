# Extract canonical study representations for persistence

Filtered study views are compacted against their visible shared entities
so the persisted object is a self-contained study rather than a view
retaining references to filtered-out registry rows.

## Usage

``` r
fds_study_representations(x)
```

## Arguments

- x:

  An `fmri_study` or filtered study view.

## Value

Named frames and collections matching `fds_study_manifest(x)`.
