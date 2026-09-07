# Bind frames along observations

Bind frames along observations

## Usage

``` r
bind_observations(
  ...,
  metadata_policy = c("identical", "merge"),
  active_assay = NULL
)
```

## Arguments

- ...:

  Frames with identical feature IDs, spaces, and assay semantics.

- metadata_policy:

  Frame-metadata reconciliation. `"identical"` requires exact equality;
  `"merge"` recursively combines non-conflicting unaligned records.

- active_assay:

  Optional active assay for the result. Required when operands have
  different active assays.

## Value

A lazily row-bound `fmri_frame`.

## Examples

``` r
sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
make <- function(prefix) {
  fmri_frame(
    assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
    observations = data.frame(.obs_id = sprintf("%s-%d", prefix, 1:4)),
    space = sp
  )
}
bound <- bind_observations(make("a"), make("b"))
dim(bound)
#> [1] 8 8
```
