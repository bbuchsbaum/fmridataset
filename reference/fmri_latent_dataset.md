# Create an fMRI Dataset Object from LatentNeuroVec Files or Objects

**\[deprecated\]**

This function is deprecated. Please use
[`latent_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md)
instead, which provides a proper interface for latent space data.

## Usage

``` r
fmri_latent_dataset(
  latent_files,
  mask_source = NULL,
  TR,
  run_length,
  event_table = data.frame(),
  base_path = ".",
  censor = NULL,
  preload = FALSE
)
```

## Arguments

- latent_files:

  Source files or objects

- mask_source:

  Ignored

- TR:

  The repetition time in seconds

- run_length:

  Vector of run lengths

- event_table:

  Event table

- base_path:

  Base path for files

- censor:

  Censor vector

- preload:

  Whether to preload data

## Value

A latent_dataset object

## See also

[`latent_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/latent_dataset.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Use latent_dataset() instead:
dset <- latent_dataset(
  source = c("run1.lv.h5", "run2.lv.h5", "run3.lv.h5"),
  TR = 2,
  run_length = c(150, 150, 150)
)
} # }
```
