# Apply a function to bounded feature blocks

Apply a function to bounded feature blocks

## Usage

``` r
block_apply(x, FUN, block_size = 4096L, assay = active_assay(x), ...)
```

## Arguments

- x:

  An `fmri_frame` or view.

- FUN:

  Function receiving an observation-by-feature matrix and feature IDs.

- block_size:

  Number of features per block.

- assay:

  Assay name.

- ...:

  Additional arguments passed to `FUN`.

## Value

A list of block results.
