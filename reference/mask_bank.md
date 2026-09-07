# Construct a deduplicated, bit-packed bank of feature masks

Construct a deduplicated, bit-packed bank of feature masks

## Usage

``` r
mask_bank(masks, space, metadata = list())
```

## Arguments

- masks:

  Logical mask-by-feature matrix. Duplicate rows are stored once.

- space:

  Exact feature space addressed by mask columns.

- metadata:

  Serializable metadata.

## Value

A serializable `mask_bank`.

## Examples

``` r
space <- index_space(6, ids = paste0("f", 1:6), namespace = "validity-ex")
masks <- rbind(
  c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE),
  c(TRUE, FALSE, FALSE, TRUE, TRUE, TRUE)
)
bank <- mask_bank(masks, space)
n_masks(bank)
#> [1] 2
```
