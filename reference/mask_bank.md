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
