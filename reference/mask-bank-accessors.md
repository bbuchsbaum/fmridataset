# Validate and inspect a mask bank

Validate and inspect a mask bank

## Usage

``` r
validate_mask_bank(x)

n_masks(x)

mask_values(x, mask = NULL)

mask_bank_digest(x)
```

## Arguments

- x:

  A `mask_bank` or validity descriptor.

- mask:

  Optional mask ID or integer position.

## Value

The validated bank, number of masks, unpacked logical masks, or
deterministic digest.
