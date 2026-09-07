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

## Examples

``` r
space <- index_space(6, ids = paste0("f", 1:6), namespace = "validity-ex")
masks <- rbind(
  c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE),
  c(TRUE, FALSE, FALSE, TRUE, TRUE, TRUE)
)
bank <- mask_bank(masks, space)
mask_values(bank)
#>      [,1]  [,2]  [,3] [,4]  [,5] [,6]
#> [1,] TRUE  TRUE FALSE TRUE FALSE TRUE
#> [2,] TRUE FALSE FALSE TRUE  TRUE TRUE
mask_bank_digest(bank)
#> [1] "657cfa8494f3165748adb65b0523701090b05bd3341bf86444b5b8dfb85933dc"
```
