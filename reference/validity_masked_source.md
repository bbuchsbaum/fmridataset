# Lazily mask invalid observation-feature cells with missing values

Lazily mask invalid observation-feature cells with missing values

## Usage

``` r
validity_masked_source(source, observation_mask_id, bank)
```

## Arguments

- source:

  Observation-by-feature array source.

- observation_mask_id:

  One mask-bank ID per source row.

- bank:

  Compatible `mask_bank`.

## Value

A serializable `validity_masked_source`.

## Examples

``` r
space <- index_space(4, ids = paste0("f", 1:4), namespace = "validity-ex")
bank <- mask_bank(
  rbind(c(TRUE, TRUE, FALSE, TRUE), c(TRUE, FALSE, FALSE, TRUE)),
  space
)
src <- memory_source(matrix(1:8, nrow = 2))
masked <- validity_masked_source(src, bank$mask_ids[c(1, 2)], bank)
source_read(masked)
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3   NA    7
#> [2,]    2   NA   NA    8
```
