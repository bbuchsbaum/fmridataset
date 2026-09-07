# Inspect basis-space operators

Inspect basis-space operators

## Usage

``` r
basis_analysis(x)

basis_synthesis(x)

basis_projection_info(x)
```

## Arguments

- x:

  A `basis_space`.

## Value

`basis_analysis()` returns the parent-to-component analysis operator;
`basis_synthesis()` returns the optional component-to-parent synthesis
operator; `basis_projection_info()` returns validation metadata. These
names deliberately avoid colliding with
[`fmrilatent::basis_decoder()`](https://rdrr.io/pkg/fmrilatent/man/basis_decoder.html),
which constructs model-level decoders.

## Examples

``` r
parent <- volume_space(c(2, 1, 1), support = 1:2)
x <- basis_space(parent, c("c1", "c2"), diag(2), diag(2))
basis_analysis(x)
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
basis_synthesis(x)
#>      [,1] [,2]
#> [1,]    1    0
#> [2,]    0    1
basis_projection_info(x)
#> $left_inverse_validated
#> [1] TRUE
#> 
#> $left_inverse_error
#> [1] 0
#> 
#> $tolerance
#> [1] 1e-08
#> 
```
