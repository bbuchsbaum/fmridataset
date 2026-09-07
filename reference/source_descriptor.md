# Inspect and validate an array-source contract

A valid canonical source is two-dimensional, has an explicit supported
dtype and chunk grid, advertises serializable block slicing, provides a
stable non-empty fingerprint, and contains no runtime handles or
closures.

## Usage

``` r
source_descriptor(x)

validate_array_source(x)
```

## Arguments

- x:

  An `array_source` descriptor.

## Value

`source_descriptor()` returns a plain serializable contract list.
`validate_array_source()` invisibly returns `x` or raises a structured
source-contract error.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
source_descriptor(src)
#> $shape
#> [1] 2 3
#> 
#> $dtype
#> [1] "float64"
#> 
#> $chunks
#> [1] 2 3
#> 
#> $capabilities
#> [1] "row_slice"          "column_slice"       "block_slice"       
#> [4] "serializable"       "pushdown:all"       "pushdown:range"    
#> [7] "pushdown:positions"
#> 
#> $fingerprint
#> [1] "30947908b93776b583740eade4957b7809f9e3cc651bccead684b21754af7b10"
#> 
validate_array_source(src)
```
