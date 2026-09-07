# Entity-frame accessors

Entity-frame accessors

## Usage

``` r
entity_key(x)

entity_ids(x)

entity_data(x)

entity_blocks(x)
```

## Arguments

- x:

  An `entity_frame`.

## Value

The stable key name, entity IDs, scalar data, or aligned blocks.

## Examples

``` r
embedding <- axis_block(matrix(as.double(1:8), 2, 4), role = "embedding")
x <- entity_frame(
  data = tibble::tibble(stimulus_id = c("stim-1", "stim-2")),
  key = "stimulus_id",
  blocks = list(semantic = embedding)
)
entity_key(x)
#> [1] "stimulus_id"
entity_ids(x)
#> [1] "stim-1" "stim-2"
entity_data(x)
#> # A tibble: 2 × 1
#>   stimulus_id
#>   <chr>      
#> 1 stim-1     
#> 2 stim-2     
entity_blocks(x)
#> $semantic
#> $data
#>      [,1] [,2] [,3] [,4]
#> [1,]    1    3    5    7
#> [2,]    2    4    6    8
#> 
#> $components
#> # A tibble: 4 × 1
#>   .component_id   
#>   <chr>           
#> 1 component-000001
#> 2 component-000002
#> 3 component-000003
#> 4 component-000004
#> 
#> $role
#> [1] "embedding"
#> 
#> $units
#> NULL
#> 
#> $metadata
#> list()
#> 
#> attr(,"class")
#> [1] "axis_block"
#> 
```
