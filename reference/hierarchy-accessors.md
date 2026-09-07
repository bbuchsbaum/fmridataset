# Access derived hierarchy index data

Access derived hierarchy index data

## Usage

``` r
hierarchy_ids(x)

hierarchy_groups(x)

hierarchy_levels(x)

hierarchy_relations(x)

hierarchy_complete(x)
```

## Arguments

- x:

  An `fmri_hierarchy_index`.

## Value

`hierarchy_ids()` returns stable entity IDs; `hierarchy_groups()`
returns stable integer grouping codes; `hierarchy_levels()` and
`hierarchy_relations()` return named character vectors;
`hierarchy_complete()` returns a logical vector.

## Examples

``` r
subject <- entity_frame(
  data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
  key = "subject_id"
)
observations <- tibble::tibble(
  .obs_id = paste0("obs-", 1:4),
  subject_id = c("sub-1", "sub-1", "sub-2", "sub-2")
)
frame <- fmri_frame(
  assays = list(beta = memory_source(matrix(as.double(1:12), 4, 3))),
  observations = observations,
  entities = list(subject = subject),
  relations = list(
    observation_subject = key_relation("subject_id", target = "subject")
  )
)
idx <- hierarchy_index(frame, levels = "subject")
hierarchy_ids(idx)
#> # A tibble: 4 × 2
#>   .obs_id subject
#>   <chr>   <chr>  
#> 1 obs-1   sub-1  
#> 2 obs-2   sub-1  
#> 3 obs-3   sub-2  
#> 4 obs-4   sub-2  
hierarchy_groups(idx)
#> # A tibble: 4 × 2
#>   .obs_id subject
#>   <chr>     <int>
#> 1 obs-1         1
#> 2 obs-2         1
#> 3 obs-3         2
#> 4 obs-4         2
hierarchy_levels(idx)
#> [1] "subject"
hierarchy_relations(idx)
#>               subject 
#> "observation_subject" 
hierarchy_complete(idx)
#> [1] TRUE TRUE TRUE TRUE
```
