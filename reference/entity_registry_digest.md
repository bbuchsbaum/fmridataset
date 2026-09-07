# Compute a stable entity-registry digest

Compute a stable entity-registry digest

## Usage

``` r
entity_registry_digest(x)
```

## Arguments

- x:

  A frame, view, or entity registry.

## Value

A hexadecimal digest over the normalized registry.

## Examples

``` r
subjects <- entity_frame(
  data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
  key = "subject_id"
)
registry <- entity_registry(subject = subjects)
entity_registry_digest(registry)
#> [1] "58e6b7f88532cfa91f853422cc0c8d89808b949f2c154b9305fcb7841bba864d"
```
