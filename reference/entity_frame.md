# Construct a keyed entity frame

An entity frame stores one row per subject, session, run, stimulus,
item, or other study entity. Scalar annotations live in `data`;
multivariate values live in named, first-axis-aligned `axis_block`
objects.

## Usage

``` r
entity_frame(data, key, blocks = list(), entity_type = NULL, metadata = list())
```

## Arguments

- data:

  Scalar entity annotations with one row per entity.

- key:

  Name of the stable primary-key column.

- blocks:

  Named entity-aligned `axis_block` objects.

- entity_type:

  Optional semantic type such as `"subject"` or `"stimulus"`.

- metadata:

  Additional serializable metadata.

## Value

An `entity_frame`, also implementing the `axis_frame` contract.

## Examples

``` r
x <- entity_frame(
  data = tibble::tibble(
    stimulus_id = c("stim-1", "stim-2", "stim-3"),
    category = c("face", "scene", "object")
  ),
  key = "stimulus_id"
)
entity_ids(x)
#> [1] "stim-1" "stim-2" "stim-3"
```
