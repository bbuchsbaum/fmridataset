# Construct an annotated axis

Construct an annotated axis

## Usage

``` r
axis_frame(
  data,
  blocks = list(),
  id = NULL,
  axis = c("observation", "feature", "entity", "component"),
  id_col = NULL,
  metadata = list(),
  id_policy = c("require", "deterministic", "ephemeral"),
  id_keys = NULL,
  id_namespace = NULL
)

axis_data(x)

axis_blocks(x)

axis_ids(x)
```

## Arguments

- data:

  A data frame with one row per axis element.

- blocks:

  Named `axis_block` objects aligned on their first dimension.

- id:

  Optional stable IDs.

- axis:

  Axis role. Observation is the public default.

- id_col:

  Name of the ID column.

- metadata:

  Additional serializable metadata.

- id_policy:

  ID policy. `"require"` accepts only supplied durable IDs;
  `"deterministic"` derives durable IDs from `id_keys` and
  `id_namespace`; `"ephemeral"` creates visibly marked session-only IDs
  that cannot be persisted or used for certified semantic identity.

- id_keys:

  Columns that uniquely identify rows under deterministic policy.

- id_namespace:

  Stable namespace under deterministic policy.

- x:

  An `axis_frame`.

## Value

An `axis_frame`.

## Examples

``` r
x <- axis_frame(data.frame(value = 1:3), id_policy = "ephemeral")
axis_ids(x)
#> [1] "ephemeral-obs-9238f0a6-5f20-4dfa-8479-18a09bc50370"
#> [2] "ephemeral-obs-b49a3774-2ed1-4543-ba57-abbb3002c8fc"
#> [3] "ephemeral-obs-1586c7c5-82ca-4dff-9490-bccb110be4d0"
axis_data(x)
#> # A tibble: 3 × 2
#>   .obs_id                                            value
#>   <chr>                                              <int>
#> 1 ephemeral-obs-9238f0a6-5f20-4dfa-8479-18a09bc50370     1
#> 2 ephemeral-obs-b49a3774-2ed1-4543-ba57-abbb3002c8fc     2
#> 3 ephemeral-obs-1586c7c5-82ca-4dff-9490-bccb110be4d0     3
```
