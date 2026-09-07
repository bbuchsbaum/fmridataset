# Execute a bounded frame block plan

Execute a bounded frame block plan

## Usage

``` r
execute_block_plan(x, plan, FUN, ..., assay = plan$assay)
```

## Arguments

- x:

  The same frame or view used to construct `plan`.

- plan:

  A `frame_block_plan`.

- FUN:

  Function receiving `values`, `observation_ids`, `feature_ids`, and the
  one-row block manifest entry.

- ...:

  Additional arguments passed to `FUN`.

- assay:

  Assay name; defaults to the planned assay.

## Value

A list containing one result per planned block.

## Examples

``` r
frame <- fmri_frame(
  assays = list(signal = memory_source(matrix(seq_len(20), 5, 4))),
  observations = data.frame(.obs_id = sprintf("obs-%d", 1:5)),
  active_assay = "signal"
)
plan <- plan_blocks(frame, memory_budget = 10 * 1024^2)
execute_block_plan(frame, plan, function(values, observation_ids, feature_ids, block) {
  sum(values)
})
#> [[1]]
#> [1] 210
#> 
```
