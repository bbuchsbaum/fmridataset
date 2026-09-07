# Inspect a frame block plan

Inspect a frame block plan

## Usage

``` r
block_manifest(plan)
```

## Arguments

- plan:

  A `frame_block_plan`.

## Value

A data frame containing logical block bounds and byte estimates.

## Examples

``` r
frame <- fmri_frame(
  assays = list(signal = memory_source(matrix(seq_len(20), 5, 4))),
  observations = data.frame(.obs_id = sprintf("obs-%d", 1:5)),
  active_assay = "signal"
)
plan <- plan_blocks(frame, memory_budget = 10 * 1024^2)
block_manifest(plan)
#>   .block_id .observation_start .observation_end .n_observation .feature_start
#> 1         1                  1                5              5              1
#>   .feature_end .n_feature .output_bytes .peak_bytes .bytes
#> 1            4          4           160         160    160
```
