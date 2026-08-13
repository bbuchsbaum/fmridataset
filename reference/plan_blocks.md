# Plan bounded observation-by-feature blocks

`plan_blocks()` uses source chunk hints and an explicit byte budget to
build a serializable, metadata-only execution plan. `"imagewise"` blocks
retain complete feature rows, `"featurewise"` blocks retain complete
observation columns, and `"balanced"` blocks scale both axes in the
source chunk ratio.

## Usage

``` r
plan_blocks(
  x,
  assay = active_assay(x),
  layout = c("balanced", "imagewise", "featurewise"),
  memory_budget = getOption("fmridataset.block_budget", 512 * 1024^2),
  target_block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
)
```

## Arguments

- x:

  An `fmri_frame` or lazy view.

- assay:

  Assay name.

- layout:

  One of `"balanced"`, `"imagewise"`, or `"featurewise"`.

- memory_budget:

  Hard maximum bytes for one input block.

- target_block_bytes:

  Preferred block size, capped by `memory_budget`.

## Value

A serializable `frame_block_plan`.
