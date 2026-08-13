# Collect one frame assay under an explicit memory budget

Collect one frame assay under an explicit memory budget

## Usage

``` r
collect_assay(
  x,
  assay = active_assay(x),
  memory_budget = getOption("fmridataset.collect_budget", 2 * 1024^3),
  force = FALSE
)
```

## Arguments

- x:

  An `fmri_frame` or view.

- assay:

  Assay name.

- memory_budget:

  Maximum output bytes.

- force:

  Allow collection above the budget.

## Value

A dense matrix.
