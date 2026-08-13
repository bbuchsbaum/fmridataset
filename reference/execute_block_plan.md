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
