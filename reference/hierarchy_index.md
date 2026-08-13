# Derive stable observation hierarchy indices

A hierarchy index is an immutable, assay-free cache derived from
validated key relations. `levels` are supplied root-to-leaf. Each
adjacent edge must form a strict chain from observations to the deepest
entity and then through its parents. Crossed relations are therefore
never mistaken for containment.

## Usage

``` r
hierarchy_index(x, levels, relations = NULL)
```

## Arguments

- x:

  An `fmri_frame` or `fmri_view`.

- levels:

  Unique entity names in root-to-leaf order.

- relations:

  Optional named character vector mapping every level to the
  key-relation name used for its incoming edge. This is required when an
  edge is ambiguous.

## Value

An `fmri_hierarchy_index`.

## Details

Integer group codes use entity-registry order, so they remain stable
when a frame is filtered or reordered.
