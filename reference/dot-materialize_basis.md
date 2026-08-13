# Materialize a basis slot to a dense matrix

Handles concrete matrices, sparse Matrix objects, and fmrilatent
BasisHandle objects. Returns a standard matrix.

## Usage

``` r
.materialize_basis(obj)
```

## Arguments

- obj:

  A LatentNeuroVec object

## Value

A matrix (time x components)
