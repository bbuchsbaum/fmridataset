# Materialize a loadings slot to a matrix (possibly sparse)

Handles concrete matrices, sparse Matrix objects, and fmrilatent
LoadingsHandle objects. Returns a matrix or sparse Matrix.

## Usage

``` r
.materialize_loadings(obj)
```

## Arguments

- obj:

  A LatentNeuroVec object

## Value

A matrix or Matrix (voxels x components)
