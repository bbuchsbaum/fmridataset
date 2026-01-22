# Convert Dataset Objects to DelayedArray

Provides DelayedArray interface for dataset objects. These methods
convert fmri_dataset and matrix_dataset objects to DelayedArrays for
memory-efficient operations.

## Usage

``` r
# S4 method for class 'matrix_dataset'
as_delayed_array(backend, sparse_ok = FALSE)

# S4 method for class 'fmri_file_dataset'
as_delayed_array(backend, sparse_ok = FALSE)

# S4 method for class 'fmri_mem_dataset'
as_delayed_array(backend, sparse_ok = FALSE)
```

## Arguments

- backend:

  A storage backend object

- sparse_ok:

  Logical, allow sparse representation when possible
