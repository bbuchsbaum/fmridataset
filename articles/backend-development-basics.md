# Writing Custom Storage Backends

Write a custom backend when your data lives in a format that no built-in
backend supports — flat CSV, a proprietary binary format, a remote
database, or anything else. A backend is a plain S3 list with six
methods; once those methods exist, your format gains full access to
chunking, study-level operations, and every other fmridataset feature.
This vignette walks through the contract, a complete working example,
and how to register and test your backend.

## The six-method contract

Every backend must implement exactly these six S3 generics, declared in
`R/storage_backend.R`.

### `backend_open(backend)`

``` r

backend_open.my_backend <- function(backend) {
  ...
}
```

Acquire resources (file handles, connections). Must be idempotent: if
the backend is already open, return it unchanged.

### `backend_close(backend)`

``` r

backend_close.my_backend <- function(backend) {
  ...
}
```

Release all resources and set `is_open <- FALSE`. Must be idempotent and
return `invisible(backend)`.

### `backend_get_dims(backend)`

``` r

backend_get_dims.my_backend <- function(backend) {
  ...
}
```

Return `list(spatial = c(nx, ny, nz), time = n_timepoints)` — a named
list with a length-3 numeric `spatial` vector and a single positive
integer `time`.

### `backend_get_mask(backend)`

``` r

backend_get_mask.my_backend <- function(backend) {
  ...
}
```

Return a logical vector of length `prod(spatial)` with no `NA` values
and at least one `TRUE`.

### `backend_get_data(backend, rows = NULL, cols = NULL)`

``` r

backend_get_data.my_backend <- function(backend, rows = NULL, cols = NULL) {
  ...
}
```

Return a matrix in **timepoints x voxels** orientation. `rows` and
`cols` are 1-based integer vectors; `NULL` means “all”. Always use
`drop = FALSE` when subsetting.

### `backend_get_metadata(backend)`

``` r

backend_get_metadata.my_backend <- function(backend) {
  ...
}
```

Return a list — empty is fine. Include an `affine` matrix and
`voxel_dims` when your format has them.

## Minimal working example: CSV backend

The following ~40-line implementation is a complete, functional backend
for CSV files where rows are timepoints and columns are voxels.

``` r

# Constructor
csv_backend <- function(data_file) {
  if (!file.exists(data_file)) stop("File not found: ", data_file)
  backend <- list(
    data_file = data_file,
    data_cache = NULL,
    is_open = FALSE
  )
  class(backend) <- c("csv_backend", "storage_backend")
  backend
}

# Open: read the CSV once and cache it
backend_open.csv_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)
  }
  backend$data_cache <- as.matrix(read.csv(backend$data_file, check.names = FALSE))
  backend$is_open <- TRUE
  backend
}

# Close: drop the cache
backend_close.csv_backend <- function(backend) {
  backend$data_cache <- NULL
  backend$is_open <- FALSE
  invisible(backend)
}

# Dims: rows = timepoints, cols = voxels; report as flat spatial volume
backend_get_dims.csv_backend <- function(backend) {
  stopifnot(backend$is_open)
  list(
    spatial = c(ncol(backend$data_cache), 1L, 1L),
    time = nrow(backend$data_cache)
  )
}

# Mask: all voxels valid
backend_get_mask.csv_backend <- function(backend) {
  stopifnot(backend$is_open)
  rep(TRUE, ncol(backend$data_cache))
}

# Data: return timepoints x voxels, with optional subsetting
backend_get_data.csv_backend <- function(backend, rows = NULL, cols = NULL) {
  stopifnot(backend$is_open)
  d <- backend$data_cache
  if (!is.null(rows)) d <- d[rows, , drop = FALSE]
  if (!is.null(cols)) d <- d[, cols, drop = FALSE]
  d
}

# Metadata: minimal
backend_get_metadata.csv_backend <- function(backend) {
  stopifnot(backend$is_open)
  list(source_file = backend$data_file)
}
```

Open, query, and close the backend by calling the generics directly:

``` r

b <- csv_backend("my_data.csv")
b <- backend_open(b)

dims <- backend_get_dims(b) # spatial 500 x 1 x 1, time 200
mask <- backend_get_mask(b) # logical vector, length 500
d <- backend_get_data(b, rows = 1:20, cols = 1:50) # 20 x 50 matrix

b <- backend_close(b)
```

## Registering your backend

[`register_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/register_backend.md)
adds your factory function to the package-level registry. After
registration,
[`create_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/create_backend.md)
constructs instances by name.

``` r

register_backend(
  name        = "csv",
  factory     = csv_backend,
  description = "CSV file backend (rows = timepoints, cols = voxels)"
)
```

Create an instance through the registry:

``` r

b <- create_backend("csv", data_file = "my_data.csv")
```

[`create_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/create_backend.md)
calls
[`validate_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_backend.md)
automatically unless you pass `validate = FALSE`. To see everything
currently registered:

``` r

list_backend_names() # character vector of registered names
get_backend_registry() # full registration details for all backends
```

## Validation

[`validate_backend()`](https://bbuchsbaum.github.io/fmridataset/reference/validate_backend.md)
checks the standard contract: class inheritance, method presence, dims
structure, and mask constraints. Call it on an open backend instance.

``` r

b <- backend_open(csv_backend("my_data.csv"))
validate_backend(b) # returns TRUE or throws a descriptive error
backend_close(b)
```

The validator calls
[`backend_get_dims()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_get_dims.md)
and
[`backend_get_mask()`](https://bbuchsbaum.github.io/fmridataset/reference/backend_get_mask.md)
directly, so the backend must be open when you invoke it.

## Testing

Use `testthat` to verify all six methods and the subsetting contract.
The backend should be open for the data-access tests.

``` r

library(testthat)

# Write a small fixture
tmp <- tempfile(fileext = ".csv")
write.csv(matrix(seq_len(200), nrow = 20, ncol = 10), tmp, row.names = FALSE)

test_that("csv_backend satisfies the storage contract", {
  b <- backend_open(csv_backend(tmp))
  on.exit(backend_close(b))

  dims <- backend_get_dims(b)
  expect_equal(dims$spatial, c(10, 1L, 1L))
  expect_equal(dims$time, 20L)

  mask <- backend_get_mask(b)
  expect_true(is.logical(mask))
  expect_equal(length(mask), prod(dims$spatial))
  expect_false(anyNA(mask))

  d_full <- backend_get_data(b)
  expect_equal(dim(d_full), c(20L, 10L))

  d_sub <- backend_get_data(b, rows = 1:5, cols = 1:3)
  expect_equal(dim(d_sub), c(5L, 3L))

  expect_true(validate_backend(b))
})

unlink(tmp)
```
