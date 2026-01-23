test_that("zarr_backend validates inputs correctly", {
  # Test invalid source
  expect_error(
    zarr_backend(NULL),
    "source must be a single character string"
  )

  expect_error(
    zarr_backend(c("path1", "path2")),
    "source must be a single character string"
  )

  # Test that constructor returns correct class (skip if zarr not installed)
  skip_if_not_installed("zarr")
  backend <- zarr_backend("dummy.zarr")
  expect_s3_class(backend, "zarr_backend")
  expect_s3_class(backend, "storage_backend")
})

test_that("zarr_backend requires zarr package", {
  # Mock the requireNamespace function to simulate missing package
  with_mocked_bindings(
    requireNamespace = function(...) FALSE,
    .package = "base",
    {
      expect_error(
        zarr_backend("dummy.zarr"),
        "The zarr package is required"
      )
    }
  )
})

test_that("zarr_backend handles missing files gracefully", {
  skip_if_not_installed("zarr")

  backend <- zarr_backend("/nonexistent/path.zarr")
  expect_error(
    backend_open(backend),
    "Zarr store not found"
  )
})

test_that("zarr_backend works with local store", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Create test data
  arr <- array(rnorm(8 * 8 * 4 * 10), dim = c(8, 8, 4, 10))

  # Write using zarr package
  tmp_dir <- tempfile()
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr, location = tmp_dir)

  # Test backend
  backend <- zarr_backend(tmp_dir)
  backend <- backend_open(backend)

  # Test dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(8, 8, 4))
  expect_equal(dims$time, 10)

  # Test mask (should be all TRUE for zarr backend)
  mask <- backend_get_mask(backend)
  expect_equal(length(mask), 8 * 8 * 4)
  expect_true(all(mask))

  # Test data read
  data <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(data), c(5, 10))

  # Test close
  expect_silent(backend_close(backend))
})

test_that("zarr_backend handles preload option", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Create test data
  arr <- array(rnorm(2 * 3 * 2 * 2), dim = c(2, 3, 2, 2))

  tmp_dir <- tempfile()
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr, location = tmp_dir)

  # Test with preload = TRUE
  backend <- zarr_backend(tmp_dir, preload = TRUE)
  expect_message(
    backend <- backend_open(backend),
    "Preloading Zarr data"
  )

  # Data should be cached
  expect_false(is.null(backend$data_cache))

  # Multiple data accesses should use cache
  data1 <- backend_get_data(backend)
  data2 <- backend_get_data(backend)
  expect_equal(data1, data2)
})

test_that("zarr_backend validates array dimensions", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Create 3D array (wrong dimensions)
  arr <- array(rnorm(2 * 2 * 2), dim = c(2, 2, 2))

  tmp_dir <- tempfile()
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr, location = tmp_dir)

  backend <- zarr_backend(tmp_dir)
  expect_error(
    backend_open(backend),
    "Expected 4D array, got 3D"
  )
})

test_that("zarr_backend handles remote URLs", {
  skip_if_not_installed("zarr")

  # Test S3 URL (just constructor, don't try to open)
  backend <- zarr_backend("s3://bucket/path/data.zarr")
  expect_equal(backend$source, "s3://bucket/path/data.zarr")

  # Test HTTPS URL
  backend <- zarr_backend("https://example.com/data.zarr")
  expect_equal(backend$source, "https://example.com/data.zarr")
})

test_that("zarr_backend integrates with fmri_dataset", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Create test data
  arr <- array(rnorm(2 * 2 * 2 * 10), dim = c(2, 2, 2, 10))

  tmp_dir <- tempfile()
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr, location = tmp_dir)

  # Create backend
  backend <- zarr_backend(tmp_dir)

  # Create dataset
  dataset <- fmri_dataset(
    backend,
    TR = 2,
    run_length = 10
  )

  expect_s3_class(dataset, "fmri_dataset")
  expect_equal(n_timepoints(dataset), 10)

  # Test data access
  data_mat <- get_data_matrix(dataset)
  expect_equal(dim(data_mat), c(10, 8))
})

test_that("zarr_backend chooses reading strategy", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Create larger array
  arr <- array(rnorm(8 * 8 * 8 * 20), dim = c(8, 8, 8, 20))

  tmp_dir <- tempfile()
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr, location = tmp_dir)

  backend <- zarr_backend(tmp_dir)
  backend <- backend_open(backend)

  # Test small subset
  small_data <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(small_data), c(5, 10))

  # Test large subset (>50%)
  n_voxels <- prod(backend$dims$spatial)
  large_cols <- seq_len(round(n_voxels * 0.6))
  large_data <- backend_get_data(backend, rows = 1:15, cols = large_cols)
  expect_equal(dim(large_data), c(15, length(large_cols)))

  backend_close(backend)
})

test_that("zarr_backend lifecycle works correctly", {
  skip_if_not_installed("zarr")

  # Use helper to create test data
  tmp_dir <- create_test_zarr(dims = c(4, 4, 4, 10))
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  backend <- zarr_backend(tmp_dir)

  # Initially not open
  expect_false(backend$is_open)
  expect_null(backend$zarr_array)

  # Open backend
  backend <- backend_open(backend)
  expect_true(backend$is_open)
  expect_false(is.null(backend$zarr_array))

  # Opening again should be idempotent
  backend2 <- backend_open(backend)
  expect_equal(backend2$is_open, backend$is_open)

  # Close backend (note: due to R's copy-on-write semantics,
  # backend_close doesn't actually modify the backend object in the caller's scope)
  expect_silent(backend_close(backend))

  # Backend state remains unchanged in caller's scope
  # This is consistent with other backend implementations which also don't modify state
  expect_true(backend$is_open)
})

test_that("zarr_backend get_data validates inputs", {
  skip_if_not_installed("zarr")

  tmp_dir <- create_test_zarr(dims = c(4, 4, 4, 10))
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  backend <- zarr_backend(tmp_dir)

  # Error on closed backend
  expect_error(
    backend_get_data(backend),
    "Backend must be opened"
  )

  # Open for further tests
  backend <- backend_open(backend)

  n_time <- backend$dims$time
  n_voxels <- prod(backend$dims$spatial)

  # Test invalid row indices
  expect_error(
    backend_get_data(backend, rows = 0:5),
    "Row indices must be between"
  )

  expect_error(
    backend_get_data(backend, rows = 1:(n_time + 5)),
    "Row indices must be between"
  )

  # Test invalid column indices
  expect_error(
    backend_get_data(backend, cols = 0:5),
    "Column indices must be between"
  )

  expect_error(
    backend_get_data(backend, cols = 1:(n_voxels + 5)),
    "Column indices must be between"
  )

  backend_close(backend)
})

test_that("zarr_backend get_data returns correct subsets", {
  skip_if_not_installed("zarr")

  tmp_dir <- create_test_zarr(dims = c(4, 4, 4, 10))
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  backend <- zarr_backend(tmp_dir)
  backend <- backend_open(backend)

  n_time <- backend$dims$time
  n_voxels <- prod(backend$dims$spatial)

  # Full data retrieval (NULL rows and cols)
  full_data <- backend_get_data(backend, rows = NULL, cols = NULL)
  expect_equal(dim(full_data), c(n_time, n_voxels))

  # Subset rows only
  row_subset <- backend_get_data(backend, rows = 1:5, cols = NULL)
  expect_equal(dim(row_subset), c(5, n_voxels))

  # Subset cols only
  col_subset <- backend_get_data(backend, rows = NULL, cols = 1:10)
  expect_equal(dim(col_subset), c(n_time, 10))

  # Subset both
  both_subset <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(both_subset), c(5, 10))

  backend_close(backend)
})

test_that("zarr_backend get_metadata returns format info", {
  skip_if_not_installed("zarr")

  tmp_dir <- create_test_zarr(dims = c(4, 4, 4, 10))
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  backend <- zarr_backend(tmp_dir)

  # Error on closed backend
  expect_error(
    backend_get_metadata(backend),
    "Backend must be opened"
  )

  # Open backend
  backend <- backend_open(backend)

  # Get metadata
  metadata <- backend_get_metadata(backend)

  # Check required fields
  expect_type(metadata, "list")
  expect_equal(metadata$storage_format, "zarr")
  expect_equal(metadata$zarr_version, "v3")

  # Optional fields may be present
  # dtype and chunk_shape are extracted if available from zarr_array R6 object

  backend_close(backend)
})

test_that("zarr_backend get_mask validates correctly", {
  skip_if_not_installed("zarr")

  tmp_dir <- create_test_zarr(dims = c(4, 4, 4, 10))
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  backend <- zarr_backend(tmp_dir)
  backend <- backend_open(backend)

  mask <- backend_get_mask(backend)

  # Check mask properties
  expect_type(mask, "logical")
  expect_equal(length(mask), prod(backend$dims$spatial))

  # Zarr backend returns all TRUE (no separate mask array)
  expect_true(all(mask))

  # No NA values
  expect_false(any(is.na(mask)))

  backend_close(backend)
})

test_that("zarr_backend preload parameter stored correctly", {
  skip_if_not_installed("zarr")

  # Test preload = FALSE (default)
  backend1 <- zarr_backend("dummy.zarr")
  expect_false(backend1$preload)

  # Test preload = TRUE
  backend2 <- zarr_backend("dummy.zarr", preload = TRUE)
  expect_true(backend2$preload)
})

test_that("zarr_backend validates wrong type inputs", {
  # Test numeric source
  expect_error(
    zarr_backend(123),
    "source must be a single character string"
  )

  # Test logical source
  expect_error(
    zarr_backend(TRUE),
    "source must be a single character string"
  )

  # Test list source
  expect_error(
    zarr_backend(list("path")),
    "source must be a single character string"
  )
})
