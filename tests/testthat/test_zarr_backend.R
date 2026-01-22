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

  # Create test data
  arr <- array(rnorm(8 * 8 * 4 * 10), dim = c(8, 8, 4, 10))

  # Write using zarr package
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

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

test_that("zarr_backend works with mock zarr data", {
  skip_if_not_installed("zarr")

  # Create mock ZarrArray R6 object
  mock_zarr <- list(
    shape = c(10, 10, 10, 100),
    dtype = "float64",
    chunks = c(5, 5, 5, 50)
  )

  # Add array indexing behavior
  mock_data <- array(rnorm(10 * 10 * 10 * 100), c(10, 10, 10, 100))

  # Mock zarr::open_zarr to return our mock object
  with_mocked_bindings(
    open_zarr = function(path) {
      # Create mock R6-like object
      obj <- mock_zarr
      # Add subsetting method
      class(obj) <- c("ZarrArray", "R6")
      obj
    },
    .package = "zarr",
    {
      backend <- zarr_backend("test.zarr")

      # Mock the backend open with data injection
      with_mocked_bindings(
        `[.ZarrArray` = function(x, ..., drop = FALSE) mock_data,
        {
          backend <- backend_open(backend)

          dims <- backend_get_dims(backend)
          expect_equal(dims$spatial, c(10, 10, 10))
          expect_equal(dims$time, 100)

          # Test metadata
          metadata <- backend_get_metadata(backend)
          expect_equal(metadata$storage_format, "zarr")
          expect_equal(metadata$zarr_version, "v3")

          backend_close(backend)
        }
      )
    }
  )
})

test_that("zarr_backend handles preload option", {
  skip_if_not_installed("zarr")

  # Create test data
  arr <- array(rnorm(2 * 3 * 2 * 2), dim = c(2, 3, 2, 2))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

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

  # Create 3D array (wrong dimensions)
  arr <- array(rnorm(2 * 2 * 2), dim = c(2, 2, 2))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

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

  # Create test data
  arr <- array(rnorm(2 * 2 * 2 * 10), dim = c(2, 2, 2, 10))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

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

  # Create larger array
  arr <- array(rnorm(8 * 8 * 8 * 20), dim = c(8, 8, 8, 20))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

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
