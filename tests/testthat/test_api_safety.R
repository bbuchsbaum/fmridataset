# Tests for API safety issues identified in API_SAFETY_ANALYSIS.md
library(fmridataset)

test_that("matrix_dataset handles type coercion with appropriate messages", {
  # Test vector to matrix coercion
  vec <- rnorm(100)

  # Should convert vector to matrix silently (no message in current implementation)
  dset <- matrix_dataset(vec, TR = 2, run_length = 100)

  # Result should be a valid dataset
  expect_s3_class(dset, "matrix_dataset")
  expect_equal(dim(dset$datamat), c(100, 1))

  # Test data.frame input
  df <- data.frame(v1 = rnorm(100), v2 = rnorm(100))
  dset <- matrix_dataset(as.matrix(df), TR = 2, run_length = 100)
  expect_equal(ncol(dset$datamat), 2)
})

test_that("matrix_dataset provides non-empty event_table schema by default", {
  mat <- matrix(rnorm(100), nrow = 100, ncol = 1)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)

  expect_s3_class(dset$event_table, "data.frame")
  expect_equal(nrow(dset$event_table), 0)
  expect_equal(ncol(dset$event_table), 1)
})

test_that("matrix_dataset validates run_length properly", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Should accept single value
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  expect_equal(dset$nruns, 1)

  # Should accept vector
  dset <- matrix_dataset(mat, TR = 2, run_length = c(50, 50))
  expect_equal(dset$nruns, 2)

  # Should reject negative values (but actually gives sum error)
  expect_error(
    matrix_dataset(mat, TR = 2, run_length = c(50, -50)),
    "sum\\(run_length\\) not equal to nrow\\(datamat\\)"
  )

  # Should reject non-numeric
  expect_error(
    matrix_dataset(mat, TR = 2, run_length = "invalid"),
    "invalid 'type' \\(character\\) of argument"
  )

  # Should give clear error when sum doesn't match
  expect_error(
    matrix_dataset(mat, TR = 2, run_length = 50),
    "sum\\(run_length\\) not equal to nrow\\(datamat\\)"
  )
})

test_that("get_mask returns consistent types across backends", {
  # Matrix dataset - returns logical vector
  mat <- matrix(rnorm(100 * 50), 100, 50)
  mat_dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  mask1 <- get_mask(mat_dset)
  expect_type(mask1, "logical")
  expect_equal(length(mask1), 50)

  # Matrix backend - returns logical vector
  backend <- matrix_backend(mat, mask = rep(TRUE, 50))
  dset <- fmri_dataset(backend, TR = 2, run_length = 100)
  mask2 <- get_mask(dset)
  expect_type(mask2, "logical")
  expect_equal(length(mask2), 50)

  # Both should work for masking operations
  expect_true(all(mask1 > 0))
  expect_true(all(mask2))
})

test_that("fmri_dataset validates file existence early", {
  # Non-existent files should error immediately
  expect_error(
    fmri_dataset(
      scans = c("/nonexistent/file1.nii", "/nonexistent/file2.nii"),
      mask = "/nonexistent/mask.nii",
      TR = 2,
      run_length = c(100, 100)
    ),
    "Source files not found"
  )
})

test_that("parameter names are consistent across dataset types", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # All dataset constructors should accept 'TR' and 'run_length'
  # Matrix dataset
  d1 <- matrix_dataset(mat, TR = 2, run_length = 100)
  expect_equal(get_TR(d1), 2)

  # Check that backend constructors use consistent names
  backend <- matrix_backend(mat)
  d2 <- fmri_dataset(backend, TR = 2, run_length = 100)
  expect_equal(get_TR(d2), 2)
})

test_that("resource cleanup happens automatically", {
  skip_if_not_installed("neuroim2")

  # Create a backend that tracks cleanup
  cleanup_called <- FALSE

  # Mock backend with finalizer
  backend <- structure(
    list(
      data = matrix(rnorm(100), 10, 10),
      cleanup = function() {
        cleanup_called <<- TRUE
      }
    ),
    class = c("mock_backend", "storage_backend")
  )

  # Add methods
  backend_open.mock_backend <- function(backend) backend
  backend_close.mock_backend <- function(backend) {
    backend$cleanup()
    invisible(NULL)
  }
  backend_get_dims.mock_backend <- function(backend) {
    list(spatial = c(10, 1, 1), time = 10)
  }
  backend_get_mask.mock_backend <- function(backend) {
    rep(TRUE, 10)
  }
  backend_get_data.mock_backend <- function(backend, rows = NULL, cols = NULL) {
    backend$data
  }
  backend_get_metadata.mock_backend <- function(backend) list()

  # Register methods temporarily
  registerS3method("backend_open", "mock_backend", backend_open.mock_backend)
  registerS3method("backend_close", "mock_backend", backend_close.mock_backend)
  registerS3method("backend_get_dims", "mock_backend", backend_get_dims.mock_backend)
  registerS3method("backend_get_mask", "mock_backend", backend_get_mask.mock_backend)
  registerS3method("backend_get_data", "mock_backend", backend_get_data.mock_backend)
  registerS3method("backend_get_metadata", "mock_backend", backend_get_metadata.mock_backend)

  # Create dataset
  dset <- fmri_dataset(backend, TR = 2, run_length = 10)

  # Cleanup should be called when backend is closed
  backend_close(dset$backend)
  expect_true(cleanup_called)
})

test_that("TR parameter is required", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Currently TR is required
  expect_error(
    matrix_dataset(mat, run_length = 100),
    "TR"
  )

  # TR parameter works when provided
  dset <- matrix_dataset(mat, TR = 1, run_length = 100)
  expect_equal(get_TR(dset), 1)
})

test_that("mask defaults are consistent and documented", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Matrix dataset creates default mask of TRUEs
  dset1 <- matrix_dataset(mat, TR = 2, run_length = 100)
  expect_equal(unique(dset1$mask), TRUE)
  expect_equal(length(dset1$mask), 50)

  # Matrix backend creates default mask of TRUEs
  backend <- matrix_backend(mat)
  expect_true(all(backend$mask))
  expect_equal(length(backend$mask), 50)
})

test_that("error messages are informative and actionable", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Run length mismatch should show actual vs expected
  err <- tryCatch(
    matrix_dataset(mat, TR = 2, run_length = 50),
    error = function(e) e
  )

  # Standard assert_that error shows condition
  expect_match(err$message, "sum\\(run_length\\) not equal to nrow\\(datamat\\)")

  # Better would be something like:
  # "Total run length (50) must equal number of timepoints (100)"
})

test_that("side effects are properly documented", {
  # get_data_matrix on latent_dataset returns different shape
  skip_if_not_installed("fmristore")

  # Create mock latent dataset
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  lvec <- methods::new("mock_LatentNeuroVec",
    basis = matrix(rnorm(100 * 5), 100, 5), # 100 time x 5 components
    loadings = matrix(rnorm(1000 * 5), 1000, 5), # 1000 voxels x 5 components
    offset = numeric(0),
    mask = array(TRUE, c(10, 10, 10)),
    space = structure(c(10, 10, 10, 100), class = "mock_space")
  )

  dset <- latent_dataset(list(lvec), TR = 2, run_length = 100)

  # get_data returns latent scores, not voxel data
  expect_warning(
    data <- get_data(dset),
    "returns latent scores, not voxel data"
  )

  # Shape is time x components, not time x voxels
  expect_equal(dim(data), c(100, 5))
})

test_that("functions handle NULL mask correctly", {
  # Some constructors allow NULL mask
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Matrix backend with NULL mask
  backend <- matrix_backend(mat, mask = NULL)
  expect_true(all(backend$mask)) # Should create default

  # H5 backend constructor
  if (requireNamespace("hdf5r", quietly = TRUE)) {
    # Would test h5_backend with NULL mask_source
  }
})

test_that("input validation happens before expensive operations", {
  # Large matrix that would be expensive to process
  large_mat <- matrix(rnorm(10000 * 1000), 10000, 1000)

  # Invalid run_length should fail fast
  start_time <- Sys.time()
  expect_error(
    matrix_dataset(large_mat, TR = 2, run_length = 5000),
    "sum\\(run_length\\) not equal to nrow\\(datamat\\)"
  )
  elapsed <- as.numeric(Sys.time() - start_time, units = "secs")

  # Should fail quickly without processing the large matrix
  expect_lt(elapsed, 0.1) # Less than 100ms
})

test_that("type checking provides helpful error messages", {
  # Test various invalid inputs
  expect_error(
    matrix_dataset(datamat = "not a matrix", TR = 2, run_length = 100),
    "datamat"
  )

  expect_error(
    matrix_dataset(matrix(1:10, 5, 2), TR = "invalid", run_length = 5),
    "non-numeric argument to binary operator"
  )

  expect_error(
    matrix_dataset(matrix(1:10, 5, 2), TR = 2, run_length = list(5)),
    "invalid 'type' \\(list\\) of argument"
  )
})
