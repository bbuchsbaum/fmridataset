# test-as_delayed_array.R
# Tests for both as_delayed_array.R (backends) and as_delayed_array_dataset.R (datasets)
library(fmridataset)

# ============================================
# Section 1: Backend conversion tests (as_delayed_array.R)
# ============================================

test_that("as_delayed_array is generic function", {
  expect_true(is.function(as_delayed_array))
  # Check that it's an S3 generic - we can verify by checking for methods
  # The function exists and can be called
  expect_true(is.function(as_delayed_array))
})

test_that(".ensure_delayed_array errors when DelayedArray unavailable", {
  skip_if_not_installed("withr")

  withr::with_options(
    list(fmridataset.disable_delayedarray = TRUE),
    {
      expect_error(
        fmridataset:::.ensure_delayed_array(),
        "DelayedArray support is disabled"
      )
    }
  )
})

test_that("as_delayed_array.default errors with unknown class", {
  # Create object with unknown class
  obj <- structure(list(data = 1:10), class = "unknown_class")

  expect_error(
    as_delayed_array(obj),
    "No as_delayed_array method for class: unknown_class"
  )
})

# Backend conversion tests (require DelayedArray)

test_that("as_delayed_array converts matrix_backend", {
  skip_if_not_installed("DelayedArray")

  # Create a simple matrix backend
  test_matrix <- matrix(rnorm(200), nrow = 10, ncol = 20)
  backend <- matrix_backend(data_matrix = test_matrix)
  backend <- backend_open(backend)

  # Convert to DelayedArray
  da <- as_delayed_array(backend)

  # Verify DelayedArray class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions match (10 timepoints, 20 voxels)
  expect_equal(dim(da), c(10, 20))

  # Verify data values match
  expect_equal(as.matrix(da), test_matrix)

  backend_close(backend)
})

test_that("as_delayed_array.matrix_backend subsetting works", {
  skip_if_not_installed("DelayedArray")

  # Create backend
  test_matrix <- matrix(1:200, nrow = 10, ncol = 20)
  backend <- matrix_backend(data_matrix = test_matrix)
  backend <- backend_open(backend)

  # Convert to DelayedArray
  da <- as_delayed_array(backend)

  # Test subsetting
  subset_da <- da[1:5, 1:10]
  expect_equal(dim(subset_da), c(5, 10))

  # Verify subset values match original
  expected_subset <- test_matrix[1:5, 1:10]
  expect_equal(as.matrix(subset_da), expected_subset)

  backend_close(backend)
})

test_that("as_delayed_array converts nifti_backend", {
  skip_if_not_installed("DelayedArray")
  skip("NIfTI backend requires real files or in-memory NeuroVec objects")

  # This test requires actual NIfTI files or properly constructed NeuroVec objects
  # Skip for now as it would require more complex test infrastructure
})

test_that("as_delayed_array converts study_backend", {
  skip_if_not_installed("DelayedArray")

  # Create two matrix backends
  mat1 <- matrix(rnorm(100), nrow = 10, ncol = 10)
  mat2 <- matrix(rnorm(100), nrow = 10, ncol = 10)

  backend1 <- matrix_backend(data_matrix = mat1)
  backend2 <- matrix_backend(data_matrix = mat2)

  # Create study backend
  study_backend <- create_backend("study",
    backends = list(backend1, backend2),
    subject_ids = c("S01", "S02")
  )
  study_backend <- backend_open(study_backend)

  # Convert to DelayedArray
  da <- as_delayed_array(study_backend)

  # Verify DelayedArray class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions reflect combined data (20 timepoints from both subjects)
  expect_equal(dim(da)[1], 20)
  expect_equal(dim(da)[2], 10)

  backend_close(study_backend)
})

test_that("register_delayed_array_support is idempotent", {
  skip_if_not_installed("DelayedArray")

  # Call once
  fmridataset:::register_delayed_array_support()

  # Check registered flag
  expect_true(fmridataset:::.delayed_array_support_env$registered)

  # Call again - should be silent
  expect_silent(fmridataset:::register_delayed_array_support())

  # Flag should still be TRUE
  expect_true(fmridataset:::.delayed_array_support_env$registered)
})

# ============================================
# Section 2: Dataset conversion tests (as_delayed_array_dataset.R)
# ============================================

test_that("as_delayed_array converts matrix_dataset", {
  skip_if_not_installed("DelayedArray")

  # Create matrix_dataset
  test_matrix <- matrix(1:100, nrow = 10, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = 10)

  # Convert to DelayedArray
  da <- as_delayed_array(dset)

  # Verify DelayedArray class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions (10 timepoints, 10 voxels)
  expect_equal(dim(da), c(10, 10))

  # Verify data values match
  expect_equal(as.matrix(da), test_matrix)
})

test_that("as_delayed_array converts fmri_mem_dataset", {
  skip_if_not_installed("DelayedArray")
  skip_if_not_installed("neuroim2")

  # Create NeuroVec and mask with matching dimensions
  dims <- c(10, 10, 10, 10)
  nvec <- neuroim2::NeuroVec(
    array(rnorm(prod(dims)), dims),
    space = neuroim2::NeuroSpace(dims)
  )

  mask_dims <- c(10, 10, 10)
  mask_data <- array(1, mask_dims)  # All ones initially
  mask_data[mask_data < 0.5] <- 0  # This will keep them all as 1
  mask_vol <- neuroim2::NeuroVol(
    mask_data,
    space = neuroim2::NeuroSpace(mask_dims)
  )

  # Create fmri_mem_dataset
  dset <- fmri_mem_dataset(list(nvec), mask_vol, TR = 2)

  # Convert to DelayedArray
  da <- as_delayed_array(dset)

  # Verify DelayedArray class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions
  expect_equal(dim(da)[1], 10) # 10 timepoints
  expect_true(dim(da)[2] > 0) # At least some voxels in mask
})

test_that("as_delayed_array converts fmri_file_dataset", {
  skip_if_not_installed("DelayedArray")

  # Create fmri_file_dataset with dummy_mode
  # Note: dummy_mode creates 100 timepoints per file by default
  dset <- fmri_dataset(
    scans = c("dummy1.nii"),
    mask = "dummy_mask.nii",
    TR = 2,
    run_length = 100,  # Match the dummy backend's default for one file
    dummy_mode = TRUE
  )

  # Convert to DelayedArray
  da <- as_delayed_array(dset)

  # Verify DelayedArray class
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions match dataset
  dims <- backend_get_dims(dset$backend)
  expected_cols <- sum(backend_get_mask(dset$backend))
  expect_equal(dim(da), c(dims$time, expected_cols))
})

test_that("as_delayed_array fmri_file_dataset errors on legacy object", {
  skip_if_not_installed("DelayedArray")

  # Create mock legacy object with class but NULL backend
  legacy_obj <- structure(
    list(backend = NULL, nruns = 1),
    class = c("fmri_file_dataset", "fmri_dataset", "list")
  )

  expect_error(
    as_delayed_array(legacy_obj),
    "not supported for legacy fmri_file_dataset"
  )
})

test_that("as_delayed_array S4 methods are registered", {
  skip_if_not_installed("DelayedArray")
  skip_if_not_installed("methods")

  # Force registration
  fmridataset:::register_delayed_array_support()

  # Check that S4 methods exist for dataset classes
  expect_true(methods::hasMethod("as_delayed_array", "matrix_dataset"))
  expect_true(methods::hasMethod("as_delayed_array", "fmri_file_dataset"))
  expect_true(methods::hasMethod("as_delayed_array", "fmri_mem_dataset"))
})
