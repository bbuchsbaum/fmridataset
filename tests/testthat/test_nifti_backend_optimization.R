library(testthat)

test_that("nifti_backend uses read_header for efficient dimension extraction", {
  skip_if_not_installed("neuroim2")
  
  # Use test data file
  test_file <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(test_file))
  
  # Create backend
  backend <- nifti_backend(
    source = test_file,
    mask_source = test_file,  # Use same file as mask for testing
    preload = FALSE
  )
  
  # Get dimensions - this should use read_header, not read_vec
  dims <- backend_get_dims(backend)
  
  # Verify dimensions are correct
  expect_type(dims, "list")
  expect_named(dims, c("spatial", "time"))
  expect_length(dims$spatial, 3)
  expect_type(dims$time, "integer")
  
  # Check that dimensions match what we'd get from read_vec
  vec <- neuroim2::read_vec(test_file)
  expected_dims <- dim(vec)
  expect_equal(dims$spatial, expected_dims[1:3])
  expect_equal(dims$time, expected_dims[4])
})

test_that("nifti_backend handles multiple files efficiently", {
  skip_if_not_installed("neuroim2")
  
  # Use test data file
  test_file <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(test_file))
  
  # Create backend with multiple files
  backend <- nifti_backend(
    source = rep(test_file, 3),  # Simulate 3 runs
    mask_source = test_file,
    preload = FALSE
  )
  
  # Get dimensions - should sum time dimension across files
  dims <- backend_get_dims(backend)
  
  # Each file has 4 timepoints, so 3 files = 12 total
  header <- neuroim2::read_header(test_file)
  single_time <- header@dims[4]
  expect_equal(dims$time, single_time * 3)
})

test_that("nifti_backend metadata extraction uses read_header", {
  skip_if_not_installed("neuroim2")
  
  # Use test data file
  test_file <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(test_file))
  
  # Create backend
  backend <- nifti_backend(
    source = test_file,
    mask_source = test_file,
    preload = FALSE
  )
  
  # Get metadata - this should use read_header
  metadata <- backend_get_metadata(backend)
  
  # Verify metadata structure
  expect_type(metadata, "list")
  expect_named(metadata, c("affine", "voxel_dims", "space", "origin", "dims"))
  
  # Check metadata values match header
  header <- neuroim2::read_header(test_file)
  expect_equal(metadata$dims, header@dims)
  expect_equal(metadata$voxel_dims, header@spacing)
})

test_that("nifti_backend dimension caching works", {
  skip_if_not_installed("neuroim2")
  
  # Use test data file
  test_file <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(test_file))
  
  # Create backend
  backend <- nifti_backend(
    source = test_file,
    mask_source = test_file,
    preload = FALSE
  )
  
  # First call - reads from header
  dims1 <- backend_get_dims(backend)
  
  # Second call - should use cached value
  dims2 <- backend_get_dims(backend)
  
  # Results should be identical
  expect_identical(dims1, dims2)
  
  # Check that dims are cached
  expect_false(is.null(backend$dims))
  expect_identical(backend$dims, dims1)
})