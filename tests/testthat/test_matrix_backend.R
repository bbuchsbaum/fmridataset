test_that("matrix_backend validates inputs correctly", {
  # Test non-matrix input
  expect_error(
    matrix_backend(data_matrix = "not a matrix"),
    class = "fmridataset_error_config"
  )
  
  # Test invalid mask type
  test_matrix <- matrix(1:100, nrow = 10, ncol = 10)
  expect_error(
    matrix_backend(data_matrix = test_matrix, mask = "not logical"),
    class = "fmridataset_error_config"
  )
  
  # Test mask length mismatch
  expect_error(
    matrix_backend(data_matrix = test_matrix, mask = c(TRUE, FALSE)),
    "mask length .* must equal number of columns"
  )
  
  # Test invalid spatial dimensions
  expect_error(
    matrix_backend(data_matrix = test_matrix, spatial_dims = c(5, 5)),
    "spatial_dims must be a numeric vector of length 3"
  )
  
  # Test spatial dims product mismatch
  expect_error(
    matrix_backend(data_matrix = test_matrix, spatial_dims = c(2, 2, 2)),
    "Product of spatial_dims .* must equal number of voxels"
  )
})

test_that("matrix_backend creates valid backend with defaults", {
  test_matrix <- matrix(rnorm(100), nrow = 10, ncol = 10)
  
  backend <- matrix_backend(data_matrix = test_matrix)
  
  expect_s3_class(backend, "matrix_backend")
  expect_s3_class(backend, "storage_backend")
  
  # Check defaults
  expect_equal(length(backend$mask), 10)
  expect_true(all(backend$mask))
  expect_equal(backend$spatial_dims, c(10, 1, 1))
})

test_that("matrix_backend methods work correctly", {
  # Create test data
  n_time <- 20
  n_voxels <- 100
  test_matrix <- matrix(seq_len(n_time * n_voxels), 
                       nrow = n_time, 
                       ncol = n_voxels)
  
  # Create mask with some FALSE values
  mask <- rep(TRUE, n_voxels)
  mask[1:10] <- FALSE
  
  backend <- matrix_backend(
    data_matrix = test_matrix,
    mask = mask,
    spatial_dims = c(10, 10, 1),
    metadata = list(source = "test")
  )
  
  # Test open/close (should be no-ops)
  opened <- backend_open(backend)
  expect_identical(opened, backend)
  expect_silent(backend_close(backend))
  
  # Test dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(10, 10, 1))
  expect_equal(dims$time, n_time)
  
  # Test mask
  retrieved_mask <- backend_get_mask(backend)
  expect_identical(retrieved_mask, mask)
  
  # Test full data retrieval
  data <- backend_get_data(backend)
  expect_identical(data, test_matrix)
  
  # Test metadata
  metadata <- backend_get_metadata(backend)
  expect_equal(metadata$source, "test")
})

test_that("matrix_backend data subsetting works", {
  # Create test data with known pattern
  test_matrix <- matrix(1:200, nrow = 20, ncol = 10)
  
  backend <- matrix_backend(data_matrix = test_matrix)
  
  # Test row subsetting
  rows_subset <- backend_get_data(backend, rows = 1:5)
  expect_equal(dim(rows_subset), c(5, 10))
  expect_equal(rows_subset[1, 1], 1)
  expect_equal(rows_subset[5, 1], 5)
  
  # Test column subsetting
  cols_subset <- backend_get_data(backend, cols = c(1, 3, 5))
  expect_equal(dim(cols_subset), c(20, 3))
  expect_equal(cols_subset[1, 1], 1)
  expect_equal(cols_subset[1, 2], 41)  # Column 3
  expect_equal(cols_subset[1, 3], 81)  # Column 5
  
  # Test both row and column subsetting
  both_subset <- backend_get_data(backend, rows = 1:5, cols = 1:3)
  expect_equal(dim(both_subset), c(5, 3))
  expect_equal(both_subset[1, 1], 1)
  expect_equal(both_subset[5, 3], 45)
  
  # Test single row/column (should not drop dimensions)
  single_row <- backend_get_data(backend, rows = 1)
  expect_equal(dim(single_row), c(1, 10))
  
  single_col <- backend_get_data(backend, cols = 1)
  expect_equal(dim(single_col), c(20, 1))
})

test_that("matrix_backend validates with validate_backend", {
  test_matrix <- matrix(rnorm(500), nrow = 50, ncol = 10)
  
  backend <- matrix_backend(
    data_matrix = test_matrix,
    spatial_dims = c(5, 2, 1)
  )
  
  # Should pass validation
  expect_true(validate_backend(backend))
  
  # Test with all FALSE mask (should fail validation)
  backend_fail <- matrix_backend(
    data_matrix = test_matrix,
    mask = rep(FALSE, 10),
    spatial_dims = c(5, 2, 1)
  )
  
  expect_error(
    validate_backend(backend_fail),
    "mask must contain at least one TRUE value"
  )
})

test_that("matrix_backend works with fmri_dataset", {
  # Create test data
  test_data <- matrix(rnorm(300), nrow = 30, ncol = 10)
  
  backend <- matrix_backend(
    data_matrix = test_data,
    spatial_dims = c(10, 1, 1)
  )
  
  # Create dataset using backend
  dataset <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = 30
  )
  
  expect_s3_class(dataset, "fmri_dataset")
  expect_s3_class(dataset$backend, "matrix_backend")
  
  # Test that data access works
  data_retrieved <- get_data_matrix(dataset)
  expect_equal(dim(data_retrieved), dim(test_data))
  expect_equal(data_retrieved, test_data)
})

test_that("matrix_backend preserves data integrity", {
  # Create data with specific patterns to verify integrity
  n_time <- 15
  n_voxels <- 20
  
  # Create data where each column has a distinct pattern
  test_data <- matrix(0, nrow = n_time, ncol = n_voxels)
  for (i in 1:n_voxels) {
    test_data[, i] <- sin(seq(0, 2*pi, length.out = n_time) + i)
  }
  
  backend <- matrix_backend(
    data_matrix = test_data,
    spatial_dims = c(4, 5, 1)
  )
  
  # Retrieve and verify data
  retrieved <- backend_get_data(backend)
  expect_equal(retrieved, test_data, tolerance = .Machine$double.eps^0.5)
  
  # Verify specific patterns are preserved
  expect_equal(retrieved[, 1], test_data[, 1])
  expect_equal(retrieved[, n_voxels], test_data[, n_voxels])
})