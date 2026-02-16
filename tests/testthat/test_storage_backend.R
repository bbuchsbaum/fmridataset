test_that("storage backend contract validation works", {
  # Create a minimal mock backend without any methods defined
  mock_backend <- structure(
    list(),
    class = c("nonexistent_backend_type", "storage_backend")
  )

  # Test that validation fails without required methods
  expect_error(
    validate_backend(mock_backend),
    class = "error" # Will be a generic error about missing methods
  )
})

test_that("backend validation checks dimension requirements", {
  # Use an existing backend (matrix_backend) and mock the get_dims method
  test_matrix <- matrix(1:1000, 100, 10)
  mock_backend <- matrix_backend(test_matrix)

  # Test normal case first
  expect_true(validate_backend(mock_backend))

  # Test invalid spatial dimensions
  with_mocked_bindings(
    backend_get_dims = function(backend) {
      list(spatial = c(10, 10), time = 100) # Wrong length
    },
    .package = "fmridataset",
    {
      expect_error(
        validate_backend(mock_backend),
        "spatial dimensions must be a numeric vector of length 3"
      )
    }
  )

  # Test invalid time dimension
  with_mocked_bindings(
    backend_get_dims = function(backend) {
      list(spatial = c(10, 10, 10), time = -1)
    },
    .package = "fmridataset",
    {
      expect_error(
        validate_backend(mock_backend),
        "time dimension must be a positive integer"
      )
    }
  )
})

test_that("backend validation checks mask requirements", {
  # Use existing backend and mock the mask method
  test_matrix <- matrix(1:1000, 100, 10)
  mock_backend <- matrix_backend(test_matrix)

  # Test all FALSE mask
  with_mocked_bindings(
    backend_get_mask = function(backend) rep(FALSE, 10),
    .package = "fmridataset",
    {
      expect_error(
        validate_backend(mock_backend),
        "mask must contain at least one TRUE value"
      )
    }
  )

  # Test mask with NA values
  with_mocked_bindings(
    backend_get_mask = function(backend) c(rep(TRUE, 9), NA),
    .package = "fmridataset",
    {
      expect_error(
        validate_backend(mock_backend),
        "mask cannot contain NA values"
      )
    }
  )

  # Test wrong mask length
  with_mocked_bindings(
    backend_get_mask = function(backend) rep(TRUE, 5), # Should be 10
    backend_get_dims = function(backend) list(spatial = c(2, 5, 1), time = 100),
    .package = "fmridataset",
    {
      expect_error(
        validate_backend(mock_backend),
        "mask length .* must equal prod"
      )
    }
  )
})

test_that("error classes work correctly", {
  # Test fmridataset_error_backend_io
  err <- fmridataset_error_backend_io(
    message = "Failed to read file",
    file = "test.nii",
    operation = "read"
  )

  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_s3_class(err, "fmridataset_error")
  expect_equal(err$file, "test.nii")
  expect_equal(err$operation, "read")

  # Test fmridataset_error_config
  err <- fmridataset_error_config(
    message = "Invalid parameter",
    parameter = "mask",
    value = NULL
  )

  expect_s3_class(err, "fmridataset_error_config")
  expect_equal(err$parameter, "mask")

  # Test stop_fmridataset
  expect_error(
    stop_fmridataset(
      fmridataset_error_config,
      message = "Test error"
    ),
    class = "fmridataset_error_config"
  )
})
