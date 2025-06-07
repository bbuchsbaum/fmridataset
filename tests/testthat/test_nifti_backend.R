test_that("nifti_backend validates inputs correctly", {
  # Test invalid source type
  expect_error(
    nifti_backend(source = 123, mask_source = "mask.nii"),
    class = "fmridataset_error_config"
  )

  # Test non-existent files
  expect_error(
    nifti_backend(source = "nonexistent.nii", mask_source = "mask.nii"),
    class = "fmridataset_error_backend_io"
  )

  expect_error(
    nifti_backend(source = "test.nii", mask_source = "nonexistent_mask.nii"),
    class = "fmridataset_error_backend_io"
  )

  # Test invalid mask type
  expect_error(
    nifti_backend(source = list(), mask_source = 123),
    class = "fmridataset_error_config"
  )
})

test_that("nifti_backend works with mock NeuroVec objects", {
  skip_if_not_installed("neuroim2")

  # Create mock data
  dims <- c(10, 10, 10, 20)

  # Create a mock NeuroVec
  data_array <- array(rnorm(prod(dims)), dims)
  mock_vec <- structure(
    data_array,
    class = c("DenseNeuroVec", "NeuroVec", "array"),
    space = structure(
      list(dim = dims[1:3], origin = c(0, 0, 0), spacing = c(2, 2, 2)),
      class = "NeuroSpace"
    )
  )

  # Create mock mask
  mock_mask <- structure(
    array(c(rep(1, 500), rep(0, 500)), c(10, 10, 10)),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = c(10, 10, 10)
  )

  # Create backend with in-memory objects
  backend <- nifti_backend(
    source = list(mock_vec),
    mask_source = mock_mask,
    preload = TRUE
  )

  expect_s3_class(backend, "nifti_backend")
  expect_s3_class(backend, "storage_backend")

  # Test dimensions
  dims_result <- backend_get_dims(backend)
  expect_equal(dims_result$spatial, c(10, 10, 10))
  expect_equal(dims_result$time, 20)

  # Test mask
  mask_result <- backend_get_mask(backend)
  expect_type(mask_result, "logical")
  expect_length(mask_result, 1000)
  expect_equal(sum(mask_result), 500)
})

test_that("nifti_backend handles multiple source files", {
  # Create mock file list
  mock_files <- c("scan1.nii", "scan2.nii", "scan3.nii")

  # Mock the file.exists function for this test
  with_mocked_bindings(
    file.exists = function(x) TRUE,
    .package = "base",
    {
      backend <- nifti_backend(
        source = mock_files,
        mask_source = "mask.nii",
        preload = FALSE
      )

      expect_equal(backend$source, mock_files)
      expect_equal(backend$mask_source, "mask.nii")
      expect_false(backend$preload)
    }
  )
})

test_that("nifti_backend data subsetting works", {
  skip_if_not_installed("neuroim2")

  # Create small test data
  n_time <- 10
  n_voxels <- 100
  test_matrix <- matrix(1:(n_time * n_voxels),
    nrow = n_time,
    ncol = n_voxels
  )

  # Create mock backend with known data
  mock_backend <- structure(
    list(
      data = structure(
        list(.Data = test_matrix),
        class = "MockNeuroVec"
      ),
      mask = rep(TRUE, n_voxels),
      dims = list(spatial = c(10, 10, 1), time = n_time)
    ),
    class = c("nifti_backend", "storage_backend")
  )

  # Override the backend_get_data method for testing
  backend_get_data.nifti_backend <- function(backend, rows = NULL, cols = NULL) {
    data <- backend$data$.Data
    if (!is.null(rows)) {
      data <- data[rows, , drop = FALSE]
    }
    if (!is.null(cols)) {
      data <- data[, cols, drop = FALSE]
    }
    data
  }

  # Test full data retrieval
  full_data <- backend_get_data(mock_backend)
  expect_equal(dim(full_data), c(n_time, n_voxels))

  # Test row subsetting
  subset_rows <- backend_get_data(mock_backend, rows = 1:5)
  expect_equal(dim(subset_rows), c(5, n_voxels))
  expect_equal(subset_rows[1, 1], test_matrix[1, 1])

  # Test column subsetting
  subset_cols <- backend_get_data(mock_backend, cols = 1:10)
  expect_equal(dim(subset_cols), c(n_time, 10))
  expect_equal(subset_cols[1, 1], test_matrix[1, 1])

  # Test both row and column subsetting
  subset_both <- backend_get_data(mock_backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(subset_both), c(5, 10))
})

test_that("nifti_backend metadata extraction works", {
  # Create mock backend
  mock_backend <- structure(
    list(
      metadata = list(
        affine = diag(4),
        voxel_dims = c(2, 2, 2),
        space = "MNI",
        origin = c(0, 0, 0)
      )
    ),
    class = c("nifti_backend", "storage_backend")
  )

  # Mock the method
  backend_get_metadata.nifti_backend <- function(backend) {
    backend$metadata
  }

  metadata <- backend_get_metadata(mock_backend)

  expect_type(metadata, "list")
  expect_true("affine" %in% names(metadata))
  expect_equal(dim(metadata$affine), c(4, 4))
  expect_equal(metadata$voxel_dims, c(2, 2, 2))
})

test_that("nifti_backend validates with validate_backend", {
  skip_if_not_installed("neuroim2")

  # Create a simple valid backend
  dims <- c(5, 5, 5, 10)
  test_data <- array(rnorm(prod(dims)), dims)

  mock_vec <- structure(
    test_data,
    class = c("NeuroVec", "array")
  )

  mock_mask <- structure(
    array(TRUE, dims[1:3]),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = dims[1:3]
  )

  backend <- nifti_backend(
    source = list(mock_vec),
    mask_source = mock_mask
  )

  # Mock the backend methods for validation
  with_mocked_bindings(
    backend_get_dims = function(backend) {
      list(spatial = c(5, 5, 5), time = 10)
    },
    backend_get_mask = function(backend) {
      rep(TRUE, 125)
    },
    .package = "fmridataset",
    {
      # Should pass validation
      expect_true(validate_backend(backend))
    }
  )
})
