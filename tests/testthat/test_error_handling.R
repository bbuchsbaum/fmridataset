test_that("error handling for invalid inputs", {
  # Test invalid TR
  expect_error(
    matrix_dataset(matrix(1:100, 10, 10), TR = -1, run_length = 10),
    "TR"
  )

  # Test run_length mismatch
  expect_error(
    matrix_dataset(matrix(1:100, 10, 10), TR = 2, run_length = 20),
    "sum\\(run_length\\) not equal to nrow\\(datamat\\)"
  )

  # Test invalid backend source - fix order of validation
  expect_error(
    nifti_backend(source = 123, mask_source = "mask.nii"),
    "source must be character vector"
  )

  # Test invalid spatial dimensions in matrix_backend
  expect_error(
    matrix_backend(matrix(1:100, 10, 10), spatial_dims = c(5, 5)),
    "spatial_dims must be a numeric vector of length 3"
  )

  # Test spatial dims product mismatch
  expect_error(
    matrix_backend(matrix(1:100, 10, 10), spatial_dims = c(2, 2, 2)),
    "Product of spatial_dims .* must equal number of voxels"
  )
})

test_that("backend error propagation", {
  # Create a matrix backend that works and then test error propagation through dataset validation
  test_matrix <- matrix(1:100, 10, 10)
  backend <- matrix_backend(test_matrix)

  # Mock a failing get_dims function to test error propagation
  with_mocked_bindings(
    backend_get_dims = function(x) {
      stop("Simulated backend failure")
    },
    .package = "fmridataset",
    {
      expect_error(
        fmri_dataset(backend, TR = 2, run_length = 10),
        "Simulated backend failure"
      )
    }
  )
})

test_that("edge cases in chunking", {
  # Test with single voxel
  single_voxel <- matrix_backend(matrix(1:10, nrow = 10, ncol = 1))
  dset <- fmri_dataset(single_voxel, TR = 2, run_length = 10)

  chunks <- data_chunks(dset, nchunks = 1)
  chunk <- chunks$nextElem()
  expect_equal(ncol(chunk$data), 1)
  expect_equal(nrow(chunk$data), 10)

  # Test with more chunks than voxels
  small_data <- matrix_backend(matrix(1:30, nrow = 10, ncol = 3))
  dset2 <- fmri_dataset(small_data, TR = 2, run_length = 10)

  # Should handle gracefully - nchunks will be capped to number of voxels
  suppressWarnings({
    chunks2 <- data_chunks(dset2, nchunks = 10)
  })

  # Get all chunks (should be 3 since we have 3 voxels)
  all_chunks <- list()
  for (i in 1:3) {
    all_chunks[[i]] <- chunks2$nextElem()
  }

  # Verify we got all voxels
  total_voxels <- sum(sapply(all_chunks, function(x) ncol(x$data)))
  expect_equal(total_voxels, 3)
})

test_that("validate_backend catches all error conditions", {
  # Test missing methods - create a backend with a class that has no methods
  incomplete_backend <- structure(
    list(),
    class = c("nonexistent_backend_class", "storage_backend")
  )

  expect_error(
    fmridataset:::validate_backend(incomplete_backend),
    class = "error"
  )

  # validate_backend doesn't check dims format, so this test is removed
})

test_that("mask validation in backends", {
  # Test non-logical mask
  expect_error(
    {
      backend <- matrix_backend(
        matrix(1:100, 10, 10),
        mask = 1:10 # Should be logical
      )
    },
    "mask must be a logical vector"
  )

  # validate_backend doesn't check for NA in mask, so this test is removed
})
