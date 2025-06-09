test_that("as.matrix_dataset works with all dataset types", {
  # Test with matrix_dataset (should return itself)
  mat_data <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mat_dset <- matrix_dataset(mat_data, TR = 2, run_length = 20)

  converted <- as.matrix_dataset(mat_dset)
  expect_identical(converted, mat_dset)
  expect_equal(converted$datamat, mat_data)

  # Test with fmri_mem_dataset
  skip_if_not_installed("neuroim2")

  dims <- c(5, 5, 5, 20)
  mock_vec <- structure(
    array(rnorm(prod(dims)), dims),
    class = c("NeuroVec", "array"),
    dim = dims
  )

  mock_mask <- structure(
    array(TRUE, dims[1:3]),
    class = c("NeuroVol", "array"),
    dim = dims[1:3]
  )

  mem_dset <- fmri_mem_dataset(
    scans = list(mock_vec),
    mask = mock_mask,
    TR = 2
  )

  # Mock the series function for testing
  with_mocked_bindings(
    series = function(vec, indices) {
      matrix(rnorm(length(indices) * dims[4]),
        nrow = dims[4],
        ncol = length(indices)
      )
    },
    .package = "neuroim2",
    {
      converted_mem <- as.matrix_dataset(mem_dset)
      expect_s3_class(converted_mem, "matrix_dataset")
      expect_equal(nrow(converted_mem$datamat), 20)
      expect_equal(ncol(converted_mem$datamat), 125) # 5*5*5
    }
  )

  # Test with backend-based fmri_dataset
  backend <- matrix_backend(mat_data)
  backend_dset <- fmri_dataset(backend, TR = 2, run_length = 20)

  # For now, this should use the fmri_file_dataset method
  # which will call get_data_matrix
  converted_backend <- as.matrix_dataset(backend_dset)
  expect_s3_class(converted_backend, "matrix_dataset")
  expect_equal(converted_backend$datamat, mat_data)
})

test_that("conversion preserves essential properties", {
  # Create dataset with specific properties
  test_data <- matrix(1:300, nrow = 30, ncol = 10)
  event_table <- data.frame(
    onset = c(5, 15, 25),
    duration = rep(2, 3),
    condition = c("A", "B", "A")
  )

  original <- matrix_dataset(
    datamat = test_data,
    TR = 1.5,
    run_length = c(15, 15),
    event_table = event_table
  )

  # Convert to itself
  converted <- as.matrix_dataset(original)

  # Check all properties preserved
  expect_equal(converted$TR, original$TR)
  expect_equal(converted$nruns, original$nruns)
  expect_equal(converted$event_table, original$event_table)
  expect_equal(converted$sampling_frame$blocklens, original$sampling_frame$blocklens)
  expect_equal(length(converted$mask), ncol(test_data))
})
