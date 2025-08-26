# Golden tests for storage backends

test_that("matrix backend produces consistent output", {
  ref_data <- load_golden_data("reference_data")

  # Create backend directly
  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  # Test backend interface
  expect_s3_class(backend, "matrix_backend")

  # Test data retrieval
  backend_open(backend)

  # Get dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims, dim(ref_data$matrix_data))

  # Get full data
  full_data <- backend_get_data(backend)
  compare_golden(full_data, ref_data$matrix_data)

  # Get subset
  subset_data <- backend_get_data(backend,
    voxel_idx = 1:10,
    time_idx = 1:20
  )
  expected_subset <- ref_data$matrix_data[1:10, 1:20]
  compare_golden(subset_data, expected_subset)

  backend_close(backend)
})

test_that("multi-run matrix backend handles correctly", {
  ref_data <- load_golden_data("reference_data")

  # Create multi-run backend - concatenate runs
  combined_data <- do.call(rbind, ref_data$multirun_data)
  backend <- matrix_backend(
    data_matrix = combined_data
  )

  backend_open(backend)

  # Test dimensions
  dims <- backend_get_dims(backend)
  expected_nrow <- sum(sapply(ref_data$multirun_data, nrow))
  expected_ncol <- ncol(ref_data$multirun_data[[1]]) # all runs have same ncol
  expect_equal(dims$spatial, c(expected_ncol, 1, 1))
  expect_equal(dims$time, expected_nrow)

  # Test run boundaries
  run_idx <- 1
  row_start <- 1
  for (run_data in ref_data$multirun_data) {
    row_end <- row_start + nrow(run_data) - 1

    subset <- backend_get_data(backend, rows = row_start:row_end)
    compare_golden(subset, run_data)

    row_start <- row_end + 1
    run_idx <- run_idx + 1
  }

  backend_close(backend)
})

test_that("backend metadata is consistent", {
  ref_data <- load_golden_data("reference_data")

  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  # Test backend structure
  expect_true(is.list(backend))
  expect_true("data_matrix" %in% names(backend))
  expect_true("mask" %in% names(backend))
})

test_that("backend validation works consistently", {
  ref_data <- load_golden_data("reference_data")

  # Valid backend
  valid_backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  expect_silent(validate_backend(valid_backend))

  # Test invalid backends
  expect_error(
    matrix_backend(data_matrix = "not a matrix"),
    class = "fmridataset_error"
  )

  expect_error(
    matrix_backend(data_matrix = data.frame(a = 1:10)),
    "must be a matrix"
  )
})

test_that("backend print output matches snapshot", {
  skip_if(testthat::edition_get() < 3, "Snapshot tests require testthat edition 3")

  ref_data <- load_golden_data("reference_data")

  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  expect_snapshot({
    print(backend)
  })
})

test_that("mock NeuroVec backend works correctly", {
  mock_vec <- load_golden_data("mock_neurvec")

  # Skip this test as it requires S3 method registration
  skip("Mock NeuroVec backend test requires S3 method registration")

  # Test backend operations
  backend_open(mock_vec)

  dims <- backend_get_dims(mock_vec)
  expect_equal(length(dims), 4)
  expect_equal(dims, dim(mock_vec))

  # Test data extraction
  full_data <- backend_get_data(mock_vec)
  expect_equal(dim(full_data), c(prod(dims[1:3]), dims[4]))

  backend_close(mock_vec)
})

test_that("backend edge cases handle correctly", {
  # Single voxel
  single_voxel <- matrix(rnorm(50), nrow = 1, ncol = 50)
  backend_sv <- matrix_backend(data_matrix = single_voxel)

  backend_open(backend_sv)
  dims <- backend_get_dims(backend_sv)
  expect_equal(dims, c(1, 50))

  data <- backend_get_data(backend_sv)
  compare_golden(data, single_voxel)
  backend_close(backend_sv)

  # Single timepoint
  single_time <- matrix(rnorm(100), nrow = 100, ncol = 1)
  backend_st <- matrix_backend(data_matrix = single_time)

  backend_open(backend_st)
  dims <- backend_get_dims(backend_st)
  expect_equal(dims, c(100, 1))

  data <- backend_get_data(backend_st)
  compare_golden(data, single_time)
  backend_close(backend_st)
})
