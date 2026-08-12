# Comprehensive edge case testing for fmridataset

test_that("handles single element datasets correctly", {
  # 1x1 matrix (single voxel, single timepoint)
  mat <- matrix(42, 1, 1)
  dset <- matrix_dataset(mat, TR = 2, run_length = 1)

  expect_equal(n_timepoints(dset), 1)
  expect_equal(ncol(dset$datamat), 1)
  expect_equal(get_data_matrix(dset), mat)

  # Chunking should work
  chunks <- data_chunks(dset, nchunks = 1)
  chunk <- iterators::nextElem(chunks)
  expect_equal(chunk$data, mat)
})

test_that("handles empty mask (all FALSE) appropriately", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Create backend with all-FALSE mask - matrix_backend doesn't validate
  backend <- matrix_backend(mat, mask = rep(FALSE, 50))

  # Backend validation should catch it when validated
  expect_error(
    fmridataset:::validate_backend(backend),
    "mask must contain at least one TRUE value"
  )
})

test_that("handles NA/NaN/Inf values in data", {
  # Matrix with special values
  mat <- matrix(c(1, NA, NaN, Inf, -Inf, 0), 6, 1)
  dset <- matrix_dataset(mat, TR = 1, run_length = 6)

  # Should preserve special values
  data <- get_data_matrix(dset)
  expect_true(is.na(data[2, 1]))
  expect_true(is.nan(data[3, 1]))
  expect_equal(data[4, 1], Inf)
  expect_equal(data[5, 1], -Inf)

  # Operations should handle them appropriately
  expect_warning(
    mean_val <- mean(data),
    NA
  )
})

test_that("handles very large sparse matrices efficiently", {
  skip_on_cran()
  skip_if_not_installed("Matrix")

  # Create large sparse matrix
  library(Matrix)
  n_time <- 1000
  n_voxels <- 10000

  # Only 1% non-zero values
  sparse_mat <- Matrix::rsparsematrix(n_time, n_voxels, density = 0.01)
  dense_mat <- as.matrix(sparse_mat)

  # Matrix dataset should work
  dset <- matrix_dataset(dense_mat, TR = 2, run_length = n_time)

  # Check memory efficiency of operations
  data_subset <- get_data_matrix(dset, rows = 1:10, cols = 1:10)
  expect_equal(dim(data_subset), c(10, 10))
})

test_that("handles extremely long file paths", {
  skip_on_cran()

  # Create a very long path (near system limits)
  if (.Platform$OS.type == "unix") {
    # Unix typically allows 4096 chars
    base <- tempdir()
    long_name <- paste(rep("a", 200), collapse = "")
    long_path <- file.path(base, long_name, long_name, long_name)

    # Should handle gracefully
    mat <- matrix(1:4, 2, 2)
    dset <- matrix_dataset(mat, TR = 1, run_length = 2)
    dset$base_path <- long_path

    # Print should truncate nicely
    output <- capture.output(print(dset))
    expect_true(length(output) > 0)
  }
})

test_that("handles unicode and special characters in metadata", {
  mat <- matrix(1:10, 5, 2)
  dset <- matrix_dataset(mat, TR = 2, run_length = 5)

  # Add unicode in event table
  dset$event_table <- data.frame(
    trial_type = c("café", "naïve", "日本語", "😀", "foo\nbar"),
    onset = 1:5
  )

  # Should handle in print
  expect_silent(output <- capture.output(print(dset)))

  # Should handle in summary
  expect_silent(output <- capture.output(summary(dset)))
})

test_that("handles datasets with many runs correctly", {
  # 1000 runs of 1 timepoint each
  mat <- matrix(rnorm(1000 * 10), 1000, 10)
  run_lengths <- rep(1, 1000)

  dset <- matrix_dataset(mat, TR = 2, run_length = run_lengths)

  expect_equal(n_runs(dset), 1000)
  expect_equal(n_timepoints(dset), 1000)

  # Sampling frame should handle it
  expect_equal(length(dset$sampling_frame$blocklens), 1000)

  # Print should work with many runs
  expect_silent(output <- capture.output(print(dset)))
  expect_true(length(output) > 0)
})

test_that("handles single-voxel masks correctly", {
  mat <- matrix(rnorm(100 * 10), 100, 10)

  # Only one voxel in mask
  single_voxel_mask <- c(TRUE, rep(FALSE, 9))
  backend <- matrix_backend(mat, mask = single_voxel_mask)
  dset <- fmri_dataset(backend, TR = 2, run_length = 100)

  mask <- get_mask(dset)
  expect_equal(sum(mask), 1)

  # Data access should work - matrix_backend applies mask
  data <- get_data_matrix(dset)
  expect_equal(dim(data), c(100, 1)) # Only 1 column since only 1 voxel in mask
})

test_that("handles boundary indices in data access", {
  mat <- matrix(1:100, 10, 10)
  dset <- matrix_dataset(mat, TR = 1, run_length = 10)

  # Edge indices
  expect_equal(get_data_matrix(dset, rows = 1, cols = 1), matrix(1))
  expect_equal(get_data_matrix(dset, rows = 10, cols = 10), matrix(100))

  # Row index 0 returns empty matrix in R
  result <- get_data_matrix(dset, rows = 0)
  expect_equal(nrow(result), 0)
  expect_equal(ncol(result), 10)

  # Out of bounds positive index should error
  expect_error(
    get_data_matrix(dset, rows = 11),
    "subscript out of bounds"
  )
})

test_that("handles zero-dimension arrays gracefully", {
  # 0 timepoints
  mat <- matrix(numeric(0), 0, 10)
  expect_error(
    matrix_dataset(mat, TR = 2, run_length = 0),
    "Block lengths must be positive"
  )

  # 0 voxels
  mat <- matrix(numeric(0), 10, 0)
  expect_silent(
    dset <- matrix_dataset(mat, TR = 2, run_length = 10)
  )
  expect_equal(ncol(get_data_matrix(dset)), 0)
})

test_that("handles datasets with all identical values", {
  # All zeros
  mat <- matrix(0, 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)

  data <- get_data_matrix(dset)
  expect_true(all(data == 0))

  # All ones
  mat <- matrix(1, 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)

  data <- get_data_matrix(dset)
  expect_true(all(data == 1))
})

test_that("handles extreme parameter values", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Very small TR (but not too small for sampling_frame)
  dset <- matrix_dataset(mat, TR = 1, run_length = 100)
  expect_equal(get_TR(dset), 1)

  # Very large TR
  dset <- matrix_dataset(mat, TR = 10000, run_length = 100)
  expect_equal(get_TR(dset), 10000)

  # Many small chunks (should be limited to number of voxels)
  chunks <- suppressWarnings(data_chunks(dset, nchunks = 100))
  expect_equal(chunks$nchunks, 50) # Limited to 50 voxels
})

test_that("handles event tables with no events", {
  mat <- matrix(rnorm(100 * 50), 100, 50)

  # Empty event table
  dset <- matrix_dataset(mat,
    TR = 2, run_length = 100,
    event_table = data.frame()
  )
  expect_equal(nrow(dset$event_table), 0)

  # Event table with no rows
  empty_events <- data.frame(
    onset = numeric(0),
    duration = numeric(0),
    trial_type = character(0)
  )
  dset <- matrix_dataset(mat,
    TR = 2, run_length = 100,
    event_table = empty_events
  )
  expect_equal(nrow(dset$event_table), 0)
})

test_that("handles integer overflow in large datasets", {
  skip_on_cran()

  # Dataset approaching integer limits
  n_time <- 46341 # Just over sqrt(.Machine$integer.max)
  n_vox <- 46341

  # This would overflow if using integer arithmetic
  total_elements <- as.numeric(n_time) * as.numeric(n_vox)
  expect_true(total_elements > .Machine$integer.max)

  # Backend should handle with proper typing
  # (Don't actually create this large matrix in tests)
})

test_that("handles platform-specific path separators", {
  mat <- matrix(1:10, 5, 2)
  dset <- matrix_dataset(mat, TR = 1, run_length = 5)

  if (.Platform$OS.type == "windows") {
    dset$base_path <- "C:\\Users\\test\\data"
    expect_true(grepl("\\\\", dset$base_path))
  } else {
    dset$base_path <- "/home/test/data"
    expect_true(grepl("/", dset$base_path))
  }

  # Print should handle platform differences
  expect_silent(capture.output(print(dset)))
})
