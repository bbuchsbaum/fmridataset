# Golden tests for FmriSeries objects

test_that("FmriSeries conversion produces consistent output", {
  ref_data <- load_golden_data("reference_data")
  
  # Create dataset and convert to FmriSeries
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Load expected data
  expected_data <- load_golden_data("fmri_series")
  
  # Test class and dimensions
  expect_s4_class(fmri_series, "FmriSeries")
  expect_equal(dim(fmri_series), expected_data$dims)
  
  # Test data content (convert to array for comparison)
  actual_array <- as.array(fmri_series)
  compare_golden(actual_array, expected_data$data)
})

test_that("FmriSeries metadata is preserved", {
  ref_data <- load_golden_data("reference_data")
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Test metadata accessors
  expect_equal(get_TR(fmri_series), ref_data$metadata$TR)
  expect_equal(n_timepoints(fmri_series), ncol(ref_data$matrix_data))
})

test_that("FmriSeries tibble conversion works consistently", {
  ref_data <- load_golden_data("reference_data")
  
  # Small dataset for tibble testing
  small_data <- ref_data$matrix_data[1:10, 1:20]
  dset <- matrix_dataset(
    small_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(small_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Convert to tibble
  tbl <- as_tibble(fmri_series)
  
  # Test structure
  expect_s3_class(tbl, "tbl_df")
  expect_equal(nrow(tbl), prod(dim(small_data)))
  expect_true("voxel" %in% names(tbl))
  expect_true("time" %in% names(tbl))
  expect_true("value" %in% names(tbl))
  
  # Test data integrity
  # First voxel, all time points
  voxel1_data <- tbl[tbl$voxel == 1, "value", drop = TRUE]
  expect_equal(voxel1_data, as.vector(small_data[1, ]))
})

test_that("FmriSeries print output matches snapshot", {
  ref_data <- load_golden_data("reference_data")
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  expect_snapshot({
    show(fmri_series)
  })
})

test_that("FmriSeries subsetting maintains consistency", {
  ref_data <- load_golden_data("reference_data")
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Test various subsetting operations
  subset1 <- fmri_series[1:10, 1:20]
  expect_equal(dim(subset1), c(10, 20))
  
  # Extract and compare data
  subset_data <- as.array(subset1)
  expected_subset <- ref_data$matrix_data[1:10, 1:20]
  compare_golden(subset_data, expected_subset)
  
  # Test single row/column extraction
  single_row <- fmri_series[1, ]
  expect_equal(length(single_row), ncol(ref_data$matrix_data))
  
  single_col <- fmri_series[, 1]
  expect_equal(length(single_col), nrow(ref_data$matrix_data))
})

test_that("FmriSeries operations produce consistent results", {
  ref_data <- load_golden_data("reference_data")
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Test arithmetic operations
  scaled <- fmri_series * 2
  expect_equal(as.array(scaled)[1, 1], as.array(fmri_series)[1, 1] * 2)
  
  # Test summary statistics
  col_means <- DelayedMatrixStats::colMeans2(fmri_series)
  expected_means <- colMeans(ref_data$matrix_data)
  expect_equal(col_means, expected_means, tolerance = 1e-10)
  
  row_means <- DelayedMatrixStats::rowMeans2(fmri_series)
  expected_row_means <- rowMeans(ref_data$matrix_data)
  expect_equal(row_means, expected_row_means, tolerance = 1e-10)
})

test_that("FmriSeries multi-run handling is consistent", {
  ref_data <- load_golden_data("reference_data")
  
  # Create multi-run dataset
  dset <- matrix_dataset(
    ref_data$multirun_data,
    TR = ref_data$metadata$TR
  )
  
  fmri_series <- as_delayed_array(dset)
  
  # Test dimensions match concatenated data
  total_cols <- sum(sapply(ref_data$multirun_data, ncol))
  expect_equal(ncol(fmri_series), total_cols)
  expect_equal(nrow(fmri_series), nrow(ref_data$multirun_data[[1]]))
  
  # Test run boundaries
  run_lens <- get_run_lengths(fmri_series)
  expect_equal(length(run_lens), length(ref_data$multirun_data))
  expect_equal(sum(run_lens), total_cols)
})