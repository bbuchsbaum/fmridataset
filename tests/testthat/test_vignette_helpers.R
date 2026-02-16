# Tests for R/vignette_helpers.R

test_that("generate_example_fmri_data returns correct dimensions", {
  data <- generate_example_fmri_data(n_timepoints = 50, n_voxels = 100)
  expect_true(is.matrix(data))
  expect_equal(nrow(data), 50)
  expect_equal(ncol(data), 100)
})

test_that("generate_example_fmri_data is reproducible with same seed", {
  d1 <- generate_example_fmri_data(n_timepoints = 20, n_voxels = 10, seed = 42)
  d2 <- generate_example_fmri_data(n_timepoints = 20, n_voxels = 10, seed = 42)
  expect_equal(d1, d2)
})

test_that("generate_example_fmri_data with activation adds signal", {
  n_tp <- 50
  n_vox <- 100
  n_active <- 10
  active_periods <- 10:20
  d_no_act <- generate_example_fmri_data(n_tp, n_vox, n_active = 0, seed = 1)
  d_act <- generate_example_fmri_data(n_tp, n_vox,
    n_active = n_active,
    activation_periods = active_periods,
    signal_strength = 5.0,
    seed = 1
  )
  # With strong signal, the activation should differ from no-activation
  expect_false(identical(d_no_act, d_act))
})

test_that("generate_example_events creates correct event table", {
  events <- generate_example_events(n_runs = 3, events_per_run = 5)
  expect_true(is.data.frame(events))
  expect_equal(nrow(events), 15) # 3 runs * 5 events

  expect_true("onset" %in% names(events))
  expect_true("duration" %in% names(events))
  expect_true("trial_type" %in% names(events))
  expect_true("run" %in% names(events))
  expect_equal(length(unique(events$run)), 3)
})

test_that("generate_example_paths creates correct paths", {
  paths <- generate_example_paths(n_runs = 3)
  expect_length(paths, 3)
  expect_true(all(grepl("sub-001_run-", paths)))
  expect_true(all(grepl("bold.nii.gz$", paths)))
})

test_that("generate_example_paths with custom subject", {
  paths <- generate_example_paths(n_runs = 2, subject_id = "sub-042")
  expect_length(paths, 2)
  expect_true(all(grepl("sub-042", paths)))
})

test_that("generate_example_mask creates mask with correct fraction", {
  mask <- generate_example_mask(n_voxels = 100, fraction_valid = 0.5)
  expect_type(mask, "logical")
  expect_length(mask, 100)
  expect_equal(sum(mask), 50)
})

test_that("generate_example_mask default fraction", {
  mask <- generate_example_mask(n_voxels = 1000)
  expect_length(mask, 1000)
  expect_equal(sum(mask), 800) # 80% default
})

test_that("generate_benchmark_data creates valid result", {
  result <- generate_benchmark_data()
  expect_true(is.data.frame(result))
  expect_true("size" %in% names(result))
  expect_true("operation" %in% names(result))
  expect_true("time_ms" %in% names(result))
  expect_true(all(result$time_ms >= 1)) # pmax ensures minimum 1
})

test_that("print_dataset_info works with matrix_dataset", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  out <- capture.output(print_dataset_info(ds, title = "Test Dataset"))
  expect_true(any(grepl("Test Dataset", out)))
  expect_true(any(grepl("Dataset class", out)))
})

test_that("print_dataset_info works without title", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  out <- capture.output(print_dataset_info(ds))
  expect_true(any(grepl("Dataset class", out)))
})

test_that("analyze_run computes mean correctly", {
  data <- matrix(1:20, nrow = 4, ncol = 5)
  result <- analyze_run(data, method = "mean")
  expect_equal(result, colMeans(data))
})

test_that("analyze_run computes var", {
  data <- matrix(rnorm(20), nrow = 4, ncol = 5)
  result <- analyze_run(data, method = "var")
  expected <- apply(data, 2, var)
  expect_equal(result, expected)
})

test_that("analyze_run computes max", {
  data <- matrix(rnorm(20), nrow = 4, ncol = 5)
  result <- analyze_run(data, method = "max")
  expected <- apply(data, 2, max)
  expect_equal(result, expected)
})

test_that("analyze_run default is mean", {
  data <- matrix(1:20, nrow = 4, ncol = 5)
  result <- analyze_run(data, method = "unknown_method")
  expect_equal(result, colMeans(data))
})
