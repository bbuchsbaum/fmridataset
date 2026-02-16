# test-dataset_methods.R
# Tests for dataset_methods.R - delegation to sampling_frame

# ============================================
# matrix_dataset delegation tests
# ============================================

test_that("matrix_dataset delegates get_TR correctly", {
  # Create dataset with TR=2.5
  test_matrix <- matrix(rnorm(100), nrow = 10, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2.5, run_length = 10)

  # Verify get_TR delegates to sampling_frame
  expect_equal(get_TR(dset), 2.5)
  expect_equal(get_TR(dset), get_TR(dset$sampling_frame))
})

test_that("matrix_dataset delegates n_timepoints correctly", {
  # Create dataset with 100 row matrix
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = 100)

  # Verify n_timepoints delegates to sampling_frame
  expect_equal(n_timepoints(dset), 100)
  expect_equal(n_timepoints(dset), n_timepoints(dset$sampling_frame))
})

test_that("matrix_dataset delegates n_runs correctly", {
  # Create with run_length = c(50, 50)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(50, 50))

  # Verify n_runs delegates to sampling_frame
  expect_equal(n_runs(dset), 2)
  expect_equal(n_runs(dset), n_runs(dset$sampling_frame))
})

test_that("matrix_dataset delegates get_run_lengths correctly", {
  # Create with run_length = c(30, 40, 30)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(30, 40, 30))

  # Verify get_run_lengths delegates to sampling_frame
  expect_equal(get_run_lengths(dset), c(30, 40, 30))
  expect_equal(get_run_lengths(dset), get_run_lengths(dset$sampling_frame))
})

test_that("matrix_dataset delegates blocklens correctly", {
  # Create with run_length
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(30, 40, 30))

  # Verify blocklens delegates to sampling_frame
  expect_equal(blocklens(dset), c(30, 40, 30))
  expect_equal(blocklens(dset), blocklens(dset$sampling_frame))
})

test_that("matrix_dataset delegates blockids correctly", {
  # Create with run_length
  test_matrix <- matrix(rnorm(600), nrow = 60, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(20, 20, 20))

  # Verify blockids delegates to sampling_frame
  block_ids <- blockids(dset)
  expect_equal(length(block_ids), 60)
  # First 20 should be run 1, next 20 run 2, last 20 run 3
  expect_equal(unique(block_ids[1:20]), 1)
  expect_equal(unique(block_ids[21:40]), 2)
  expect_equal(unique(block_ids[41:60]), 3)
  expect_equal(blockids(dset), blockids(dset$sampling_frame))
})

test_that("matrix_dataset delegates get_run_duration correctly", {
  # TR=2, run_length=50 means duration=100s per run
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(50, 50))

  # Verify get_run_duration delegates to sampling_frame
  durations <- get_run_duration(dset)
  expect_equal(durations, c(100, 100))
  expect_equal(get_run_duration(dset), get_run_duration(dset$sampling_frame))
})

test_that("matrix_dataset delegates get_total_duration correctly", {
  # TR=2, total 100 timepoints = 200s total
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = c(50, 50))

  # Verify get_total_duration delegates to sampling_frame
  expect_equal(get_total_duration(dset), 200)
  expect_equal(get_total_duration(dset), get_total_duration(dset$sampling_frame))
})

test_that("matrix_dataset delegates samples correctly", {
  # Create dataset
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2, run_length = 100)

  # Verify samples delegates to sampling_frame
  sample_indices <- samples(dset)
  expect_equal(length(sample_indices), 100)
  expect_equal(samples(dset), samples(dset$sampling_frame))
})

# ============================================
# fmri_mem_dataset delegation tests
# ============================================

test_that("fmri_mem_dataset delegation works", {
  skip_if_not_installed("neuroim2")

  # Create NeuroVec and mask
  dims <- c(10, 10, 10, 50)
  nvec <- neuroim2::NeuroVec(
    array(rnorm(prod(dims)), dims),
    space = neuroim2::NeuroSpace(dims)
  )

  mask_dims <- c(10, 10, 10)
  mask_vol <- neuroim2::NeuroVol(
    array(1, mask_dims),
    space = neuroim2::NeuroSpace(mask_dims)
  )

  # Create fmri_mem_dataset with 2 runs
  dset <- fmri_mem_dataset(list(nvec), mask_vol, TR = 2.5, run_length = 50)

  # Test get_TR delegation
  expect_equal(get_TR(dset), 2.5)
  expect_equal(get_TR(dset), get_TR(dset$sampling_frame))

  # Test n_timepoints delegation
  expect_equal(n_timepoints(dset), 50)
  expect_equal(n_timepoints(dset), n_timepoints(dset$sampling_frame))

  # Test n_runs delegation
  expect_equal(n_runs(dset), 1)
  expect_equal(n_runs(dset), n_runs(dset$sampling_frame))
})

# ============================================
# fmri_file_dataset delegation tests
# ============================================

test_that("fmri_file_dataset delegation works", {
  # Create fmri_dataset with dummy_mode=TRUE
  dset <- fmri_dataset(
    scans = c("dummy1.nii"),
    mask = "dummy_mask.nii",
    TR = 2.0,
    run_length = 100,
    dummy_mode = TRUE
  )

  # Test delegation methods
  expect_equal(get_TR(dset), 2.0)
  expect_equal(n_timepoints(dset), 100)
  expect_equal(n_runs(dset), 1)
  expect_equal(get_run_lengths(dset), 100)
  expect_equal(blocklens(dset), 100)
  expect_equal(length(blockids(dset)), 100)
  expect_equal(get_run_duration(dset), 200) # TR=2, 100 timepoints
  expect_equal(get_total_duration(dset), 200)
  expect_equal(length(samples(dset)), 100)

  # Verify all delegate to sampling_frame
  expect_equal(get_TR(dset), get_TR(dset$sampling_frame))
  expect_equal(n_timepoints(dset), n_timepoints(dset$sampling_frame))
  expect_equal(n_runs(dset), n_runs(dset$sampling_frame))
})

test_that("fmri_file_dataset with multiple runs delegates correctly", {
  # Create fmri_dataset with 2 runs
  dset <- fmri_dataset(
    scans = c("dummy1.nii"),
    mask = "dummy_mask.nii",
    TR = 2.0,
    run_length = c(50, 50),
    dummy_mode = TRUE
  )

  # Test multi-run delegation
  expect_equal(n_runs(dset), 2)
  expect_equal(get_run_lengths(dset), c(50, 50))
  expect_equal(blocklens(dset), c(50, 50))
  expect_equal(get_run_duration(dset), c(100, 100))
})

# ============================================
# fmri_study_dataset delegation tests
# ============================================

test_that("fmri_study_dataset n_runs implementation", {
  # Create two matrix_datasets with 2 runs each
  mat1 <- matrix(rnorm(200), nrow = 20, ncol = 10)
  dset1 <- matrix_dataset(mat1, TR = 2, run_length = c(10, 10))

  mat2 <- matrix(rnorm(200), nrow = 20, ncol = 10)
  dset2 <- matrix_dataset(mat2, TR = 2, run_length = c(10, 10))

  # Create study dataset
  study_dset <- fmri_study_dataset(
    list(dset1, dset2),
    subject_ids = c("S01", "S02")
  )

  # n_runs.fmri_study_dataset delegates to sampling_frame
  expect_equal(n_runs(study_dset), 4)
})

test_that("fmri_study_dataset subject_ids method works", {
  # Create study dataset
  mat1 <- matrix(rnorm(100), nrow = 10, ncol = 10)
  dset1 <- matrix_dataset(mat1, TR = 2, run_length = 10)

  mat2 <- matrix(rnorm(100), nrow = 10, ncol = 10)
  dset2 <- matrix_dataset(mat2, TR = 2, run_length = 10)

  study_dset <- fmri_study_dataset(
    list(dset1, dset2),
    subject_ids = c("S01", "S02")
  )

  # Verify subject_ids method returns correct IDs
  expect_equal(subject_ids(study_dset), c("S01", "S02"))
})

test_that("fmri_study_dataset delegation to sampling_frame works", {
  # Create study dataset
  mat1 <- matrix(rnorm(300), nrow = 30, ncol = 10)
  dset1 <- matrix_dataset(mat1, TR = 2, run_length = c(15, 15))

  mat2 <- matrix(rnorm(300), nrow = 30, ncol = 10)
  dset2 <- matrix_dataset(mat2, TR = 2, run_length = c(15, 15))

  study_dset <- fmri_study_dataset(
    list(dset1, dset2),
    subject_ids = c("S01", "S02")
  )

  # Test delegation methods
  expect_equal(get_TR(study_dset), 2)
  expect_equal(n_timepoints(study_dset), 60) # 30 per subject × 2 subjects
  expect_equal(get_run_lengths(study_dset), c(15, 15, 15, 15))
  expect_equal(blocklens(study_dset), c(15, 15, 15, 15))
  expect_equal(length(blockids(study_dset)), 60)
  expect_equal(get_run_duration(study_dset), c(30, 30, 30, 30))
  expect_equal(get_total_duration(study_dset), 120)
  expect_equal(length(samples(study_dset)), 60)

  # Verify delegation to sampling_frame
  expect_equal(get_TR(study_dset), get_TR(study_dset$sampling_frame))
  expect_equal(n_timepoints(study_dset), n_timepoints(study_dset$sampling_frame))
})

# ============================================
# Edge cases and consistency checks
# ============================================

test_that("all dataset classes have consistent delegation", {
  # Create datasets of each type
  mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
  mat_dset <- matrix_dataset(mat, TR = 2, run_length = 10)

  file_dset <- fmri_dataset(
    scans = "dummy1.nii",
    mask = "dummy_mask.nii",
    TR = 2,
    run_length = 100,
    dummy_mode = TRUE
  )

  # All should have the same methods available
  methods_to_test <- c(
    "get_TR", "n_timepoints", "n_runs", "get_run_lengths",
    "blocklens", "blockids", "get_run_duration",
    "get_total_duration", "samples"
  )

  # Methods are defined on fmri_dataset and inherited by all subclasses
  for (method_name in methods_to_test) {
    expect_true(
      any(grepl(method_name, methods(class = "fmri_dataset"))),
      info = paste(method_name, "should exist for fmri_dataset")
    )
  }

  # Verify methods actually work on each subclass via inheritance
  for (method_name in methods_to_test) {
    fn <- match.fun(method_name)
    expect_no_error(fn(mat_dset))
    expect_no_error(fn(file_dset))
  }
})

test_that("delegation preserves sampling_frame values exactly", {
  # Create dataset
  test_matrix <- matrix(rnorm(500), nrow = 50, ncol = 10)
  dset <- matrix_dataset(test_matrix, TR = 2.5, run_length = c(20, 30))

  # Extract sampling_frame
  sf <- dset$sampling_frame

  # Verify ALL delegated methods return identical values
  expect_identical(get_TR(dset), get_TR(sf))
  expect_identical(n_timepoints(dset), n_timepoints(sf))
  expect_identical(n_runs(dset), n_runs(sf))
  expect_identical(get_run_lengths(dset), get_run_lengths(sf))
  expect_identical(blocklens(dset), blocklens(sf))
  expect_identical(blockids(dset), blockids(sf))
  expect_identical(get_run_duration(dset), get_run_duration(sf))
  expect_identical(get_total_duration(dset), get_total_duration(sf))
  expect_identical(samples(dset), samples(sf))
})
