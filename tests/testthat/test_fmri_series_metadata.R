library(testthat)
library(fmridataset)

# Helper function to create test dataset
create_test_dataset <- function() {
  mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
  backend <- matrix_backend(mat, spatial_dims = c(2, 5, 1))
  fmri_dataset(backend, TR = 2, run_length = 10)
}

# Helper function to create test study dataset
create_test_study_dataset <- function() {
  # Create two individual datasets
  mat1 <- matrix(rnorm(50), nrow = 5, ncol = 10)
  mat2 <- matrix(rnorm(60), nrow = 6, ncol = 10)

  backend1 <- matrix_backend(mat1, spatial_dims = c(2, 5, 1))
  backend2 <- matrix_backend(mat2, spatial_dims = c(2, 5, 1))

  dset1 <- fmri_dataset(backend1, TR = 2, run_length = 5)
  dset2 <- fmri_dataset(backend2, TR = 2, run_length = 6)

  # Create study dataset
  fmri_study_dataset(list(dset1, dset2), subject_ids = c("subj01", "subj02"))
}

test_that("build_temporal_info_lazy works for fmri_dataset", {
  dset <- create_test_dataset()

  # Test with all timepoints
  time_indices <- 1:10
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, time_indices)

  expect_s3_class(temporal_info, "data.frame")
  expect_equal(nrow(temporal_info), 10)
  expect_true("run_id" %in% colnames(temporal_info))
  expect_true("timepoint" %in% colnames(temporal_info))

  # All timepoints should be in run 1 for single-run dataset
  expect_equal(temporal_info$run_id, rep(1, 10))
  expect_equal(temporal_info$timepoint, 1:10)
})

test_that("build_temporal_info_lazy works with subset of timepoints", {
  dset <- create_test_dataset()

  # Test with subset of timepoints
  time_indices <- c(2, 5, 8)
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, time_indices)

  expect_equal(nrow(temporal_info), 3)
  expect_equal(temporal_info$timepoint, c(2, 5, 8))
  expect_equal(temporal_info$run_id, rep(1, 3))
})

test_that("build_temporal_info_lazy works for multi-run fmri_dataset", {
  # Create multi-run dataset
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  backend <- matrix_backend(mat, spatial_dims = c(2, 5, 1))
  dset <- fmri_dataset(backend, TR = 2, run_length = c(8, 12)) # Two runs

  time_indices <- 1:20
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, time_indices)

  expect_equal(nrow(temporal_info), 20)

  # Check run assignments
  expected_run_ids <- c(rep(1, 8), rep(2, 12))
  expect_equal(temporal_info$run_id, expected_run_ids)
})

test_that("build_temporal_info_lazy works for fmri_study_dataset", {
  study_dset <- create_test_study_dataset()

  # Test with all timepoints (5 + 6 = 11 total)
  time_indices <- 1:11
  temporal_info <- fmridataset:::build_temporal_info_lazy(study_dset, time_indices)

  expect_s3_class(temporal_info, "data.frame")
  expect_equal(nrow(temporal_info), 11)
  expect_true("subject_id" %in% colnames(temporal_info))
  expect_true("run_id" %in% colnames(temporal_info))
  expect_true("timepoint" %in% colnames(temporal_info))

  # Check subject assignments
  expected_subjects <- c(rep("subj01", 5), rep("subj02", 6))
  expect_equal(temporal_info$subject_id, expected_subjects)

  # Check run assignments
  expected_runs <- c(rep(1, 5), rep(2, 6))
  expect_equal(temporal_info$run_id, expected_runs)

  # Check timepoints
  expect_equal(temporal_info$timepoint, 1:11)
})

test_that("build_temporal_info_lazy handles subset timepoints for study dataset", {
  study_dset <- create_test_study_dataset()

  # Test with subset spanning both subjects
  time_indices <- c(3, 7, 10) # subj01 timepoint 3, subj02 timepoints 2 and 5
  temporal_info <- fmridataset:::build_temporal_info_lazy(study_dset, time_indices)

  expect_equal(nrow(temporal_info), 3)
  expect_equal(temporal_info$timepoint, c(3, 7, 10))
  expect_equal(temporal_info$subject_id, c("subj01", "subj02", "subj02"))
})

test_that("build_temporal_info_lazy validates run length consistency", {
  # Create inconsistent study dataset - this should be caught by build function
  mat1 <- matrix(rnorm(50), nrow = 5, ncol = 10)
  mat2 <- matrix(rnorm(60), nrow = 6, ncol = 10)

  backend1 <- matrix_backend(mat1, spatial_dims = c(2, 5, 1))
  backend2 <- matrix_backend(mat2, spatial_dims = c(2, 5, 1))

  # Create backends list directly to simulate inconsistency
  backends <- list(backend1, backend2)
  study_backend <- structure(
    list(backends = backends),
    class = c("study_backend", "storage_backend")
  )

  # Create sampling frame with wrong run lengths
  wrong_frame <- fmrihrf::sampling_frame(blocklens = c(3, 4), TR = 2) # Wrong totals

  study_dset <- list(
    backend = study_backend,
    sampling_frame = wrong_frame,
    subject_ids = c("subj01", "subj02")
  )
  class(study_dset) <- c("fmri_study_dataset", "fmri_dataset", "list")

  # This should trigger the consistency check error
  expect_error(
    fmridataset:::build_temporal_info_lazy(study_dset, 1:7),
    "run lengths inconsistent with backend dimensions"
  )
})

test_that("build_temporal_info_lazy edge cases", {
  dset <- create_test_dataset()

  # Test with empty timepoints vector
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, integer(0))
  expect_equal(nrow(temporal_info), 0)
  expect_true("run_id" %in% colnames(temporal_info))
  expect_true("timepoint" %in% colnames(temporal_info))

  # Test with single timepoint
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, 5)
  expect_equal(nrow(temporal_info), 1)
  expect_equal(temporal_info$timepoint, 5)
  expect_equal(temporal_info$run_id, 1)
})

test_that("build_temporal_info_lazy returns data.frame", {
  dset <- create_test_dataset()
  temporal_info <- fmridataset:::build_temporal_info_lazy(dset, 1:5)

  expect_s3_class(temporal_info, "data.frame")
  expect_true(is.data.frame(temporal_info))
})
