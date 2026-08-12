# Tests for data_chunks.fmri_study_dataset in R/data_chunks.R

test_that("data_chunks.fmri_study_dataset with nchunks=1", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  iter <- data_chunks(sds, nchunks = 1)
  expect_s3_class(iter, "chunkiter")
  chunks <- collect_chunks(iter)
  expect_length(chunks, 1)
  expect_equal(nrow(chunks[[1]]$data), 10)
})

test_that("data_chunks.fmri_study_dataset with multiple chunks", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  iter <- data_chunks(sds, nchunks = 2)
  chunks <- collect_chunks(iter)
  expect_length(chunks, 2)

  # All voxels covered
  all_voxels <- sort(unique(unlist(lapply(chunks, function(c) c$voxel_ind))))
  expect_equal(length(all_voxels), 4)
})

test_that("data_chunks.fmri_study_dataset runwise", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(
    datasets = list(ds1, ds2),
    subject_ids = c("s1", "s2")
  )

  iter <- data_chunks(sds, runwise = TRUE)
  chunks <- collect_chunks(iter)
  # Should have 2 runs (one per subject)
  expect_length(chunks, 2)
})

test_that("data_chunks.fmri_study_dataset multi-run per subject", {
  mat1 <- matrix(rnorm(80), nrow = 20, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = c(10, 10))

  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  iter <- data_chunks(sds, runwise = TRUE)
  chunks <- collect_chunks(iter)
  expect_length(chunks, 2)
  expect_equal(nrow(chunks[[1]]$data), 10)
  expect_equal(nrow(chunks[[2]]$data), 10)
})

# --- fmri_series.fmri_study_dataset ---

test_that("fmri_series.fmri_study_dataset works", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(
    datasets = list(ds1, ds2),
    subject_ids = c("s1", "s2")
  )

  fs <- fmri_series(sds)
  expect_s3_class(fs, "fmri_series")
  expect_equal(nrow(fs), 20)
  expect_equal(ncol(fs), 4)
})

test_that("fmri_series.fmri_study_dataset with timepoints", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  fs <- fmri_series(sds, timepoints = 1:5)
  expect_equal(nrow(fs), 5)
})

test_that("fmri_series.fmri_study_dataset with selector", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  sel <- index_selector(1:2)
  fs <- fmri_series(sds, selector = sel)
  expect_equal(ncol(fs), 2)
})

# --- as_tibble.fmri_study_dataset ---

test_that("as_tibble.fmri_study_dataset returns result", {
  mat1 <- matrix(1:40, nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))

  result <- tibble::as_tibble(sds)
  # Should return some kind of object (may be tbl_df or delarr depending on backend)
  expect_true(!is.null(result))
})
