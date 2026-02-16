# Tests for R/FmriSeries.R nrow/ncol and R/fmri_series.R

test_that("new_fmri_series constructs valid object", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)
  selection_info <- list(selector = NULL, timepoints = NULL)
  dataset_info <- list(backend_type = "matrix_backend")

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = selection_info,
    dataset_info = dataset_info
  )

  expect_s3_class(fs, "fmri_series")
  expect_true(is.fmri_series(fs))
  expect_false(is.fmri_series(42))
})

test_that("nrow.fmri_series returns timepoints", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(),
    dataset_info = list()
  )

  expect_equal(nrow(fs), 4)
})

test_that("ncol.fmri_series returns voxels", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(),
    dataset_info = list()
  )

  expect_equal(ncol(fs), 5)
})

test_that("dim.fmri_series returns dimensions", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(),
    dataset_info = list()
  )

  expect_equal(dim(fs), c(4, 5))
})

test_that("as.matrix.fmri_series materializes data", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(),
    dataset_info = list()
  )

  result <- as.matrix(fs)
  expect_true(is.matrix(result))
  expect_equal(dim(result), c(4, 5))
  expect_equal(result, mat)
})

test_that("as_tibble.fmri_series creates tibble", {
  mat <- matrix(1:12, nrow = 3, ncol = 4)
  voxel_info <- data.frame(voxel = 1:4)
  temporal_info <- data.frame(time = 1:3)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(),
    dataset_info = list()
  )

  tbl <- tibble::as_tibble(fs)
  expect_s3_class(tbl, "tbl_df")
  expect_equal(nrow(tbl), 12) # 3 * 4
  expect_true("signal" %in% names(tbl))
  expect_true("time" %in% names(tbl))
  expect_true("voxel" %in% names(tbl))
})

test_that("print.fmri_series shows summary", {
  mat <- matrix(rnorm(20), nrow = 4, ncol = 5)
  voxel_info <- data.frame(voxel = 1:5)
  temporal_info <- data.frame(time = 1:4)

  fs <- fmridataset:::new_fmri_series(
    data = mat,
    voxel_info = voxel_info,
    temporal_info = temporal_info,
    selection_info = list(selector = "custom_sel"),
    dataset_info = list(backend_type = "matrix_backend")
  )

  out <- capture.output(print(fs))
  expect_true(any(grepl("fmri_series", out)))
  expect_true(any(grepl("5 voxels", out)))
  expect_true(any(grepl("4 timepoints", out)))
  expect_true(any(grepl("matrix_backend", out)))
})

test_that("fmri_series.fmri_dataset works with matrix_backend", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)
  dataset <- fmri_dataset(backend, TR = 2, run_length = c(10, 10))

  fs <- fmri_series(dataset)
  expect_s3_class(fs, "fmri_series")
  expect_equal(nrow(fs), 20)
  expect_equal(ncol(fs), 10)
})

test_that("fmri_series.fmri_dataset with index_selector", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)
  dataset <- fmri_dataset(backend, TR = 2, run_length = c(10, 10))

  sel <- index_selector(1:5)
  fs <- fmri_series(dataset, selector = sel)
  expect_equal(ncol(fs), 5)
})

test_that("fmri_series.fmri_dataset with timepoints", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)
  dataset <- fmri_dataset(backend, TR = 2, run_length = c(10, 10))

  fs <- fmri_series(dataset, timepoints = 1:5)
  expect_equal(nrow(fs), 5)
})
