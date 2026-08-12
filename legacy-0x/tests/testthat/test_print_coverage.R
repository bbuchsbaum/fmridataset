# Tests for R/print_methods.R - coverage improvement

test_that("print.data_chunk prints correctly", {
  mat <- matrix(rnorm(50), nrow = 5, ncol = 10)
  chunk <- data_chunk(mat, voxel_ind = 1:10, row_ind = 1:5, chunk_num = 1)

  out <- capture.output(print(chunk))
  expect_true(any(grepl("Data Chunk Object", out)))
  expect_true(any(grepl("voxels", out)))
  expect_true(any(grepl("rows", out)))
  expect_true(any(grepl("dimensions", out)))
})

test_that("print.data_chunk with chunkid field", {
  chunk <- list(
    data = matrix(1:20, nrow = 4, ncol = 5),
    voxel_ind = 1:5,
    row_ind = 1:4,
    chunkid = 2,
    nchunks = 5
  )
  class(chunk) <- c("data_chunk", "list")

  out <- capture.output(print(chunk))
  expect_true(any(grepl("chunk 2 of 5", out)))
})

test_that("print.data_chunk without dim on data", {
  chunk <- list(
    data = rnorm(20),
    voxel_ind = 1:5,
    row_ind = 1:4,
    chunk_num = 1
  )
  class(chunk) <- c("data_chunk", "list")

  out <- capture.output(print(chunk))
  expect_true(any(grepl("Data length", out)))
})

test_that("print.fmri_dataset prints correctly for matrix_dataset", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  out <- capture.output(print(ds))
  expect_true(any(grepl("fMRI Dataset", out)))
  expect_true(any(grepl("Timepoints", out)))
  expect_true(any(grepl("TR", out)))
})

test_that("print.fmri_dataset full mode shows mask", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  out <- capture.output(print(ds, full = TRUE))
  expect_true(any(grepl("Voxels in mask", out)))
})

test_that("print.fmri_dataset with empty event table", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  out <- capture.output(print(ds))
  expect_true(any(grepl("Empty event table", out)))
})

test_that("print.fmri_dataset with event table", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  events <- data.frame(onset = c(0, 10), duration = c(5, 5), trial_type = c("A", "B"))
  ds <- matrix_dataset(
    datamat = mat, TR = 2, run_length = c(10, 10),
    event_table = events
  )

  out <- capture.output(print(ds))
  expect_true(any(grepl("Rows:", out)))
  expect_true(any(grepl("Variables:", out)))
})

test_that("print.fmri_dataset with many runs truncates", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = rep(1, 20))

  out <- capture.output(print(ds))
  expect_true(any(grepl("runs total", out)))
})

test_that("summary.fmri_dataset works", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  events <- data.frame(
    onset = c(0, 10), duration = c(5, 5),
    trial_type = c("A", "B")
  )
  ds <- matrix_dataset(
    datamat = mat, TR = 2, run_length = c(10, 10),
    event_table = events
  )

  out <- capture.output(summary(ds))
  expect_true(any(grepl("Summary", out)))
  expect_true(any(grepl("Trial types", out)))
})

test_that("print.chunkiter works", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  iter <- data_chunks(ds, nchunks = 3)

  out <- capture.output(print(iter))
  expect_true(any(grepl("Chunk Iterator", out)))
  expect_true(any(grepl("nchunks", out)))
})
