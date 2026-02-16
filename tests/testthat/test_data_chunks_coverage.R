# Tests for R/data_chunks.R - coverage improvement

test_that("data_chunk constructor creates valid object", {
  mat <- matrix(rnorm(50), nrow = 5, ncol = 10)
  chunk <- data_chunk(mat, voxel_ind = 1:10, row_ind = 1:5, chunk_num = 1)

  expect_s3_class(chunk, "data_chunk")
  expect_equal(chunk$data, mat)
  expect_equal(chunk$voxel_ind, 1:10)
  expect_equal(chunk$row_ind, 1:5)
  expect_equal(chunk$chunk_num, 1)
})

test_that("data_chunks.matrix_dataset with nchunks=1 returns all data", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  iter <- data_chunks(ds, nchunks = 1)
  chunks <- collect_chunks(iter)

  expect_length(chunks, 1)
  expect_equal(dim(chunks[[1]]$data), c(20, 10))
})

test_that("data_chunks.matrix_dataset with multiple chunks", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  iter <- data_chunks(ds, nchunks = 3)
  chunks <- collect_chunks(iter)

  expect_length(chunks, 3)
  # All voxels covered
  all_voxels <- sort(unlist(lapply(chunks, function(c) c$voxel_ind)))
  expect_equal(all_voxels, 1:10)
})

test_that("data_chunks.matrix_dataset runwise creates run chunks", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  iter <- data_chunks(ds, runwise = TRUE)
  chunks <- collect_chunks(iter)

  expect_length(chunks, 2) # 2 runs
  expect_equal(nrow(chunks[[1]]$data), 10)
  expect_equal(nrow(chunks[[2]]$data), 10)
})

test_that("data_chunks.matrix_dataset warns when nchunks > voxels", {
  mat <- matrix(rnorm(20), nrow = 5, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 5)

  expect_warning(
    iter <- data_chunks(ds, nchunks = 10),
    "greater than number of voxels"
  )
})

test_that("collect_chunks collects all from iterator", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  iter <- data_chunks(ds, nchunks = 2)
  chunks <- collect_chunks(iter)

  expect_length(chunks, 2)
  expect_s3_class(chunks[[1]], "data_chunk")
})

test_that("exec_strategy runwise", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  strat <- exec_strategy("runwise")
  iter <- strat(ds)
  expect_s3_class(iter, "chunkiter")
  expect_equal(iter$nchunks, 2)
})

test_that("exec_strategy chunkwise", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  strat <- exec_strategy("chunkwise", nchunks = 3)
  iter <- strat(ds)
  expect_s3_class(iter, "chunkiter")
})

test_that("exec_strategy voxelwise creates one chunk per voxel", {
  mat <- matrix(rnorm(60), nrow = 6, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 6)

  strat <- exec_strategy("voxelwise")
  iter <- strat(ds)
  expect_equal(iter$nchunks, 10)
})

test_that("exec_strategy chunkwise warns when nchunks > voxels", {
  mat <- matrix(rnorm(20), nrow = 5, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 5)

  strat <- exec_strategy("chunkwise", nchunks = 10)
  expect_warning(iter <- strat(ds), "greater than number of voxels")
})

test_that("chunk_iter stops iteration after all chunks", {
  mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 10)

  iter <- data_chunks(ds, nchunks = 2)
  iter$nextElem() # chunk 1
  iter$nextElem() # chunk 2
  expect_error(iter$nextElem(), "StopIteration")
})

test_that("data_chunks.matrix_dataset runwise with 3 runs", {
  mat <- matrix(rnorm(300), nrow = 30, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10, 10))

  iter <- data_chunks(ds, runwise = TRUE)
  chunks <- collect_chunks(iter)

  expect_length(chunks, 3)
  expect_equal(chunks[[1]]$chunk_num, 1)
  expect_equal(chunks[[2]]$chunk_num, 2)
  expect_equal(chunks[[3]]$chunk_num, 3)
})
