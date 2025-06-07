context("chunk utilities")

library(fmridataset)

## tests for arbitrary_chunks and one_chunk

test_that("arbitrary_chunks handles too many chunks", {
  Y <- matrix(1:20, nrow = 10, ncol = 2)
  dset <- matrix_dataset(Y, TR = 1, run_length = 10)
  expect_warning(
    ch <- fmridataset:::arbitrary_chunks(dset, 5),
    "greater than number of voxels"
  )
  expect_length(ch, 2)
  expect_equal(sort(unique(unlist(ch))), 1:2)
})

test_that("one_chunk returns all voxel indices", {
  Y <- matrix(1:20, nrow = 10, ncol = 2)
  dset <- matrix_dataset(Y, TR = 1, run_length = 10)
  oc <- fmridataset:::one_chunk(dset)
  expect_equal(oc[[1]], 1:2)
})

## tests for exec_strategy warnings and print methods

test_that("exec_strategy handles large requested chunks", {
  Y <- matrix(1:20, nrow = 10, ncol = 2)
  dset <- matrix_dataset(Y, TR = 1, run_length = 10)
  strat <- fmridataset:::exec_strategy("chunkwise", nchunks = 5)
  expect_warning(iter <- strat(dset), "greater than number of voxels")
  expect_equal(iter$nchunks, 2)
})

test_that("print methods for chunk objects work", {
  Y <- matrix(rnorm(20), nrow = 10, ncol = 2)
  dset <- matrix_dataset(Y, TR = 1, run_length = 10)
  iter <- data_chunks(dset, nchunks = 1)
  chunk <- iter$nextElem()
  skip_if_not_installed("crayon")
  expect_output(print(iter), "Chunk Iterator")
  expect_output(print(chunk), "Data Chunk Object")
})
