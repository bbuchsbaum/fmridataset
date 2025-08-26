library(testthat)
library(fmridataset)

# Helper functions for testing
create_test_matrix_dataset <- function(n_time = 100, n_voxels = 50, n_runs = 1) {
  mat <- matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels)
  run_lengths <- if (n_runs == 1) n_time else rep(n_time %/% n_runs, n_runs)
  matrix_dataset(mat, TR = 2, run_length = run_lengths)
}

test_that("data_chunk constructor creates valid objects", {
  mat <- matrix(1:20, nrow = 4, ncol = 5)
  voxel_ind <- 1:5
  row_ind <- 1:4
  chunk_num <- 1

  chunk <- fmridataset:::data_chunk(mat, voxel_ind, row_ind, chunk_num)

  expect_s3_class(chunk, "data_chunk")
  expect_s3_class(chunk, "list")
  expect_equal(chunk$data, mat)
  expect_equal(chunk$voxel_ind, voxel_ind)
  expect_equal(chunk$row_ind, row_ind)
  expect_equal(chunk$chunk_num, chunk_num)
})

test_that("chunk_iter creates valid iterator", {
  get_chunk_func <- function(i) {
    fmridataset:::data_chunk(
      matrix(i, nrow = 2, ncol = 2),
      1:2, 1:2, i
    )
  }

  iter <- fmridataset:::chunk_iter(x = NULL, nchunks = 3, get_chunk = get_chunk_func)

  expect_s3_class(iter, "chunkiter")
  expect_s3_class(iter, "abstractiter")
  expect_s3_class(iter, "iter")
  expect_equal(iter$nchunks, 3)

  # Test iteration
  chunk1 <- iter$nextElem()
  expect_equal(chunk1$chunk_num, 1)

  chunk2 <- iter$nextElem()
  expect_equal(chunk2$chunk_num, 2)

  chunk3 <- iter$nextElem()
  expect_equal(chunk3$chunk_num, 3)

  # Should throw StopIteration after exhaustion
  expect_error(iter$nextElem(), "StopIteration")
})

test_that("data_chunks.matrix_dataset creates correct number of chunks", {
  dset <- create_test_matrix_dataset(n_time = 50, n_voxels = 100)

  # Test single chunk
  iter1 <- data_chunks(dset, nchunks = 1)
  expect_equal(iter1$nchunks, 1)

  # Test multiple chunks
  iter4 <- data_chunks(dset, nchunks = 4)
  expect_equal(iter4$nchunks, 4)

  # Test requesting more chunks than voxels
  expect_warning(
    iter_many <- data_chunks(dset, nchunks = 200),
    "requested number of chunks.*is greater than number of voxels"
  )
})

test_that("data_chunks.matrix_dataset handles runwise chunking", {
  # Create multi-run dataset
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  dset <- matrix_dataset(mat, TR = 2, run_length = c(8, 12)) # Two runs

  iter <- data_chunks(dset, runwise = TRUE)
  expect_equal(iter$nchunks, 2) # Should have 2 chunks for 2 runs

  # Check first run chunk
  chunk1 <- iter$nextElem()
  expect_equal(chunk1$chunk_num, 1)
  expect_equal(length(chunk1$row_ind), 8) # First run has 8 timepoints
  expect_equal(ncol(chunk1$data), 10) # All voxels

  # Check second run chunk
  chunk2 <- iter$nextElem()
  expect_equal(chunk2$chunk_num, 2)
  expect_equal(length(chunk2$row_ind), 12) # Second run has 12 timepoints
  expect_equal(ncol(chunk2$data), 10) # All voxels
})

test_that("data_chunks.matrix_dataset voxel indices are distributed correctly", {
  dset <- create_test_matrix_dataset(n_time = 10, n_voxels = 20)

  iter <- data_chunks(dset, nchunks = 4)
  chunks <- collect_chunks(iter)

  # Check that all voxels are covered
  all_voxels <- sort(unlist(lapply(chunks, function(c) c$voxel_ind)))
  expect_equal(all_voxels, 1:20)

  # Check that chunks are roughly equal size
  chunk_sizes <- sapply(chunks, function(c) length(c$voxel_ind))
  expect_true(max(chunk_sizes) - min(chunk_sizes) <= 1)
})

test_that("arbitrary_chunks handles edge cases", {
  # Test with small dataset
  dset <- create_test_matrix_dataset(n_time = 5, n_voxels = 3)

  # Request more chunks than voxels
  expect_warning(
    chunks <- fmridataset:::arbitrary_chunks(dset, nchunks = 10),
    "requested number of chunks.*is greater than number of voxels"
  )

  # Should cap at number of voxels
  expect_equal(length(chunks), 3)

  # Test single voxel per chunk
  chunks_single <- fmridataset:::arbitrary_chunks(dset, nchunks = 3)
  expect_equal(length(chunks_single), 3)
  expect_true(all(sapply(chunks_single, length) == 1))
})

test_that("one_chunk includes all voxels", {
  dset <- create_test_matrix_dataset(n_time = 10, n_voxels = 25)

  chunks <- fmridataset:::one_chunk(dset)
  expect_equal(length(chunks), 1)
  expect_equal(length(chunks[[1]]), 25) # All voxels
  expect_equal(chunks[[1]], 1:25)
})

test_that("exec_strategy creates appropriate chunk iterators", {
  dset <- create_test_matrix_dataset(n_time = 20, n_voxels = 50)

  # Test voxelwise strategy
  strategy_voxel <- exec_strategy("voxelwise")
  iter_voxel <- strategy_voxel(dset)
  expect_equal(iter_voxel$nchunks, 50) # One chunk per voxel

  # Test runwise strategy for single run dataset
  strategy_run <- exec_strategy("runwise")
  iter_run <- strategy_run(dset)
  expect_equal(iter_run$nchunks, 1) # Single run

  # Test chunkwise strategy
  strategy_chunk <- exec_strategy("chunkwise", nchunks = 5)
  iter_chunk <- strategy_chunk(dset)
  expect_equal(iter_chunk$nchunks, 5)
})

test_that("exec_strategy validates nchunks parameter", {
  dset <- create_test_matrix_dataset()

  # Should require nchunks for chunkwise strategy
  expect_error(
    exec_strategy("chunkwise")(dset),
    "is not TRUE"
  )

  # Should warn if nchunks exceeds voxel count
  strategy <- exec_strategy("chunkwise", nchunks = 1000)
  expect_warning(
    strategy(dset),
    "requested number of chunks is greater than number of voxels"
  )
})

test_that("collect_chunks gathers all chunks from iterator", {
  dset <- create_test_matrix_dataset(n_time = 10, n_voxels = 12)

  iter <- data_chunks(dset, nchunks = 3)
  chunks <- collect_chunks(iter)

  expect_equal(length(chunks), 3)
  expect_true(all(sapply(chunks, function(c) inherits(c, "data_chunk"))))

  # Check chunk numbers are sequential
  chunk_nums <- sapply(chunks, function(c) c$chunk_num)
  expect_equal(chunk_nums, 1:3)
})

test_that("data chunking preserves data integrity", {
  # Create known data
  mat <- matrix(1:60, nrow = 6, ncol = 10) # Known values
  dset <- matrix_dataset(mat, TR = 2, run_length = 6)

  # Get single chunk (should include all data)
  iter <- data_chunks(dset, nchunks = 1)
  chunk <- iter$nextElem()

  expect_equal(chunk$data, mat)
  expect_equal(chunk$voxel_ind, 1:10)
  expect_equal(chunk$row_ind, 1:6)

  # Test multiple chunks preserve data when recombined
  iter2 <- data_chunks(dset, nchunks = 2)
  chunks <- collect_chunks(iter2)

  # Combine chunks back into original matrix
  combined_data <- do.call(cbind, lapply(chunks, function(c) c$data))
  combined_voxel_ind <- unlist(lapply(chunks, function(c) c$voxel_ind))

  # Should recover original data (possibly reordered by voxel)
  expect_equal(ncol(combined_data), 10)
  expect_equal(nrow(combined_data), 6)
  expect_equal(sort(combined_voxel_ind), 1:10)
})

test_that("chunking works with different data types", {
  # Test with integer data
  int_mat <- matrix(as.integer(1:50), nrow = 5, ncol = 10)
  dset_int <- matrix_dataset(int_mat, TR = 2, run_length = 5)

  iter_int <- data_chunks(dset_int, nchunks = 2)
  chunk_int <- iter_int$nextElem()
  expect_true(is.integer(chunk_int$data))

  # Test with sparse-like data (lots of zeros)
  sparse_mat <- matrix(0, nrow = 10, ncol = 20)
  sparse_mat[c(1, 3, 7), c(2, 8, 15)] <- c(1, 5, 9)
  dset_sparse <- matrix_dataset(sparse_mat, TR = 2, run_length = 10)

  iter_sparse <- data_chunks(dset_sparse, nchunks = 4)
  chunk_sparse <- iter_sparse$nextElem()
  expect_equal(nrow(chunk_sparse$data), 10)
})

test_that("chunking respects memory constraints", {
  # Test that chunking doesn't load entire dataset unnecessarily
  # This is more of a design test - chunking should work with large datasets

  # Simulate larger dataset
  dset <- create_test_matrix_dataset(n_time = 500, n_voxels = 1000)

  # Should be able to create iterator without loading all data
  iter <- data_chunks(dset, nchunks = 50)
  expect_equal(iter$nchunks, 50)

  # Should be able to get individual chunks
  chunk1 <- iter$nextElem()
  expect_true(ncol(chunk1$data) <= 1000 / 50 + 1) # Roughly 1/50th of voxels

  # Subsequent chunks should be independent
  chunk2 <- iter$nextElem()
  expect_false(identical(chunk1$voxel_ind, chunk2$voxel_ind))
})

test_that("chunk iterator state management works correctly", {
  dset <- create_test_matrix_dataset(n_time = 10, n_voxels = 15)

  iter <- data_chunks(dset, nchunks = 3)

  # Track progression through iterator
  chunks_retrieved <- 0
  while (chunks_retrieved < 3) {
    chunk <- iter$nextElem()
    chunks_retrieved <- chunks_retrieved + 1
    expect_equal(chunk$chunk_num, chunks_retrieved)
  }

  # Should be exhausted now
  expect_error(iter$nextElem(), "StopIteration")

  # Multiple calls to exhausted iterator should keep throwing
  expect_error(iter$nextElem(), "StopIteration")
  expect_error(iter$nextElem(), "StopIteration")
})

test_that("data chunking edge cases", {
  # Single voxel dataset
  single_voxel_mat <- matrix(rnorm(20), nrow = 20, ncol = 1)
  dset_single <- matrix_dataset(single_voxel_mat, TR = 2, run_length = 20)

  iter_single <- data_chunks(dset_single, nchunks = 1)
  chunk_single <- iter_single$nextElem()
  expect_equal(chunk_single$voxel_ind, 1)
  expect_equal(ncol(chunk_single$data), 1)

  # Single timepoint dataset
  single_time_mat <- matrix(rnorm(50), nrow = 1, ncol = 50)
  dset_time <- matrix_dataset(single_time_mat, TR = 2, run_length = 1)

  iter_time <- data_chunks(dset_time, nchunks = 5)
  chunk_time <- iter_time$nextElem()
  expect_equal(nrow(chunk_time$data), 1)
  expect_equal(chunk_time$row_ind, 1)
})
