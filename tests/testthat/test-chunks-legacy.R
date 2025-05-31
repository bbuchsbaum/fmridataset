# Legacy Chunk Tests from fmrireg
# These tests ensure backward compatibility with the original fmrireg data chunking interface

library(foreach)

test_that("matrix_dataset chunking works correctly", {
  # Create test data similar to fmrireg's matrix_dataset
  n_time <- 100
  n_vox <- 10
  n_runs <- 2
  
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  run_length <- rep(n_time/n_runs, n_runs)
  
  # Use our new matrix_dataset function (fmrireg compatible)
  dset <- matrix_dataset(Y, TR = 1, run_length = run_length)
  
  # Test runwise chunking (equivalent to runwise = TRUE in fmrireg)
  chunks <- data_chunks(dset, by = "run")
  expect_s3_class(chunks, "fmri_chunk_iterator")
  
  # Should have 2 chunks (one per run)
  expect_equal(attr(chunks, "total_chunks"), 2)
  
  # Collect all chunks using foreach-compatible iteration
  chunk_list <- list()
  chunk_count <- 0
  for (chunk in chunks) {
    chunk_count <- chunk_count + 1
    chunk_list[[chunk_count]] <- chunk
  }
  
  expect_equal(length(chunk_list), n_runs)
  
  # Check first chunk structure
  chunk1 <- chunk_list[[1]]
  expect_s3_class(chunk1, "fmri_data_chunk")
  expect_true(all(c("data", "voxel_indices", "timepoint_indices", "chunk_num") %in% names(chunk1)))
  
  # Check dimensions
  expect_equal(nrow(chunk1$data), n_time/n_runs)
  expect_equal(ncol(chunk1$data), n_vox)
  expect_equal(chunk1$chunk_num, 1)
  expect_equal(chunk1$timepoint_indices, 1:(n_time/n_runs))
})

test_that("matrix_dataset single chunk works", {
  n_time <- 50
  n_vox <- 5
  
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  
  # Use matrix_dataset function
  dset <- matrix_dataset(Y, TR = 1, run_length = n_time)
  
  chunks <- data_chunks(dset, nchunks = 1)
  chunk <- chunks$nextElem()
  
  expect_s3_class(chunk, "fmri_data_chunk")
  expect_equal(dim(chunk$data), dim(Y))
  expect_equal(chunk$chunk_num, 1)
  expect_equal(chunk$voxel_indices, 1:n_vox)
})

test_that("matrix_dataset voxel chunking works", {
  n_time <- 50
  n_vox = 20
  
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  
  # Use matrix_dataset function
  dset <- matrix_dataset(Y, TR = 1, run_length = n_time)
  
  # Split into 4 chunks by voxel (default behavior)
  chunks <- data_chunks(dset, nchunks = 4, by = "voxel")
  expect_equal(attr(chunks, "total_chunks"), 4)
  
  chunk_list <- list()
  chunk_count <- 0
  for (chunk in chunks) {
    chunk_count <- chunk_count + 1
    chunk_list[[chunk_count]] <- chunk
  }
  
  expect_equal(length(chunk_list), 4)
  
  # Check that all voxels are covered
  all_vox_ind <- unlist(lapply(chunk_list, function(ch) ch$voxel_indices))
  expect_equal(sort(all_vox_ind), 1:n_vox)
  
  # Check chunk dimensions
  for (i in 1:4) {
    expect_equal(nrow(chunk_list[[i]]$data), n_time)
    expect_true(ncol(chunk_list[[i]]$data) > 0)
    expect_equal(chunk_list[[i]]$chunk_num, i)
  }
})

test_that("data_chunk object has correct structure", {
  # Test the structure of our fmri_data_chunk vs old data_chunk
  mat <- matrix(1:12, 3, 4)
  
  # Create a chunk manually using our constructor
  chunk <- fmri_data_chunk(
    data = mat, 
    voxel_indices = 1:4, 
    timepoint_indices = 1:3, 
    chunk_num = 1,
    total_chunks = 1
  )
  
  expect_s3_class(chunk, "fmri_data_chunk")
  expect_identical(chunk$data, mat)
  expect_equal(chunk$voxel_indices, 1:4)
  expect_equal(chunk$timepoint_indices, 1:3)
  expect_equal(chunk$chunk_num, 1)
})

test_that("chunk iterator interface compatibility", {
  # Test compatibility with fmrireg's chunk iterator interface
  
  n_time <- 80
  n_vox <- 12
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  dset <- matrix_dataset(Y, TR = 2, run_length = c(40, 40))
  
  # Test that chunks can be accessed in multiple ways
  chunks <- data_chunks(dset, nchunks = 3, by = "voxel")
  
  # Method 1: Direct nextElem() access (fmrireg style)
  chunk1 <- chunks$nextElem()
  chunk2 <- chunks$nextElem()
  
  expect_s3_class(chunk1, "fmri_data_chunk")
  expect_s3_class(chunk2, "fmri_data_chunk")
  expect_equal(chunk1$chunk_num, 1)
  expect_equal(chunk2$chunk_num, 2)
  
  # Method 2: foreach loop (should reset and iterate from beginning)
  chunk_nums <- foreach(chunk = chunks) %do% {
    chunk$chunk_num
  }
  
  expect_equal(unlist(chunk_nums), c(1, 2, 3))
})

test_that("chunking preserves data integrity", {
  # Test that chunking doesn't lose or modify data
  
  set.seed(12345)
  n_time <- 60
  n_vox <- 15
  original_data <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  
  dset <- matrix_dataset(original_data, TR = 1.5, run_length = 60)
  
  # Chunk by voxels
  chunks <- data_chunks(dset, nchunks = 5, by = "voxel")
  
  # Reconstruct data from chunks
  reconstructed_data <- matrix(0, n_time, n_vox)
  for (chunk in chunks) {
    reconstructed_data[, chunk$voxel_indices] <- chunk$data
  }
  
  # Should be identical
  expect_equal(reconstructed_data, original_data)
})

test_that("run-wise chunking works like fmrireg", {
  # Test run-wise chunking similar to fmrireg's runwise = TRUE
  
  n_time <- 120
  n_vox <- 8
  run_lengths <- c(40, 50, 30)
  
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  dset <- matrix_dataset(Y, TR = 2, run_length = run_lengths)
  
  # Run-wise chunking
  chunks <- data_chunks(dset, by = "run")
  
  chunk_list <- list()
  for (chunk in chunks) {
    chunk_list[[length(chunk_list) + 1]] <- chunk
  }
  
  expect_equal(length(chunk_list), 3)  # One chunk per run
  
  # Check run lengths match
  expect_equal(nrow(chunk_list[[1]]$data), 40)
  expect_equal(nrow(chunk_list[[2]]$data), 50)
  expect_equal(nrow(chunk_list[[3]]$data), 30)
  
  # Check timepoint indices
  expect_equal(chunk_list[[1]]$timepoint_indices, 1:40)
  expect_equal(chunk_list[[2]]$timepoint_indices, 41:90)
  expect_equal(chunk_list[[3]]$timepoint_indices, 91:120)
})

test_that("legacy chunking parameter compatibility", {
  # Test that our interface handles fmrireg-style parameters
  
  n_time <- 100
  n_vox <- 20
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  
  dset <- matrix_dataset(Y, TR = 1, run_length = c(50, 50))
  
  # Test various chunking scenarios that fmrireg supported
  
  # Scenario 1: By number of chunks
  chunks1 <- data_chunks(dset, nchunks = 4)
  expect_equal(attr(chunks1, "total_chunks"), 4)
  
  # Scenario 2: By run (equivalent to runwise = TRUE)
  chunks2 <- data_chunks(dset, by = "run")
  expect_equal(attr(chunks2, "total_chunks"), 2)
  
  # Scenario 3: Single chunk (all data)
  chunks3 <- data_chunks(dset, nchunks = 1)
  chunk_all <- chunks3$nextElem()
  expect_equal(dim(chunk_all$data), c(n_time, n_vox))
}) 