test_that("data_chunks creates correct number of voxel chunks", {
  set.seed(123)
  test_matrix <- matrix(rnorm(2000), nrow = 100, ncol = 20)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Create 4 voxel chunks
  chunks <- data_chunks(dataset, nchunks = 4, by = "voxel")
  
  expect_s3_class(chunks, "fmri_chunk_iterator")
  
  # Collect all chunks
  chunk_list <- list()
  for (chunk in chunks) {
    chunk_list <- append(chunk_list, list(chunk))
  }
  
  expect_length(chunk_list, 4)
  
  # Check chunk properties
  for (i in 1:4) {
    expect_s3_class(chunk_list[[i]], "fmri_data_chunk")
    expect_equal(chunk_list[[i]]$chunk_num, i)
    expect_equal(chunk_list[[i]]$total_chunks, 4)
    expect_equal(nrow(chunk_list[[i]]$data), 100)  # Same number of timepoints
  }
  
  # Check that all voxels are covered
  all_voxel_indices <- unlist(lapply(chunk_list, function(x) x$voxel_indices))
  expect_equal(sort(all_voxel_indices), 1:20)
})

test_that("data_chunks creates correct number of timepoint chunks", {
  set.seed(123)
  test_matrix <- matrix(rnorm(2000), nrow = 100, ncol = 20)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Create 5 timepoint chunks
  chunks <- data_chunks(dataset, nchunks = 5, by = "timepoint")
  
  chunk_list <- list()
  for (chunk in chunks) {
    chunk_list <- append(chunk_list, list(chunk))
  }
  
  expect_length(chunk_list, 5)
  
  # Check chunk properties
  for (i in 1:5) {
    expect_s3_class(chunk_list[[i]], "fmri_data_chunk")
    expect_equal(ncol(chunk_list[[i]]$data), 20)  # Same number of voxels
  }
  
  # Check that all timepoints are covered
  all_timepoint_indices <- unlist(lapply(chunk_list, function(x) x$timepoint_indices))
  expect_equal(sort(all_timepoint_indices), 1:100)
})

test_that("data_chunks runwise creates one chunk per run", {
  set.seed(123)
  test_matrix <- matrix(rnorm(3000), nrow = 150, ncol = 20)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 60, 40)
  )
  
  # Create runwise chunks
  chunks <- data_chunks(dataset, runwise = TRUE)
  
  chunk_list <- list()
  for (chunk in chunks) {
    chunk_list <- append(chunk_list, list(chunk))
  }
  
  expect_length(chunk_list, 3)  # One per run
  
  # Check run-specific properties
  expect_equal(nrow(chunk_list[[1]]$data), 50)  # Run 1: 50 timepoints
  expect_equal(nrow(chunk_list[[2]]$data), 60)  # Run 2: 60 timepoints
  expect_equal(nrow(chunk_list[[3]]$data), 40)  # Run 3: 40 timepoints
  
  # All chunks should have same number of voxels
  for (chunk in chunk_list) {
    expect_equal(ncol(chunk$data), 20)
  }
})

test_that("data_chunks applies preprocessing correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    temporal_zscore = TRUE
  )
  
  # With preprocessing
  chunks_processed <- data_chunks(dataset, nchunks = 2, apply_preprocessing = TRUE)
  chunk1_processed <- chunks_processed$nextElem()
  
  # Without preprocessing
  chunks_raw <- data_chunks(dataset, nchunks = 2, apply_preprocessing = FALSE)
  chunk1_raw <- chunks_raw$nextElem()
  
  # Should be different due to preprocessing
  expect_false(identical(chunk1_processed$data, chunk1_raw$data))
})

test_that("data_chunks handles masking correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  chunks <- data_chunks(dataset, nchunks = 2, by = "voxel")
  chunk1 <- chunks$nextElem()
  
  # Should have only masked voxels
  expect_equal(ncol(chunk1$data), length(chunk1$voxel_indices))
  expect_true(all(chunk1$voxel_indices <= sum(mask_vector)))
})

test_that("data_chunks handles censoring correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  censor_vector <- rep(TRUE, 100)
  censor_vector[c(10:15, 60:65)] <- FALSE
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = censor_vector
  )
  
  chunks <- data_chunks(dataset, nchunks = 2, by = "voxel")
  chunk1 <- chunks$nextElem()
  
  # Should have only non-censored timepoints
  expect_equal(nrow(chunk1$data), sum(censor_vector))
})

test_that("fmri_data_chunk constructor works correctly", {
  test_data <- matrix(rnorm(200), nrow = 20, ncol = 10)
  voxel_indices <- 1:10
  
  chunk <- fmri_data_chunk(
    data = test_data,
    voxel_indices = voxel_indices,
    chunk_num = 1,
    total_chunks = 3
  )
  
  expect_s3_class(chunk, "fmri_data_chunk")
  expect_equal(chunk$data, test_data)
  expect_equal(chunk$voxel_indices, voxel_indices)
  expect_equal(chunk$chunk_num, 1)
  expect_equal(chunk$total_chunks, 3)
})

test_that("is.fmri_data_chunk works correctly", {
  test_data <- matrix(rnorm(200), nrow = 20, ncol = 10)
  
  chunk <- fmri_data_chunk(
    data = test_data,
    voxel_indices = 1:10,
    chunk_num = 1,
    total_chunks = 2
  )
  
  expect_true(is.fmri_data_chunk(chunk))
  expect_false(is.fmri_data_chunk(list(data = test_data)))
  expect_false(is.fmri_data_chunk(test_data))
  expect_false(is.fmri_data_chunk(NULL))
})

test_that("fmri_data_chunk print method works", {
  test_data <- matrix(rnorm(200), nrow = 20, ncol = 10)
  
  chunk <- fmri_data_chunk(
    data = test_data,
    voxel_indices = 1:10,
    chunk_num = 2,
    total_chunks = 5
  )
  
  expect_output(print(chunk), "fmri_data_chunk")
  expect_output(print(chunk), "Chunk 2 of 5")
  expect_output(print(chunk), "20.*10")  # Dimensions
})

test_that("data_chunks iterator is reusable", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  chunks <- data_chunks(dataset, nchunks = 3)
  
  # First iteration
  chunk_list1 <- list()
  for (chunk in chunks) {
    chunk_list1 <- append(chunk_list1, list(chunk))
  }
  
  # Second iteration should work
  chunk_list2 <- list()
  for (chunk in chunks) {
    chunk_list2 <- append(chunk_list2, list(chunk))
  }
  
  expect_length(chunk_list1, 3)
  expect_length(chunk_list2, 3)
  
  # Should get same chunks
  expect_equal(chunk_list1[[1]]$voxel_indices, chunk_list2[[1]]$voxel_indices)
})

test_that("data_chunks manual iteration works", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  iter <- data_chunks(dataset, nchunks = 2)
  
  # Manual step through
  chunk1 <- iter$nextElem()
  expect_s3_class(chunk1, "fmri_data_chunk")
  expect_equal(chunk1$chunk_num, 1)
  
  chunk2 <- iter$nextElem()
  expect_s3_class(chunk2, "fmri_data_chunk")
  expect_equal(chunk2$chunk_num, 2)
  
  # Should throw StopIteration error when exhausted
  expect_error(iter$nextElem(), "StopIteration")
})

test_that("data_chunks with different chunk sizes covers all data", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Test with different numbers of chunks
  for (nchunks in c(1, 2, 3, 5, 7)) {
    chunks <- data_chunks(dataset, nchunks = nchunks, by = "voxel")
    
    all_voxels <- c()
    chunk_count <- 0
    for (chunk in chunks) {
      chunk_count <- chunk_count + 1
      all_voxels <- c(all_voxels, chunk$voxel_indices)
    }
    
    expect_equal(chunk_count, nchunks)
    expect_equal(sort(all_voxels), 1:10)
  }
})

test_that("data_chunks edge cases work correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Single chunk
  chunks_single <- data_chunks(dataset, nchunks = 1)
  chunk_single <- chunks_single$nextElem()
  expect_equal(ncol(chunk_single$data), 10)
  expect_equal(nrow(chunk_single$data), 100)
  
  # More chunks than voxels
  chunks_many <- data_chunks(dataset, nchunks = 15, by = "voxel")
  chunk_list <- list()
  for (chunk in chunks_many) {
    chunk_list <- append(chunk_list, list(chunk))
  }
  
  # Should still only get 10 chunks (one per voxel max)
  expect_length(chunk_list, 10)
}) 