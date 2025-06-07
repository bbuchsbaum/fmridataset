test_that("backend chunking doesn't load full dataset into memory", {
  skip("Memory benchmarking can be unreliable in test environments")
  skip_if_not_installed("bench")
  skip_if_not_installed("neuroim2")
  
  # Create a moderately large test dataset
  n_timepoints <- 300
  n_voxels <- 10000
  spatial_dims <- c(100, 100, 1)
  
  # Create test data
  test_data <- matrix(rnorm(n_timepoints * n_voxels), 
                     nrow = n_timepoints, 
                     ncol = n_voxels)
  
  # Create mask
  mask <- rep(TRUE, n_voxels)
  
  # Create backend
  backend <- matrix_backend(
    data_matrix = test_data,
    mask = mask,
    spatial_dims = spatial_dims
  )
  
  # Create dataset
  dset <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = c(150, 150)
  )
  
  # Benchmark chunked iteration
  bench_result <- bench::mark(
    chunked = {
      chunks <- data_chunks(dset, nchunks = 10)
      total <- 0
      for (i in 1:10) {
        chunk <- chunks$nextElem()
        total <- total + sum(chunk$data)
      }
      total
    },
    iterations = 1,
    check = FALSE
  )
  
  # Memory allocated should be much less than full dataset size
  full_size <- object.size(test_data)
  allocated_mem <- bench_result$mem_alloc[[1]]
  
  # The allocated memory should be closer to chunk size than full dataset
  # Allowing for some overhead, but should be less than 50% of full size
  expect_true(
    allocated_mem < 0.5 * full_size,
    info = sprintf("Allocated memory (%s) should be much less than full dataset (%s)",
                   format(allocated_mem, units = "auto"),
                   format(full_size, units = "auto"))
  )
})

test_that("backend chunking produces correct results with matrix backend", {
  skip_if_not_installed("neuroim2")
  
  # Create small test dataset
  test_data <- matrix(1:100, nrow = 10, ncol = 10)
  mask <- rep(TRUE, 10)
  
  backend <- matrix_backend(
    data_matrix = test_data,
    mask = mask,
    spatial_dims = c(10, 1, 1)
  )
  
  dset <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = 10
  )
  
  # Test single chunk
  chunks <- data_chunks(dset, nchunks = 1)
  chunk <- chunks$nextElem()
  
  expect_equal(chunk$data, test_data)
  expect_equal(chunk$voxel_ind, 1:10)
  expect_equal(chunk$row_ind, 1:10)
  
  # Test multiple chunks
  chunks <- data_chunks(dset, nchunks = 2)
  chunk1 <- chunks$nextElem()
  chunk2 <- chunks$nextElem()
  
  # Verify chunks partition the data correctly
  expect_equal(ncol(chunk1$data) + ncol(chunk2$data), ncol(test_data))
  
  # Test runwise chunks
  dset2 <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = c(5, 5)
  )
  
  chunks <- data_chunks(dset2, runwise = TRUE)
  chunk1 <- chunks$nextElem()
  chunk2 <- chunks$nextElem()
  
  expect_equal(nrow(chunk1$data), 5)
  expect_equal(nrow(chunk2$data), 5)
  expect_equal(chunk1$row_ind, 1:5)
  expect_equal(chunk2$row_ind, 6:10)
})

test_that("backend chunking works with file-based datasets", {
  # Use matrix backend to simulate file-based behavior
  # Create test data
  n_time <- 100
  n_voxels <- 1000
  test_data <- matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels)
  
  backend <- matrix_backend(
    data_matrix = test_data,
    spatial_dims = c(10, 10, 10)
  )
  
  dset <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = c(50, 50)
  )
  
  # Test different chunking strategies
  # 1. Single chunk
  chunks <- data_chunks(dset, nchunks = 1)
  single_chunk <- chunks$nextElem()
  # Single chunk should have all timepoints
  expect_equal(nrow(single_chunk$data), 100)
  expect_equal(ncol(single_chunk$data), 1000)
  
  # 2. Multiple chunks
  chunks <- data_chunks(dset, nchunks = 5)
  all_chunks <- list()
  for (i in 1:5) {
    all_chunks[[i]] <- chunks$nextElem()
  }
  
  # Verify chunks partition the voxels
  total_voxels <- sum(sapply(all_chunks, function(x) ncol(x$data)))
  expect_equal(total_voxels, 1000)
  
  # 3. Runwise chunks
  chunks <- data_chunks(dset, runwise = TRUE)
  run1 <- chunks$nextElem()
  run2 <- chunks$nextElem()
  
  expect_equal(nrow(run1$data), 50)
  expect_equal(nrow(run2$data), 50)
  expect_equal(ncol(run1$data), 1000)
  expect_equal(ncol(run2$data), 1000)
})

test_that("chunking correctly handles subsetting", {
  # Create data with known pattern
  n_time <- 30
  n_voxels <- 20
  test_data <- matrix(0, nrow = n_time, ncol = n_voxels)
  
  # Fill with pattern: column i contains value i
  for (i in 1:n_voxels) {
    test_data[, i] <- i
  }
  
  backend <- matrix_backend(test_data)
  dset <- fmri_dataset(backend, TR = 2, run_length = n_time)
  
  # Test that chunks contain correct voxel subsets
  chunks <- data_chunks(dset, nchunks = 4)
  
  chunk1 <- chunks$nextElem()
  # First chunk should have first set of voxels
  expect_true(all(chunk1$data[1, ] %in% 1:5))
  
  chunk2 <- chunks$nextElem()
  # Second chunk should have next set of voxels
  expect_true(all(chunk2$data[1, ] %in% 6:10))
})

test_that("chunk iterator stops correctly", {
  test_data <- matrix(1:100, nrow = 10, ncol = 10)
  backend <- matrix_backend(test_data)
  dset <- fmri_dataset(backend, TR = 2, run_length = 10)
  
  chunks <- data_chunks(dset, nchunks = 3)
  
  # Get all chunks
  for (i in 1:3) {
    chunk <- chunks$nextElem()
    expect_s3_class(chunk, "data_chunk")
  }
  
  # Next call should error
  expect_error(chunks$nextElem(), "StopIteration")
})

test_that("data_chunks preserves chunk metadata", {
  test_data <- matrix(rnorm(300), nrow = 30, ncol = 10)
  backend <- matrix_backend(test_data)
  dset <- fmri_dataset(backend, TR = 2, run_length = c(15, 15))
  
  # Test runwise chunks
  chunks <- data_chunks(dset, runwise = TRUE)
  
  chunk1 <- chunks$nextElem()
  expect_equal(chunk1$chunk_num, 1)
  expect_true(is.numeric(chunk1$row_ind))
  expect_true(length(chunk1$row_ind) > 0)
  expect_equal(ncol(chunk1$data), 10)
  
  chunk2 <- chunks$nextElem()
  expect_equal(chunk2$chunk_num, 2)
  expect_true(is.numeric(chunk2$row_ind))
  expect_true(length(chunk2$row_ind) > 0)
})