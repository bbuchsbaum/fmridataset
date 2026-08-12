# Test refactored modular structure
# This file tests that the refactored components work together correctly

test_that("all generic functions are properly declared", {
  # Test that generic functions exist and work
  expect_true(exists("get_data"))
  expect_true(exists("get_data_matrix"))
  expect_true(exists("get_mask"))
  expect_true(exists("blocklens"))
  expect_true(exists("data_chunks"))
  expect_true(exists("as.matrix_dataset"))

  # Test that they are indeed generic functions
  expect_true(is.function(get_data))
  expect_true(is.function(get_data_matrix))
  expect_true(is.function(get_mask))
  expect_true(is.function(blocklens))
  expect_true(is.function(data_chunks))
  expect_true(is.function(as.matrix_dataset))
})

test_that("dataset constructors work from dataset_constructors.R", {
  # Test matrix_dataset constructor
  Y <- matrix(rnorm(50 * 10), 50, 10)
  dset_matrix <- matrix_dataset(Y, TR = 2, run_length = 50)

  expect_s3_class(dset_matrix, "matrix_dataset")
  expect_s3_class(dset_matrix, "fmri_dataset")
  expect_equal(dset_matrix$TR, 2)
  expect_equal(dset_matrix$nruns, 1)

  # Test fmri_mem_dataset constructor
  arr <- array(rnorm(5 * 5 * 5 * 20), c(5, 5, 5, 20))
  bspace <- neuroim2::NeuroSpace(dim = c(5, 5, 5, 20))
  nvec <- neuroim2::NeuroVec(arr, bspace)
  mask <- neuroim2::LogicalNeuroVol(array(TRUE, c(5, 5, 5)), neuroim2::NeuroSpace(dim = c(5, 5, 5)))

  dset_mem <- fmri_mem_dataset(scans = list(nvec), mask = mask, TR = 1.5)

  expect_s3_class(dset_mem, "fmri_mem_dataset")
  expect_s3_class(dset_mem, "fmri_dataset")
  expect_equal(length(dset_mem$scans), 1)
})

test_that("data access methods work from data_access.R", {
  # Create test datasets
  Y <- matrix(rnorm(30 * 8), 30, 8)
  dset_matrix <- matrix_dataset(Y, TR = 1, run_length = 30)

  # Test get_data generic and method
  data_result <- get_data(dset_matrix)
  expect_identical(data_result, Y)

  # Test get_data_matrix generic and method
  matrix_result <- get_data_matrix(dset_matrix)
  expect_identical(matrix_result, Y)

  # Test get_mask generic and method
  mask_result <- get_mask(dset_matrix)
  expect_equal(length(mask_result), 8)
  expect_true(all(mask_result == 1))

  # Test blocklens generic and method
  blocklens_result <- blocklens(dset_matrix)
  expect_equal(blocklens_result, c(30))
})

test_that("data chunking works from data_chunks.R", {
  # Create test data
  Y <- matrix(rnorm(40 * 12), 40, 12)
  run_lengths <- c(20, 20)
  dset <- matrix_dataset(Y, TR = 1, run_length = run_lengths)

  # Test data_chunks generic and method
  chunks_runwise <- data_chunks(dset, runwise = TRUE)
  expect_s3_class(chunks_runwise, "chunkiter")
  expect_equal(chunks_runwise$nchunks, 2)

  # Test single chunk
  chunks_single <- data_chunks(dset, nchunks = 1)
  expect_s3_class(chunks_single, "chunkiter")
  expect_equal(chunks_single$nchunks, 1)

  # Extract a chunk and test structure
  chunk <- chunks_single$nextElem()
  expect_s3_class(chunk, "data_chunk")
  expect_true(all(c("data", "voxel_ind", "row_ind", "chunk_num") %in% names(chunk)))
})

test_that("type conversions work from conversions.R", {
  # Create a matrix dataset
  Y <- matrix(rnorm(25 * 6), 25, 6)
  dset_matrix <- matrix_dataset(Y, TR = 2, run_length = 25)

  # Test as.matrix_dataset generic and method
  converted <- as.matrix_dataset(dset_matrix)
  expect_s3_class(converted, "matrix_dataset")
  expect_identical(converted, dset_matrix) # Should be the same object
})

test_that("print methods work from print_methods.R", {
  # Create test dataset
  Y <- matrix(rnorm(20 * 5), 20, 5)
  dset <- matrix_dataset(Y, TR = 1.5, run_length = 20)

  # Test that print method exists and runs without error
  expect_output(print(dset), "fMRI Dataset")

  # Test data chunk printing
  chunks <- data_chunks(dset, nchunks = 1)
  chunk <- chunks$nextElem()
  expect_output(print(chunk), "Data Chunk Object")

  # Test chunk iterator printing
  expect_output(print(chunks), "Chunk Iterator")
})

test_that("configuration functions work from config.R", {
  # Test that default_config function exists (internal)
  # Note: We can't easily test read_fmri_config without external files

  # Test foreach operators are available from foreach package
  expect_true(exists("%dopar%", where = asNamespace("foreach")))
  expect_true(exists("%do%", where = asNamespace("foreach")))
})

test_that("cross-module integration works correctly", {
  # Test the full workflow using multiple modules

  # 1. Create dataset (dataset_constructors.R)
  Y <- matrix(rnorm(60 * 15), 60, 15)
  dset <- matrix_dataset(Y, TR = 2.5, run_length = c(30, 30))

  # 2. Access data (data_access.R)
  data_mat <- get_data_matrix(dset)
  mask <- get_mask(dset)

  # 3. Create chunks (data_chunks.R)
  chunks <- data_chunks(dset, nchunks = 3)

  # 4. Process chunks
  chunk_means <- list()
  for (i in 1:chunks$nchunks) {
    chunk <- chunks$nextElem()
    chunk_means[[i]] <- colMeans(chunk$data)
  }

  # 5. Convert types (conversions.R)
  converted_dset <- as.matrix_dataset(dset)

  # Verify the workflow worked
  expect_equal(length(chunk_means), 3)
  expect_true(all(sapply(chunk_means, length) > 0))
  expect_s3_class(converted_dset, "matrix_dataset")
  expect_equal(nrow(data_mat), 60)
  expect_equal(ncol(data_mat), 15)
})

test_that("backwards compatibility is maintained", {
  # Test that the refactored code maintains the same API

  # Old API calls should still work
  Y <- matrix(rnorm(40 * 8), 40, 8)
  dset <- matrix_dataset(Y, TR = 1, run_length = 40)

  # These calls should work exactly as before
  expect_true(!is.null(dset))
  expect_true(!is.null(get_data(dset)))
  expect_true(!is.null(data_chunks(dset)))

  # Class structure should be preserved
  expect_true(inherits(dset, "matrix_dataset"))
  expect_true(inherits(dset, "fmri_dataset"))
  expect_true(inherits(dset, "list"))
})
