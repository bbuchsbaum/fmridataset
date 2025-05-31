# Tests for matrix_dataset function
# This tests the new matrix_dataset function that provides fmrireg compatibility

test_that("matrix_dataset creates valid fmri_dataset", {
  # Basic single run dataset
  set.seed(123)
  X <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
  dset <- matrix_dataset(X, TR = 2, run_length = 100)
  
  expect_true(is.fmri_dataset(dset))
  expect_true(is.matrix_dataset(dset))
  expect_equal(get_dataset_type(dset), "matrix")
  expect_equal(n_timepoints(dset$sampling_frame), 100)
  expect_equal(n_runs(dset$sampling_frame), 1)
  expect_equal(get_TR(dset$sampling_frame), 2)
  expect_equal(get_num_voxels(dset), 50)
  
  # Check metadata
  expect_equal(dset$metadata$creation_method, "matrix_dataset")
  expect_equal(dset$metadata$original_dims, c(100, 50))
})

test_that("matrix_dataset works with multiple runs", {
  set.seed(456)
  Y <- matrix(rnorm(200 * 30), nrow = 200, ncol = 30)
  dset <- matrix_dataset(Y, TR = 1.5, run_length = c(100, 100))
  
  expect_true(is.fmri_dataset(dset))
  expect_equal(n_timepoints(dset$sampling_frame), 200)
  expect_equal(n_runs(dset$sampling_frame), 2)
  expect_equal(get_run_lengths(dset$sampling_frame), c(100, 100))
  expect_equal(get_TR(dset$sampling_frame), c(1.5, 1.5))
})

test_that("matrix_dataset works with variable run lengths", {
  set.seed(789)
  Z <- matrix(rnorm(250 * 20), nrow = 250, ncol = 20)
  dset <- matrix_dataset(Z, TR = 2.5, run_length = c(80, 120, 50))
  
  expect_equal(n_runs(dset$sampling_frame), 3)
  expect_equal(get_run_lengths(dset$sampling_frame), c(80, 120, 50))
  expect_equal(n_timepoints(dset$sampling_frame), 250)
})

test_that("matrix_dataset works with event table", {
  set.seed(101112)
  X <- matrix(rnorm(100 * 25), nrow = 100, ncol = 25)
  
  events <- data.frame(
    onset = c(10, 30, 60),
    duration = c(2, 2, 3),
    trial_type = c("A", "B", "A"),
    run = 1
  )
  
  dset <- matrix_dataset(X, TR = 2, run_length = 100, event_table = events)
  
  stored_events <- get_event_table(dset)
  expect_equal(nrow(stored_events), 3)
  expect_true("trial_type" %in% names(stored_events))
  expect_equal(stored_events$onset, c(10, 30, 60))
})

test_that("matrix_dataset works with censoring", {
  set.seed(131415)
  X <- matrix(rnorm(100 * 10), nrow = 100, ncol = 10)
  
  # Test logical censoring
  censor_logical <- c(rep(FALSE, 95), rep(TRUE, 5))
  dset1 <- matrix_dataset(X, TR = 2, run_length = 100, censor = censor_logical)
  
  expect_true(is.fmri_dataset(dset1))
  censor_vec <- get_censor_vector(dset1)
  expect_equal(sum(censor_vec), 5)
  
  # Test numeric censoring
  censor_numeric <- c(96, 97, 98, 99, 100)
  dset2 <- matrix_dataset(X, TR = 2, run_length = 100, censor = censor_numeric)
  
  censor_vec2 <- get_censor_vector(dset2)
  expect_equal(sum(censor_vec2), 5)
  expect_equal(which(censor_vec2), censor_numeric)
})

test_that("matrix_dataset works with masking", {
  set.seed(161718)
  X <- matrix(rnorm(100 * 20), nrow = 100, ncol = 20)
  
  # Include only first 10 voxels
  mask_vec <- c(rep(TRUE, 10), rep(FALSE, 10))
  dset <- matrix_dataset(X, TR = 2, run_length = 100, mask = mask_vec)
  
  expect_equal(get_num_voxels(dset), 10)
  
  # Get data should only return masked voxels
  data_matrix <- get_data_matrix(dset)
  expect_equal(ncol(data_matrix), 10)
})

test_that("matrix_dataset input validation works", {
  X <- matrix(rnorm(100 * 10), nrow = 100, ncol = 10)
  
  # Test run_length sum mismatch
  expect_error(
    matrix_dataset(X, TR = 2, run_length = 50),
    "Sum of run_length \\(50\\) must equal number of rows in datamat \\(100\\)"
  )
  
  # Test negative TR
  expect_error(
    matrix_dataset(X, TR = -1, run_length = 100),
    "TR must be positive"
  )
  
  # Test negative run_length
  expect_error(
    matrix_dataset(X, TR = 2, run_length = c(50, -50)),
    "All run_length values must be positive"
  )
  
  # Test censor length mismatch
  expect_error(
    matrix_dataset(X, TR = 2, run_length = 100, censor = rep(FALSE, 50)),
    "censor vector length \\(50\\) must match number of timepoints \\(100\\)"
  )
  
  # Test mask length mismatch
  expect_error(
    matrix_dataset(X, TR = 2, run_length = 100, mask = rep(TRUE, 5)),
    "mask vector length \\(5\\) must match number of voxels \\(10\\)"
  )
})

test_that("matrix_dataset works with vector input", {
  # Test that vectors are converted to matrices
  vec <- rnorm(100)
  dset <- matrix_dataset(vec, TR = 2, run_length = 100)
  
  expect_true(is.fmri_dataset(dset))
  expect_equal(get_num_voxels(dset), 1)
  expect_equal(n_timepoints(dset$sampling_frame), 100)
})

test_that("matrix_dataset is compatible with data_chunks", {
  # Test that matrix_dataset works with our chunking system
  set.seed(192021)
  X <- matrix(rnorm(100 * 20), nrow = 100, ncol = 20)
  dset <- matrix_dataset(X, TR = 2, run_length = c(50, 50))
  
  # Test voxel chunking
  chunks <- data_chunks(dset, nchunks = 4, by = "voxel")
  expect_equal(attr(chunks, "total_chunks"), 4)
  
  chunk1 <- chunks$nextElem()
  expect_s3_class(chunk1, "fmri_data_chunk")
  expect_equal(nrow(chunk1$data), 100)
  
  # Test run chunking
  run_chunks <- data_chunks(dset, by = "run")
  expect_equal(attr(run_chunks, "total_chunks"), 2)
  
  run_chunk1 <- run_chunks$nextElem()
  expect_equal(nrow(run_chunk1$data), 50)
  expect_equal(run_chunk1$timepoint_indices, 1:50)
})

test_that("matrix_dataset data integrity is preserved", {
  # Test that data can be perfectly reconstructed
  set.seed(222324)
  original_data <- matrix(rnorm(60 * 15), nrow = 60, ncol = 15)
  dset <- matrix_dataset(original_data, TR = 1.5, run_length = 60)
  
  retrieved_data <- get_data_matrix(dset)
  expect_equal(retrieved_data, original_data)
})

test_that("matrix_dataset print method works", {
  set.seed(252627)
  X <- matrix(rnorm(50 * 10), nrow = 50, ncol = 10)
  dset <- matrix_dataset(X, TR = 2, run_length = 50)
  
  # Test that print doesn't error and includes matrix information
  output <- capture.output(print(dset))
  expect_true(any(grepl("Matrix Dataset", output)))
  expect_true(any(grepl("Original matrix", output)))
})

test_that("matrix_dataset backward compatibility with legacy tests", {
  # Test the exact same pattern used in legacy chunk tests
  n_time <- 100
  n_vox <- 10
  n_runs <- 2
  
  Y <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  run_length <- rep(n_time/n_runs, n_runs)
  
  # Create with our new function
  dset <- matrix_dataset(Y, TR = 1, run_length = run_length)
  
  # Should work with legacy test patterns
  chunks <- data_chunks(dset, by = "run")
  expect_equal(attr(chunks, "total_chunks"), 2)
  
  # Collect chunks the same way legacy tests do
  chunk_list <- list()
  chunk_count <- 0
  for (chunk in chunks) {
    chunk_count <- chunk_count + 1
    chunk_list[[chunk_count]] <- chunk
  }
  
  expect_equal(length(chunk_list), n_runs)
  expect_equal(nrow(chunk_list[[1]]$data), n_time/n_runs)
  expect_equal(ncol(chunk_list[[1]]$data), n_vox)
}) 