# Cross-backend integration tests

test_that("data is consistent across backend conversions", {
  # Create source data
  n_time <- 100
  n_vox <- 50
  source_mat <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  source_mask <- rep(c(TRUE, FALSE), length.out = n_vox)
  
  # Create datasets with different backends
  mat_backend <- matrix_backend(source_mat, mask = source_mask)
  mat_dset <- fmri_dataset(mat_backend, TR = 2, run_length = n_time)
  
  # Get data from matrix backend
  data1 <- get_data_matrix(mat_dset)
  mask1 <- get_mask(mat_dset)
  
  # Convert to matrix_dataset (legacy)
  mat_dset2 <- matrix_dataset(source_mat, TR = 2, run_length = n_time)
  data2 <- get_data_matrix(mat_dset2)
  
  # Matrix backend only returns masked columns
  expect_equal(ncol(data1), sum(source_mask))
  expect_equal(dim(data1), c(n_time, sum(source_mask)))
  
  # Data2 has all columns since matrix_dataset doesn't apply mask
  expect_equal(dim(data2), c(n_time, n_vox))
  
  # Check that masked data matches
  expect_equal(data1, data2[, source_mask, drop = FALSE])
  
  # Masks should be equivalent (though types may differ)
  expect_equal(length(mask1), n_vox)
  expect_equal(as.logical(mask1), source_mask)  # matrix_backend returns the actual mask passed in
})

test_that("round-trip accuracy is maintained across backends", {
  has_rhdf5 <- requireNamespace("rhdf5", quietly = TRUE)
  has_hdf5r <- requireNamespace("hdf5r", quietly = TRUE)
  if (!has_rhdf5 && !has_hdf5r) {
    skip("Requires rhdf5 or hdf5r")
  }

  # Original data
  original <- matrix(runif(200 * 30, min = -100, max = 100), 200, 30)
  
  # Save to H5 and reload
  temp_h5 <- tempfile(fileext = ".h5")
  on.exit(unlink(temp_h5))
  
  # Mock H5 backend behavior
  h5_write <- function(data, file) {
    if (has_rhdf5) {
      rhdf5::h5createFile(file)
      rhdf5::h5write(data, file, "data")
    } else if (has_hdf5r) {
      file_handle <- hdf5r::H5File$new(file, mode = "w")
      on.exit(file_handle$close_all(), add = TRUE)
      file_handle$create_dataset("data", robj = data)
    }
  }
  
  h5_read <- function(file) {
    if (has_rhdf5) {
      rhdf5::h5read(file, "data")
    } else if (has_hdf5r) {
      file_handle <- hdf5r::H5File$new(file, mode = "r")
      on.exit(file_handle$close_all(), add = TRUE)
      file_handle[["data"]]$read()
    }
  }
  
  # Write and read back
  h5_write(original, temp_h5)
  recovered <- h5_read(temp_h5)
  
  if (!is.null(recovered)) {
    # Check accuracy
    max_diff <- max(abs(original - recovered))
    expect_lt(max_diff, 1e-10)  # Very high precision
    
    # Check dimensions preserved
    expect_equal(dim(original), dim(recovered))
  }
})

test_that("mixed backend operations in pipelines work correctly", {
  # Create a pipeline that uses different backends
  n_subjects <- 3
  n_time <- 50
  n_vox <- 20
  
  # Create datasets with different backend types
  datasets <- list()
  
  # Subject 1: matrix backend
  mat1 <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  backend1 <- matrix_backend(mat1)
  datasets[[1]] <- fmri_dataset(backend1, TR = 2, run_length = n_time)
  
  # Subject 2: matrix_dataset (legacy)
  mat2 <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  datasets[[2]] <- matrix_dataset(mat2, TR = 2, run_length = n_time)
  
  # Subject 3: another matrix backend
  mat3 <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  backend3 <- matrix_backend(mat3)
  datasets[[3]] <- fmri_dataset(backend3, TR = 2, run_length = n_time)
  
  # Create study dataset mixing backends
  study <- fmri_study_dataset(datasets)
  
  # Operations should work across all subjects
  expect_equal(n_timepoints(study), n_time * n_subjects)
  
  # Extract data for each subject
  for (i in 1:n_subjects) {
    data <- get_data_matrix(datasets[[i]])
    expect_equal(dim(data), c(n_time, n_vox))
  }
})

test_that("backend-specific edge cases are handled uniformly", {
  # Test edge cases across different backends
  
  # Single voxel dataset
  single_vox <- matrix(1:10, 10, 1)
  
  # Matrix backend
  mb <- matrix_backend(single_vox)
  dset1 <- fmri_dataset(mb, TR = 1, run_length = 10)
  expect_equal(ncol(get_data_matrix(dset1)), 1)
  
  # Matrix dataset
  dset2 <- matrix_dataset(single_vox, TR = 1, run_length = 10)
  expect_equal(ncol(get_data_matrix(dset2)), 1)
  
  # Single timepoint dataset
  single_time <- matrix(1:20, 1, 20)
  
  mb2 <- matrix_backend(single_time)
  dset3 <- fmri_dataset(mb2, TR = 1, run_length = 1)
  expect_equal(nrow(get_data_matrix(dset3)), 1)
})

test_that("concurrent access patterns work correctly", {
  skip_on_cran()
  skip_if_not_installed("parallel")
  
  # Create shared dataset
  mat <- matrix(rnorm(1000 * 100), 1000, 100)
  backend <- matrix_backend(mat)
  dset <- fmri_dataset(backend, TR = 2, run_length = 1000)
  
  # Function to access random subset
  access_subset <- function(seed, backend) {
    set.seed(seed)
    rows <- sort(sample(1000, 100))
    cols <- sort(sample(100, 20))
    fmridataset::backend_get_data(backend, rows = rows, cols = cols)
  }
  
  # Sequential access
  seq_results <- lapply(1:4, access_subset, backend = backend)
  
  # Parallel access (if supported)
  if (.Platform$OS.type != "windows") {
    cl <- parallel::makeCluster(2)
    on.exit(parallel::stopCluster(cl))
    
    # Export the backend object
    parallel::clusterExport(cl, "backend", envir = environment())
    
    # Use parLapply with the backend argument
    par_results <- parallel::parLapply(cl, 1:4, access_subset, backend = backend)
    
    # Results should be identical
    for (i in 1:4) {
      expect_equal(seq_results[[i]], par_results[[i]])
    }
  }
})

test_that("conversion between storage formats preserves metadata", {
  # Create dataset with rich metadata
  mat <- matrix(rnorm(100 * 50), 100, 50)
  events <- data.frame(
    onset = seq(0, 90, by = 10),
    duration = rep(2, 10),
    trial_type = rep(c("A", "B"), 5),
    response_time = runif(10, 0.5, 2)
  )
  
  # Original dataset
  dset1 <- matrix_dataset(mat, TR = 2.5, run_length = c(40, 60),
                         event_table = events)
  
  # Convert to backend-based
  backend <- matrix_backend(mat)
  dset2 <- fmri_dataset(backend, TR = 2.5, run_length = c(40, 60),
                       event_table = events)
  
  # Metadata should be preserved
  expect_equal(get_TR(dset1), get_TR(dset2))
  expect_equal(n_runs(dset1), n_runs(dset2))
  # Compare event table contents (one is data.frame, other is tibble)
  expect_equal(as.data.frame(dset1$event_table), as.data.frame(dset2$event_table))
})

test_that("as.matrix_dataset conversion works across types", {
  # Create different dataset types
  mat <- matrix(rnorm(100 * 50), 100, 50)
  
  # Matrix dataset
  mat_dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  # Backend-based dataset
  backend <- matrix_backend(mat)
  backend_dset <- fmri_dataset(backend, TR = 2, run_length = 100)
  
  # Convert to matrix_dataset
  converted1 <- as.matrix_dataset(mat_dset)
  converted2 <- as.matrix_dataset(backend_dset)
  
  # Both should have same data
  expect_equal(get_data_matrix(converted1), get_data_matrix(converted2))
  expect_equal(get_TR(converted1), get_TR(converted2))
})

test_that("chunk iteration works consistently across backends", {
  # Test data
  mat <- matrix(1:1000, 100, 10)
  
  # Different backends
  datasets <- list(
    matrix = matrix_dataset(mat, TR = 1, run_length = 100),
    backend = fmri_dataset(matrix_backend(mat), TR = 1, run_length = 100)
  )
  
  # Chunk iteration should give same results
  for (nchunks in c(1, 5, 10)) {
    results <- lapply(datasets, function(dset) {
      chunk_iter <- data_chunks(dset, nchunks = nchunks)
      chunk_sums <- numeric()
      # Properly consume the iterator
      for (i in seq_len(chunk_iter$nchunks)) {
        chunk <- chunk_iter$nextElem()
        chunk_sums <- c(chunk_sums, sum(chunk$data))
      }
      chunk_sums
    })
    
    # All backends should give same chunk sums
    expect_equal(results[[1]], results[[2]])
  }
})

test_that("backend cleanup happens correctly in pipelines", {
  # Track cleanup calls globally for this test
  cleanup_count <- 0
  
  # Define methods outside to avoid re-registration
  backend_open.cleanup_backend <- function(b) { 
    b$is_open <- TRUE
    b 
  }
  backend_close.cleanup_backend <- function(b) { 
    if (b$is_open) {
      b$cleanup()
      b$is_open <- FALSE
    }
    invisible(NULL) 
  }
  backend_get_dims.cleanup_backend <- function(b) {
    list(spatial = c(ncol(b$data), 1, 1), time = nrow(b$data))
  }
  backend_get_mask.cleanup_backend <- function(b) {
    rep(TRUE, ncol(b$data))
  }
  backend_get_data.cleanup_backend <- function(b, rows = NULL, cols = NULL) {
    b$data[rows %||% TRUE, cols %||% TRUE, drop = FALSE]
  }
  backend_get_metadata.cleanup_backend <- function(b) list()
  
  # Register methods once
  for (method in c("open", "close", "get_dims", "get_mask", "get_data", "get_metadata")) {
    registerS3method(
      paste0("backend_", method), 
      "cleanup_backend",
      get(paste0("backend_", method, ".cleanup_backend"))
    )
  }
  
  # Create mock backend with cleanup tracking
  create_mock_backend <- function(data) {
    structure(
      list(
        data = data,
        is_open = FALSE,
        cleanup = function() {
          cleanup_count <<- cleanup_count + 1
        }
      ),
      class = c("cleanup_backend", "storage_backend")
    )
  }
  
  # Create and use backend
  mat <- matrix(1:20, 10, 2)
  backend <- create_mock_backend(mat)
  
  # Cleanup count should be 0 initially
  expect_equal(cleanup_count, 0)
  
  dset <- fmri_dataset(backend, TR = 1, run_length = 10)
  
  # After dataset creation, cleanup has not been called yet
  expect_equal(cleanup_count, 0)
  
  # Use the dataset - cleanup count should remain at 0
  data <- get_data_matrix(dset)
  expect_equal(cleanup_count, 0)
  
  # Explicitly close
  backend_close(dset$backend)
  
  # Cleanup should have been called after explicit close
  expect_equal(cleanup_count, 1)
})

test_that("error handling is consistent across backends", {
  # Test common error conditions
  
  # Invalid dimensions
  mat <- matrix(1:20, 10, 2)
  
  # Matrix dataset
  expect_error(
    matrix_dataset(mat, TR = 1, run_length = 20),  # Wrong run_length
    "sum\\(run_length\\) not equal to nrow\\(datamat\\)"
  )
  
  # Backend-based
  backend <- matrix_backend(mat)
  expect_error(
    fmri_dataset(backend, TR = 1, run_length = 20),
    "Sum of run_length.*must equal"
  )
  
  # Out of bounds access
  dset1 <- matrix_dataset(mat, TR = 1, run_length = 10)
  dset2 <- fmri_dataset(backend, TR = 1, run_length = 10)
  
  # Both should error similarly
  expect_error(get_data_matrix(dset1, rows = 11))
  expect_error(get_data_matrix(dset2, rows = 11))
})
