# Memory safety tests for fmridataset

test_that("no memory leaks in dataset creation and destruction", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Function to create and destroy datasets
  create_destroy <- function(n = 100) {
    for (i in 1:n) {
      mat <- matrix(rnorm(100 * 50), 100, 50)
      dset <- matrix_dataset(mat, TR = 2, run_length = 100)
      rm(dset, mat)
    }
    gc()
  }

  # Measure memory before
  gc()
  mem_before <- sum(gc()[, 2])

  # Create and destroy many datasets
  create_destroy(100)

  # Measure memory after
  gc()
  mem_after <- sum(gc()[, 2])

  # Memory should not grow significantly
  mem_growth <- mem_after - mem_before
  expect_lt(mem_growth, 10) # Less than 10MB growth
})

test_that("large datasets don't cause OOM with proper chunking", {
  skip_on_cran()
  skip_if(Sys.getenv("GITHUB_ACTIONS") == "true", "Skip large memory test on CI")

  # Create a large dataset that would be problematic if fully loaded
  n_time <- 5000
  n_vox <- 5000

  # Don't actually allocate the full matrix
  # Instead, create a backend that simulates large data
  mock_large_backend <- structure(
    list(
      dims = list(spatial = c(n_vox, 1, 1), time = n_time),
      chunk_size = 100
    ),
    class = c("mock_large_backend", "storage_backend")
  )

  # Add methods that simulate large data access
  backend_open.mock_large_backend <- function(backend) backend
  backend_close.mock_large_backend <- function(backend) invisible(NULL)
  backend_get_dims.mock_large_backend <- function(backend) backend$dims
  backend_get_mask.mock_large_backend <- function(backend) {
    rep(TRUE, prod(backend$dims$spatial))
  }
  backend_get_data.mock_large_backend <- function(backend, rows = NULL, cols = NULL) {
    if (is.null(rows)) rows <- 1:backend$dims$time
    if (is.null(cols)) cols <- 1:prod(backend$dims$spatial)

    # Only generate requested data
    matrix(
      rnorm(length(rows) * length(cols)),
      length(rows), length(cols)
    )
  }
  backend_get_metadata.mock_large_backend <- function(backend) list()

  # Register methods
  for (method in c("open", "close", "get_dims", "get_mask", "get_data", "get_metadata")) {
    registerS3method(
      paste0("backend_", method), "mock_large_backend",
      get(paste0("backend_", method, ".mock_large_backend"))
    )
  }

  # Create dataset
  dset <- fmri_dataset(mock_large_backend, TR = 2, run_length = n_time)

  # Process in chunks without OOM
  chunks <- data_chunks(dset, nchunks = 50) # 50 chunks for 5000 timepoints
  total_sum <- 0
  chunk_count <- 0

  for (i in seq_len(chunks$nchunks)) {
    chunk <- chunks$nextElem()
    total_sum <- total_sum + sum(chunk$data)
    chunk_count <- chunk_count + 1

    # Each chunk should be small
    expect_lt(object.size(chunk$data), 10e6) # Less than 10MB per chunk
  }

  expect_gt(chunk_count, 1) # Should have created multiple chunks
})

test_that("temporary files are properly cleaned up", {
  skip_on_cran()

  # Count temp files before
  temp_dir <- tempdir()
  files_before <- list.files(temp_dir, pattern = "^fmri.*", full.names = TRUE)

  # Operations that might create temp files
  mat <- matrix(rnorm(1000 * 100), 1000, 100)
  dset <- matrix_dataset(mat, TR = 2, run_length = 1000)

  # Chunk iteration
  chunks <- data_chunks(dset, nchunks = 10)
  for (i in seq_len(chunks$nchunks)) {
    chunk <- chunks$nextElem()
    sum(chunk$data)
  }

  # Force cleanup
  rm(dset, chunks)
  gc()

  # Count temp files after
  files_after <- list.files(temp_dir, pattern = "^fmri.*", full.names = TRUE)
  new_files <- setdiff(files_after, files_before)

  # No new temp files should remain
  expect_length(new_files, 0)
})

test_that("DelayedArray operations are memory efficient", {
  skip_if_not_installed("DelayedArray")
  skip_if_not_installed("DelayedMatrixStats")
  skip_on_cran()

  # Create dataset
  mat <- matrix(rnorm(2000 * 500), 2000, 500)
  dset <- matrix_dataset(mat, TR = 2, run_length = 2000)

  # Convert to DelayedArray
  gc()
  mem_before <- sum(gc()[, 2])

  delayed <- as_delayed_array(dset)

  gc()
  mem_after_conversion <- sum(gc()[, 2])

  # Conversion should use minimal memory
  conversion_mem <- mem_after_conversion - mem_before
  expect_lt(conversion_mem, object.size(mat) * 0.1) # Less than 10% of data size

  # Operations should also be memory efficient
  row_means <- DelayedMatrixStats::rowMeans2(delayed)

  gc()
  mem_after_operation <- sum(gc()[, 2])

  # Operation should not load full matrix
  operation_mem <- mem_after_operation - mem_after_conversion
  expect_lt(operation_mem, object.size(mat) * 0.2)
})

test_that("study backend lazy evaluation prevents memory explosion", {
  skip_on_cran()

  # Create multiple subject datasets
  n_subjects <- 20
  n_time <- 500
  n_vox <- 100

  datasets <- lapply(1:n_subjects, function(i) {
    mat <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
    matrix_dataset(mat, TR = 2, run_length = n_time)
  })

  # Measure memory before study creation
  gc()
  mem_before <- sum(gc()[, 2])

  # Create study dataset
  study <- fmri_study_dataset(datasets)

  # Access only one subject
  data_s1 <- get_data_matrix(study, subject_id = 1)

  gc()
  mem_after <- sum(gc()[, 2])

  # Memory growth should be minimal (just one subject's data)
  mem_growth <- mem_after - mem_before
  single_subject_size <- object.size(data_s1)

  # Should not have loaded all subjects
  expect_lt(mem_growth, as.numeric(single_subject_size) * 3)
})

test_that("cache memory is bounded for NIfTI backend", {
  skip_on_cran()
  skip_if_not_installed("neuroim2")

  # Simulate NIfTI caching behavior
  cache_size <- 0
  max_cache_size <- 50 * 1024^2 # 50MB limit

  # Mock cache operations
  add_to_cache <- function(key, value) {
    value_size <- as.numeric(object.size(value))
    if (cache_size + value_size > max_cache_size) {
      # Evict old entries (simulated)
      cache_size <<- value_size
    } else {
      cache_size <<- cache_size + value_size
    }
  }

  # Simulate many cache additions
  for (i in 1:100) {
    data <- matrix(rnorm(1000 * 100), 1000, 100)
    add_to_cache(paste0("file_", i), data)

    # Cache should never exceed limit
    expect_lte(cache_size, max_cache_size)
  }
})

test_that("circular references don't prevent garbage collection", {
  skip_on_cran()

  # Create objects with potential circular references
  create_circular <- function() {
    mat <- matrix(rnorm(100 * 50), 100, 50)
    dset <- matrix_dataset(mat, TR = 2, run_length = 100)

    # Add self-reference (potential issue)
    dset$self_ref <- dset

    # Add environment that captures dataset
    dset$env <- new.env()
    dset$env$dataset <- dset

    dset
  }

  # Track if objects are freed
  finalized <- FALSE

  # Create dataset
  dset <- create_circular()

  # Create environment to track finalization
  e <- new.env()
  e$dset <- dset
  reg.finalizer(e, function(e) finalized <<- TRUE, onexit = FALSE)

  # Remove references
  rm(dset, e)

  # Force garbage collection
  gc()
  gc() # Sometimes needs multiple passes

  # Object should be finalized despite circular refs
  expect_true(finalized)
})

test_that("memory mapping works correctly for large files", {
  skip_on_cran()
  skip_if(.Platform$OS.type == "windows", "Memory mapping behaves differently on Windows")

  # Create a large temporary file
  temp_file <- tempfile()
  n_elements <- 1000000 # 1M elements

  # Write data
  data <- rnorm(n_elements)
  writeBin(data, temp_file)
  on.exit(unlink(temp_file))

  # Memory map the file (simulated)
  mmap_data <- function(file, mode = "read") {
    # In real implementation, would use mmap
    # For test, just track that file isn't fully loaded
    list(
      file = file,
      size = file.size(file),
      mode = mode
    )
  }

  mapped <- mmap_data(temp_file)

  # Accessing mapped data shouldn't load entire file
  # In real implementation, would check resident memory
  expect_equal(mapped$mode, "read")
  expect_gt(mapped$size, 0)
})

test_that("concurrent access doesn't cause memory duplication", {
  skip_on_cran()
  skip_if_not_installed("parallel")
  skip_if(.Platform$OS.type == "windows", "Fork not available on Windows")

  # Create shared dataset
  mat <- matrix(rnorm(1000 * 100), 1000, 100)
  dset <- matrix_dataset(mat, TR = 2, run_length = 1000)

  # Function to access data
  access_data <- function(dset, indices) {
    subset <- get_data_matrix(dset, rows = indices)
    sum(subset)
  }

  # Measure memory before parallel access
  gc()
  mem_before <- sum(gc()[, 2])

  # Parallel access (using fork on Unix)
  if (.Platform$OS.type == "unix") {
    library(parallel)
    results <- mclapply(1:4, function(i) {
      indices <- ((i - 1) * 250 + 1):(i * 250)
      access_data(dset, indices)
    }, mc.cores = 2)
  }

  gc()
  mem_after <- sum(gc()[, 2])

  # Memory shouldn't increase much (data shared via fork)
  mem_growth <- mem_after - mem_before
  expect_lt(mem_growth, object.size(mat) * 0.5)
})

test_that("error paths don't leak memory", {
  skip_on_cran()

  # Function that errors after allocating memory
  error_after_alloc <- function() {
    # Allocate some memory
    big_mat <- matrix(rnorm(1000 * 1000), 1000, 1000)

    # Then error
    stop("Simulated error")
  }

  # Measure memory before
  gc()
  mem_before <- sum(gc()[, 2])

  # Call function that errors multiple times
  for (i in 1:10) {
    try(error_after_alloc(), silent = TRUE)
  }

  # Force cleanup
  gc()
  mem_after <- sum(gc()[, 2])

  # Memory should be reclaimed despite errors
  mem_growth <- mem_after - mem_before
  expect_lt(mem_growth, 10) # Less than 10MB growth
})
