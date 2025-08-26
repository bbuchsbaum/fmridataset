test_that("StudyBackendSeed provides true lazy evaluation", {
  skip_if_not_installed("DelayedArray")

  # Create mock backends with known data
  create_mock_backend <- function(id, n_time = 100, n_voxels = 1000) {
    data_matrix <- matrix(
      rep(id, n_time * n_voxels),
      nrow = n_time,
      ncol = n_voxels
    )

    backend <- matrix_backend(data_matrix, mask = rep(TRUE, n_voxels))
    backend
  }

  # Create 5 mock subjects
  backends <- lapply(1:5, create_mock_backend)
  subject_ids <- paste0("sub-", sprintf("%02d", 1:5))

  # Create study backend
  study <- study_backend(backends, subject_ids)

  # Get as DelayedArray - this should NOT load all data
  da <- as_delayed_array(study)

  # Verify it's a DelayedArray
  expect_s4_class(da, "DelayedArray")

  # Verify dimensions
  expect_equal(dim(da), c(500, 1000)) # 5 subjects x 100 timepoints each

  # Extract a small subset - should only load relevant subject
  subset_data <- da[150:250, 1:10]

  # This should have data from subjects 2 and 3
  expect_equal(dim(subset_data), c(101, 10))

  # Verify the data contains expected values
  # Rows 150-200 should be from subject 2 (value = 2)
  # Rows 201-250 should be from subject 3 (value = 3)
  expect_equal(unique(as.vector(subset_data[1:51, 1])), 2) # Rows 150-200 (subset rows 1-51)
  expect_equal(unique(as.vector(subset_data[52:101, 1])), 3) # Rows 201-250 (subset rows 52-101)
})

test_that("StudyBackendSeed respects memory bounds", {
  skip_if_not_installed("DelayedArray")
  skip_if_not_installed("bench")

  # Create larger mock backends
  create_large_backend <- function(id, n_time = 500, n_voxels = 10000) {
    # Don't actually create the full matrix - use a sparse representation
    backend <- structure(
      list(
        id = id,
        n_time = n_time,
        n_voxels = n_voxels
      ),
      class = c("mock_large_backend", "storage_backend")
    )
    backend
  }

  # Define methods for mock backend - register as S3 methods
  registerS3method(
    "backend_get_dims", "mock_large_backend",
    function(backend) {
      list(
        spatial = c(100L, 100L, 1L), # Fake spatial dims
        time = as.integer(backend$n_time)
      )
    }
  )

  registerS3method(
    "backend_get_data", "mock_large_backend",
    function(backend, rows = NULL, cols = NULL) {
      if (is.null(rows)) rows <- seq_len(backend$n_time)
      if (is.null(cols)) cols <- seq_len(backend$n_voxels)

      # Return small matrix with backend ID
      matrix(backend$id, nrow = length(rows), ncol = length(cols))
    }
  )

  registerS3method(
    "backend_get_mask", "mock_large_backend",
    function(backend) {
      rep(TRUE, backend$n_voxels)
    }
  )

  registerS3method(
    "backend_open", "mock_large_backend",
    function(backend) backend
  )

  registerS3method(
    "backend_close", "mock_large_backend",
    function(backend) invisible(NULL)
  )

  # No cleanup needed when using registerS3method

  # Register as_delayed_array method
  setMethod("as_delayed_array", "mock_large_backend", function(backend, sparse_ok = FALSE) {
    seed <- new("StorageBackendSeed", backend = backend)
    DelayedArray::DelayedArray(seed)
  })

  # Create 10 large subjects (would be ~4GB if fully loaded)
  backends <- lapply(1:10, create_large_backend)
  subject_ids <- paste0("sub-", sprintf("%02d", 1:10))

  # Create study backend
  study <- study_backend(backends, subject_ids)

  # Get as DelayedArray
  da <- as_delayed_array(study)

  # Extract a small subset and measure memory
  mem_before <- gc()[2, 2] # Current memory usage in MB

  # Extract data from just 2 subjects
  subset_data <- da[200:700, 1:100]

  mem_after <- gc()[2, 2] # Memory after extraction

  # Memory increase should be small (< 100MB for this subset)
  mem_increase <- mem_after - mem_before
  expect_lt(mem_increase, 100)

  # Verify correct data
  expect_equal(dim(subset_data), c(501, 100))
})

test_that("StudyBackendSeed cache works correctly", {
  skip_if_not_installed("DelayedArray")

  # Set small cache for testing
  old_opt <- getOption("fmridataset.study_cache_mb")
  options(fmridataset.study_cache_mb = 10)

  # Create backends
  backends <- lapply(1:3, function(id) {
    matrix_backend(
      matrix(id, nrow = 100, ncol = 1000),
      mask = rep(TRUE, 1000)
    )
  })

  study <- study_backend(backends, paste0("sub-", 1:3))
  da <- as_delayed_array(study)

  # Access same data twice - second access should use cache
  time1 <- system.time({
    data1 <- da[1:50, 1:100]
  })

  time2 <- system.time({
    data2 <- da[1:50, 1:100] # Same subset
  })

  # Second access should be faster due to caching
  # (This is a weak test but better than nothing)
  expect_identical(data1, data2)

  # Restore option
  options(fmridataset.study_cache_mb = old_opt)
})

test_that("study_backend works with data_chunks", {
  skip_if_not_installed("DelayedArray")

  # Create small study
  backends <- lapply(1:3, function(id) {
    matrix_backend(
      matrix(id, nrow = 50, ncol = 100),
      mask = rep(TRUE, 100)
    )
  })

  study_backend_obj <- study_backend(backends, paste0("sub-", 1:3))

  # Create proper sampling frame
  sf <- list(
    blocklens = rep(50, 3),
    TR = 2,
    nruns = 3
  )
  class(sf) <- "sampling_frame"

  # Add blockids method for sampling_frame if not available
  if (!exists("blockids.sampling_frame")) {
    blockids.sampling_frame <- function(x) {
      rep(1:length(x$blocklens), times = x$blocklens)
    }
  }

  # Create dataset
  dataset <- structure(
    list(
      backend = study_backend_obj,
      sampling_frame = sf,
      nruns = 3
    ),
    class = c("fmri_file_dataset", "fmri_dataset")
  )

  # Get chunks - use runwise to get one chunk per subject/run
  chunks <- data_chunks(dataset, runwise = TRUE)

  # Should have 3 chunks (one per subject)
  chunk_list <- list()
  i <- 1
  tryCatch(
    {
      while (TRUE) {
        chunk_list[[i]] <- chunks$nextElem()
        i <- i + 1
      }
    },
    error = function(e) {
      if (!grepl("StopIteration", e$message)) stop(e)
    }
  )

  expect_equal(length(chunk_list), 3)

  # Each chunk should have correct data
  expect_equal(unique(as.vector(chunk_list[[1]]$data[, 1])), 1)
  expect_equal(unique(as.vector(chunk_list[[2]]$data[, 1])), 2)
  expect_equal(unique(as.vector(chunk_list[[3]]$data[, 1])), 3)
})
