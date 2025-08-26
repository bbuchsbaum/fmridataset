# Error handling robustness tests

test_that("all custom error classes are properly thrown", {
  # Test fmridataset:::fmridataset_error_backend_io
  mock_backend <- structure(
    list(source = "/nonexistent/file.nii"),
    class = c("failing_backend", "storage_backend")
  )

  backend_open.failing_backend <- function(backend) {
    stop(fmridataset:::fmridataset_error_backend_io(
      "Failed to open file",
      file = backend$source,
      operation = "open"
    ))
  }
  registerS3method("backend_open", "failing_backend", backend_open.failing_backend)

  err <- tryCatch(
    backend_open(mock_backend),
    error = function(e) e
  )

  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_s3_class(err, "fmridataset_error")
  expect_equal(err$file, "/nonexistent/file.nii")
  expect_equal(err$operation, "open")

  # Test fmridataset:::fmridataset_error_config
  err2 <- tryCatch(
    stop(fmridataset:::fmridataset_error_config(
      "Invalid parameter value",
      parameter = "TR",
      value = -1
    )),
    error = function(e) e
  )

  expect_s3_class(err2, "fmridataset_error_config")
  expect_equal(err2$parameter, "TR")
  expect_equal(err2$value, -1)
})

test_that("error messages are informative and actionable", {
  # Bad dimensions
  mat <- matrix(1:20, 10, 2)
  err <- tryCatch(
    matrix_dataset(mat, TR = 2, run_length = 5),
    error = function(e) e
  )

  # Message should indicate the problem
  expect_match(err$message, "run_length|nrow", ignore.case = TRUE)

  # Backend errors should include context
  backend <- matrix_backend(mat)
  err2 <- tryCatch(
    backend_get_data(backend, rows = 11:15),
    error = function(e) e
  )

  # Should mention bounds or indices
  expect_true(
    grepl("indices|bounds|rows", err2$message, ignore.case = TRUE) ||
      grepl("subscript", err2$message, ignore.case = TRUE)
  )
})

test_that("stack traces are clean and helpful", {
  # Create a deeply nested error
  level3 <- function() {
    stop(fmridataset:::fmridataset_error_config(
      "Deep error",
      level = 3
    ))
  }

  level2 <- function() level3()
  level1 <- function() level2()

  err <- tryCatch(level1(), error = function(e) e)

  # Error should have clean class hierarchy
  expect_s3_class(err, "fmridataset_error_config")
  expect_s3_class(err, "error")

  # Should have useful info
  expect_equal(err$level, 3)
})

test_that("recovery from partial failures works correctly", {
  # Create datasets where some succeed and some fail
  datasets <- list()

  # Good dataset
  mat1 <- matrix(rnorm(100 * 50), 100, 50)
  datasets[[1]] <- matrix_dataset(mat1, TR = 2, run_length = 100)

  # Another good dataset
  mat2 <- matrix(rnorm(100 * 50), 100, 50)
  datasets[[2]] <- matrix_dataset(mat2, TR = 2, run_length = 100)

  # Bad dataset (would fail in study creation)
  datasets[[3]] <- structure(
    list(sampling_frame = list(TR = 3)), # Different TR
    class = c("matrix_dataset", "fmri_dataset", "list")
  )

  # Study creation should fail gracefully
  err <- tryCatch(
    fmri_study_dataset(datasets),
    error = function(e) e
  )

  expect_error(fmri_study_dataset(datasets))

  # But first two datasets should still be valid
  expect_s3_class(datasets[[1]], "fmri_dataset")
  expect_s3_class(datasets[[2]], "fmri_dataset")
})

test_that("graceful degradation when optional features unavailable", {
  # Test behavior when optional packages missing
  skip_if_not_installed("mockr")

  # Use mockr to mock missing package
  mockr::local_mock(
    requireNamespace = function(package, ...) {
      if (package == "DelayedArray") {
        return(FALSE)
      }
      base::requireNamespace(package, ...)
    }
  )

  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)

  # Should error gracefully
  expect_error(
    as_delayed_array(dset),
    "DelayedArray"
  )
})

test_that("errors during chunking are handled properly", {
  # Create a backend that fails partway through
  failing_backend <- structure(
    list(
      data = matrix(rnorm(100 * 50), 100, 50),
      fail_after = 3
    ),
    class = c("chunk_failing_backend", "storage_backend")
  )

  # Add methods
  backend_open.chunk_failing_backend <- function(backend) backend
  backend_close.chunk_failing_backend <- function(backend) invisible(NULL)
  backend_get_dims.chunk_failing_backend <- function(backend) {
    list(spatial = c(ncol(backend$data), 1, 1), time = nrow(backend$data))
  }
  backend_get_mask.chunk_failing_backend <- function(backend) {
    rep(TRUE, ncol(backend$data))
  }

  # Counter for calls
  call_count <- 0
  backend_get_data.chunk_failing_backend <- function(backend, rows = NULL, cols = NULL) {
    call_count <<- call_count + 1
    if (call_count > backend$fail_after) {
      stop("Simulated failure during chunking")
    }
    backend$data[rows %||% TRUE, cols %||% TRUE, drop = FALSE]
  }

  backend_get_metadata.chunk_failing_backend <- function(backend) list()

  # Register methods
  for (method in c("open", "close", "get_dims", "get_mask", "get_data", "get_metadata")) {
    registerS3method(
      paste0("backend_", method),
      "chunk_failing_backend",
      get(paste0("backend_", method, ".chunk_failing_backend"))
    )
  }

  # Create dataset
  dset <- fmri_dataset(failing_backend, TR = 2, run_length = 100)

  # Chunking should eventually fail
  chunks <- data_chunks(dset, nchunks = 5)

  chunk_count <- 0
  expect_error(
    {
      for (i in seq_len(chunks$nchunks)) {
        chunk <- chunks$nextElem()
        chunk_count <- chunk_count + 1
        sum(chunk$data)
      }
    },
    "Simulated failure"
  )

  # Should have processed some chunks before failing
  expect_gt(chunk_count, 0)
  expect_lt(chunk_count, 5)
})

test_that("validation errors provide helpful suggestions", {
  # Test various validation failures

  # Empty mask
  mat <- matrix(rnorm(100 * 50), 100, 50)
  backend <- matrix_backend(mat, mask = rep(FALSE, 50))
  err <- tryCatch(
    fmridataset:::validate_backend(backend),
    error = function(e) e
  )

  expect_s3_class(err, "error")
  expect_match(as.character(err), "at least one TRUE value", ignore.case = TRUE)

  # Dimension mismatch
  err2 <- tryCatch(
    matrix_backend(mat, mask = rep(TRUE, 25)), # Wrong length
    error = function(e) e
  )

  expect_match(err2$message, "length|must equal", ignore.case = TRUE)
})

test_that("file I/O errors are caught and wrapped properly", {
  # Non-existent file
  err <- tryCatch(
    {
      # Simulate file backend trying to read
      stop(fmridataset:::fmridataset_error_backend_io(
        "Cannot read file: No such file or directory",
        file = "/path/to/missing.nii",
        operation = "read"
      ))
    },
    error = function(e) e
  )

  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_equal(err$operation, "read")

  # Permission denied
  err2 <- tryCatch(
    {
      stop(fmridataset:::fmridataset_error_backend_io(
        "Cannot write file: Permission denied",
        file = "/root/protected.nii",
        operation = "write"
      ))
    },
    error = function(e) e
  )

  expect_s3_class(err2, "fmridataset_error_backend_io")
  expect_equal(err2$operation, "write")
})

test_that("error recovery preserves system state", {
  # Create a backend that modifies state then fails
  state_log <- character()

  stateful_backend <- structure(
    list(data = matrix(1:20, 10, 2)),
    class = c("stateful_backend", "storage_backend")
  )

  backend_open.stateful_backend <- function(backend) {
    state_log <<- c(state_log, "opened")
    backend$is_open <- TRUE
    backend
  }

  backend_close.stateful_backend <- function(backend) {
    state_log <<- c(state_log, "closed")
    backend$is_open <- FALSE
    invisible(NULL)
  }

  backend_get_dims.stateful_backend <- function(backend) {
    state_log <<- c(state_log, "dims_accessed")
    stop("Simulated failure in get_dims")
  }

  # Register methods
  registerS3method("backend_open", "stateful_backend", backend_open.stateful_backend)
  registerS3method("backend_close", "stateful_backend", backend_close.stateful_backend)
  registerS3method("backend_get_dims", "stateful_backend", backend_get_dims.stateful_backend)

  # Add required methods
  backend_get_mask.stateful_backend <- function(backend) {
    state_log <<- c(state_log, "mask_accessed")
    rep(TRUE, 10)
  }
  backend_get_data.stateful_backend <- function(backend, ...) {
    matrix(1:10, 5, 2)
  }
  backend_get_metadata.stateful_backend <- function(backend) list()

  registerS3method("backend_get_mask", "stateful_backend", backend_get_mask.stateful_backend)
  registerS3method("backend_get_data", "stateful_backend", backend_get_data.stateful_backend)
  registerS3method("backend_get_metadata", "stateful_backend", backend_get_metadata.stateful_backend)

  # validate_backend calls backend_get_dims and backend_get_mask but not open/close
  expect_error(
    fmridataset:::validate_backend(stateful_backend),
    "Simulated failure in get_dims"
  )

  # Check that dims was accessed before the failure
  expect_true("dims_accessed" %in% state_log)
  expect_false("opened" %in% state_log)
  expect_false("closed" %in% state_log)
})

test_that("concurrent errors don't cause corruption", {
  skip_on_cran()
  skip_if_not_installed("parallel")

  # Dataset that tracks access
  access_log <- list()

  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)

  # Function that sometimes errors
  risky_operation <- function(dset, id) {
    if (runif(1) > 0.5) {
      stop(paste("Random failure in process", id))
    }
    sum(get_data_matrix(dset))
  }

  # Run parallel operations with potential failures
  if (.Platform$OS.type != "windows") {
    results <- parallel::mclapply(1:10, function(i) {
      tryCatch(
        risky_operation(dset, i),
        error = function(e) list(error = e$message)
      )
    }, mc.cores = 2)

    # Some should succeed, some should fail
    successes <- sum(sapply(results, is.numeric))
    failures <- sum(sapply(results, is.list))

    expect_gt(successes, 0)
    expect_gt(failures, 0)

    # Dataset should still be valid
    expect_equal(dim(get_data_matrix(dset)), c(100, 50))
  }
})

test_that("error messages handle special characters correctly", {
  # Path with special characters
  weird_path <- "/path/with spaces/and'quotes\"/and\\backslashes"

  err <- tryCatch(
    stop(fmridataset:::fmridataset_error_backend_io(
      sprintf("Cannot open file: %s", weird_path),
      file = weird_path,
      operation = "open"
    )),
    error = function(e) e
  )

  # Message should preserve the path correctly
  expect_match(err$message, "spaces", fixed = TRUE)
  expect_equal(err$file, weird_path)

  # Unicode in error messages
  unicode_msg <- "Failed to process café_naïve_日本語.nii"
  err2 <- tryCatch(
    stop(unicode_msg),
    error = function(e) e
  )

  expect_match(err2$message, "café", fixed = TRUE)
})
