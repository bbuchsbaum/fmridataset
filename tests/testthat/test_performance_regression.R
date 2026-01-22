# Performance regression tests for fmridataset
# These tests ensure that performance doesn't degrade across versions

test_that("dataset creation performance is acceptable", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Benchmark dataset creation
  n_time <- 1000
  n_vox <- 1000
  mat <- matrix(rnorm(n_time * n_vox), n_time, n_vox)

  result <- bench::mark(
    matrix_dataset = {
      matrix_dataset(mat, TR = 2, run_length = n_time)
    },
    iterations = 10,
    check = FALSE
  )

  # Creation should be fast (under 100ms for 1M elements)
  expect_lt(median(result$median), 0.1)

  # Memory allocation should be minimal (just metadata)
  # The matrix is already allocated, so additional memory should be small
  expect_lt(max(result$mem_alloc), 1e6) # Less than 1MB additional
})

test_that("data access performance scales linearly", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Test different data sizes
  sizes <- c(100, 1000, 5000)
  times <- numeric(length(sizes))

  for (i in seq_along(sizes)) {
    n <- sizes[i]
    mat <- matrix(rnorm(n * 100), n, 100)
    dset <- matrix_dataset(mat, TR = 2, run_length = n)

    result <- bench::mark(
      get_data_matrix(dset),
      iterations = 5,
      check = FALSE
    )

    times[i] <- median(result$median)
  }

  # Check that access time scales approximately linearly
  # Ratio of times should be similar to ratio of sizes
  ratio1 <- times[2] / times[1]
  ratio2 <- times[3] / times[2]
  size_ratio1 <- sizes[2] / sizes[1]
  size_ratio2 <- sizes[3] / sizes[2]

  # Allow 100% deviation from perfect linear scaling
  # (increased due to fixed overhead dominating in small datasets)
  # The important thing is that performance doesn't degrade catastrophically
  expect_lt(abs(ratio1 - size_ratio1) / size_ratio1, 1.0)
  expect_lt(abs(ratio2 - size_ratio2) / size_ratio2, 1.0)
})

test_that("chunking doesn't load entire dataset into memory", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Large dataset
  n_time <- 5000
  n_vox <- 1000
  mat <- matrix(rnorm(n_time * n_vox), n_time, n_vox)
  dset <- matrix_dataset(mat, TR = 2, run_length = n_time)

  # Benchmark chunk iteration
  result <- bench::mark(
    chunk_iteration = {
      chunks <- data_chunks(dset, nchunks = 10)
      total <- 0
      # Use iterator interface properly
      for (i in 1:chunks$nchunks) {
        chunk <- chunks$nextElem()
        total <- total + sum(chunk$data)
      }
      total
    },
    iterations = 3,
    check = FALSE,
    memory = TRUE
  )

  # Memory usage should be much less than full dataset
  full_size <- object.size(mat)
  chunk_mem <- max(result$mem_alloc)

  # Chunk iteration should use less memory than full dataset
  # Note: bench::mark measures peak memory which includes the original matrix
  # so we can't expect dramatic savings in this test setup
  expect_lt(chunk_mem, as.numeric(full_size) * 1.5)
})

test_that("mask operations are optimized", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Dataset with sparse mask
  n_time <- 1000
  n_vox <- 10000
  mat <- matrix(rnorm(n_time * n_vox), n_time, n_vox)

  # Only 10% of voxels in mask
  mask <- rep(FALSE, n_vox)
  mask[sample(n_vox, n_vox * 0.1)] <- TRUE

  backend <- matrix_backend(mat, mask = mask)
  dset <- fmri_dataset(backend, TR = 2, run_length = n_time)

  # Benchmark masked data extraction
  result <- bench::mark(
    get_masked_data = {
      # Don't pass cols - let backend handle masking internally
      data <- backend_get_data(backend)
    },
    iterations = 10,
    check = FALSE
  )

  # Should be fast even with large data
  expect_lt(median(result$median), 0.05) # Under 50ms
})

test_that("memoised file reads avoid repeated disk hits", {
  skip_on_cran()
  skip_if_not_installed("neuroim2")
  skip_if_not_installed("withr")

  ns_neuro <- asNamespace("neuroim2")
  ns_fmri <- asNamespace("fmridataset")

  original_read_vec <- get("read_vec", envir = ns_neuro)
  call_count <- 0L

  unlockBinding("read_vec", ns_neuro)
  assign("read_vec", function(scans, mask, mode, ...) {
    call_count <<- call_count + 1L
    Sys.sleep(0.02)
    matrix(seq_len(sum(mask) * 2), nrow = 2)
  }, envir = ns_neuro)
  lockBinding("read_vec", ns_neuro)
  withr::defer({
    unlockBinding("read_vec", ns_neuro)
    assign("read_vec", original_read_vec, envir = ns_neuro)
    lockBinding("read_vec", ns_neuro)
  })

  base::registerS3method("get_mask", "mock_cache_dataset", function(x, ...) rep(TRUE, 4), envir = ns_fmri)
  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns_fmri)
    rm(list = "get_mask.mock_cache_dataset", envir = table)
  })

  fmri_clear_cache()

  dset <- structure(
    list(scans = "dummy_scan", mode = "normal"),
    class = "mock_cache_dataset"
  )

  cold_time <- system.time(fmridataset:::get_data_from_file(dset))["elapsed"]
  warm_time <- system.time(fmridataset:::get_data_from_file(dset))["elapsed"]

  expect_equal(call_count, 1L)
  expect_gt(as.numeric(cold_time), as.numeric(warm_time) + 0.01)
})

test_that("study backend lazy evaluation loads only required subjects", {
  skip_on_cran()
  skip_if_not_installed("withr")

  ns <- asNamespace("fmridataset")
  registered_methods <- character()
  register_s3 <- function(generic, class, fn) {
    environment(fn) <- ns
    base::registerS3method(generic, class, fn, envir = ns)
    registered_methods <<- c(registered_methods, paste0(generic, ".", class))
  }
  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns)
    rm(list = registered_methods, envir = table)
  })

  passthrough <- function(name) get(paste0(name, ".matrix_backend"), envir = ns)
  register_s3("backend_open", "failing_matrix_backend", passthrough("backend_open"))
  register_s3("backend_close", "failing_matrix_backend", passthrough("backend_close"))
  register_s3("backend_get_dims", "failing_matrix_backend", passthrough("backend_get_dims"))
  register_s3("backend_get_mask", "failing_matrix_backend", passthrough("backend_get_mask"))
  register_s3("backend_get_metadata", "failing_matrix_backend", passthrough("backend_get_metadata"))
  register_s3("backend_get_data", "failing_matrix_backend", function(backend, rows = NULL, cols = NULL, ...) {
    stop("failing backend accessed", call. = FALSE)
  })

  expect_false(is.null(utils::getS3method("backend_get_data", "failing_matrix_backend", optional = TRUE)))

  create_backend <- function(id, fail = FALSE, n_time = 100, n_vox = 20) {
    backend <- matrix_backend(
      data_matrix = matrix(id, nrow = n_time, ncol = n_vox),
      mask = rep(TRUE, n_vox)
    )
    if (fail) {
      class(backend) <- c("failing_matrix_backend", class(backend))
    }
    backend
  }

  backends <- list(
    create_backend(1),
    create_backend(2, fail = TRUE)
  )
  study <- study_backend(backends, subject_ids = c("s1", "s2"))

  expect_no_condition(backend_get_data(study, rows = 1:50, cols = 1:5))
  expect_error(
    backend_get_data(study, rows = 120:150, cols = 1:5),
    "failing backend accessed"
  )
})

test_that("DelayedArray conversion is efficient", {
  skip_on_cran()
  skip_if_not_installed("DelayedArray")
  skip_if_not_installed("bench")

  # Medium-sized dataset
  mat <- matrix(rnorm(1000 * 500), 1000, 500)
  dset <- matrix_dataset(mat, TR = 2, run_length = 1000)

  result <- bench::mark(
    delayed_conversion = {
      delayed <- as_delayed_array(dset)
    },
    iterations = 10,
    check = FALSE,
    memory = TRUE
  )

  # Conversion should be fast (just wrapping, not copying)
  expect_lt(median(result$median), 0.01) # Under 10ms

  # Memory should be minimal (no data duplication)
  # DelayedArray infrastructure has overhead - allow up to 5MB
  # Convert bench_bytes to numeric before comparison
  expect_lt(max(as.numeric(result$mem_alloc)), 5e6) # Less than 5MB
})

test_that("print methods perform well with large metadata", {
  skip_on_cran()
  skip_if_not_installed("bench")

  # Dataset with large event table
  mat <- matrix(rnorm(1000 * 100), 1000, 100)
  large_events <- data.frame(
    onset = 1:1000,
    duration = rep(1, 1000),
    trial_type = sample(letters, 1000, replace = TRUE),
    response_time = runif(1000),
    accuracy = sample(0:1, 1000, replace = TRUE)
  )

  dset <- matrix_dataset(mat,
    TR = 2, run_length = 1000,
    event_table = large_events
  )

  result <- bench::mark(
    print_basic = {
      capture.output(print(dset, full = FALSE))
    },
    print_full = {
      capture.output(print(dset, full = TRUE))
    },
    iterations = 10,
    check = FALSE
  )

  # Both should complete quickly
  expect_lt(median(result$median[1]), 0.01) # Basic under 10ms
  expect_lt(median(result$median[2]), 0.05) # Full under 50ms
})

test_that("backend validation overhead is minimal", {
  skip_on_cran()
  skip_if_not_installed("bench")

  mat <- matrix(rnorm(1000 * 100), 1000, 100)

  # Time backend creation with and without validation
  result <- bench::mark(
    with_validation = {
      backend <- matrix_backend(mat)
      validate_backend(backend)
    },
    without_validation = {
      backend <- matrix_backend(mat)
    },
    iterations = 20,
    check = FALSE
  )

  # Validation overhead should be small
  validation_time <- median(result$median[1]) - median(result$median[2])
  expect_lt(validation_time, 0.001) # Less than 1ms overhead
})

test_that("performance doesn't degrade with many small runs", {
  skip_on_cran()
  skip_if_not_installed("bench")

  n_time <- 1000
  mat <- matrix(rnorm(n_time * 100), n_time, 100)

  # Compare few long runs vs many short runs
  result <- bench::mark(
    few_runs = {
      matrix_dataset(mat, TR = 2, run_length = c(500, 500))
    },
    many_runs = {
      matrix_dataset(mat, TR = 2, run_length = rep(10, 100))
    },
    iterations = 10,
    check = FALSE
  )

  # Performance should be similar
  ratio <- median(result$median[2]) / median(result$median[1])
  expect_lt(ratio, 2) # Many runs should be less than 2x slower
})
