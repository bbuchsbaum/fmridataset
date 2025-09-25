test_that("backend chunking streams subset of voxels", {
  skip_on_cran()
  skip_if_not_installed("withr")

  ns <- asNamespace("fmridataset")
  registered <- character()
  log_env <- new.env(parent = emptyenv())
  log_env$entries <- matrix(numeric(0), ncol = 2)

  register_method <- function(name, class, fn) {
    base::registerS3method(name, class, fn, envir = ns)
    registered <<- c(registered, paste0(name, ".", class))
  }

  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns)
    rm(list = registered, envir = table)
  })

  passthrough <- function(name) get(paste0(name, ".matrix_backend"), envir = ns)
  register_method("backend_open", "tracked_matrix_backend", passthrough("backend_open"))
  register_method("backend_close", "tracked_matrix_backend", passthrough("backend_close"))
  register_method("backend_get_dims", "tracked_matrix_backend", passthrough("backend_get_dims"))
  register_method("backend_get_mask", "tracked_matrix_backend", passthrough("backend_get_mask"))
  register_method("backend_get_metadata", "tracked_matrix_backend", passthrough("backend_get_metadata"))

  orig_get_data <- get("backend_get_data.matrix_backend", envir = ns)
  register_method("backend_get_data", "tracked_matrix_backend", function(backend, rows = NULL, cols = NULL, ...) {
    res <- orig_get_data(backend, rows = rows, cols = cols, ...)
    log_env$entries <- rbind(log_env$entries, c(nrow(res), ncol(res)))
    res
  })

  create_tracked_backend <- function(data_matrix, mask, spatial_dims) {
    backend <- matrix_backend(data_matrix = data_matrix, mask = mask, spatial_dims = spatial_dims)
    class(backend) <- c("tracked_matrix_backend", class(backend))
    backend
  }

  n_timepoints <- 300
  n_voxels <- 1000
  spatial_dims <- c(10, 10, 10)
  test_data <- matrix(rnorm(n_timepoints * n_voxels), nrow = n_timepoints, ncol = n_voxels)
  mask <- rep(TRUE, n_voxels)

  backend <- create_tracked_backend(test_data, mask, spatial_dims)
  dset <- fmri_dataset(backend, TR = 2, run_length = c(150, 150))

  log_env$entries <- matrix(numeric(0), ncol = 2)
  chunks <- data_chunks(dset, nchunks = 10)
  for (i in 1:10) {
    chunks$nextElem()
  }

  chunk_logs <- log_env$entries
  expect_equal(nrow(chunk_logs), 10)
  expect_true(all(chunk_logs[, 2] <= ceiling(n_voxels / 10)))

  log_env$entries <- matrix(numeric(0), ncol = 2)
  chunks <- data_chunks(dset, runwise = TRUE)
  chunks$nextElem()
  chunks$nextElem()
  run_logs <- log_env$entries
  expect_equal(nrow(run_logs), 2)
  expect_true(all(run_logs[, 1] == 150))
  expect_true(all(run_logs[, 2] == n_voxels))
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
