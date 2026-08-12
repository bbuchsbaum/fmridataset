# Golden tests for storage backends

test_that("matrix backend produces consistent output", {
  ref_data <- load_golden_data("reference_data")

  # Create backend directly
  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  # Test backend interface
  expect_s3_class(backend, "matrix_backend")

  # Test data retrieval
  backend_open(backend)

  # Get dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims$time, nrow(ref_data$matrix_data))
  expect_equal(dims$spatial, c(ncol(ref_data$matrix_data), 1, 1))

  # Get full data
  full_data <- backend_get_data(backend)
  compare_golden(full_data, ref_data$matrix_data)

  # Get subset
  subset_data <- backend_get_data(backend,
    rows = 1:10,
    cols = 1:20
  )
  expected_subset <- ref_data$matrix_data[1:10, 1:20]
  compare_golden(subset_data, expected_subset)

  backend_close(backend)
})

test_that("multi-run matrix backend handles correctly", {
  ref_data <- load_golden_data("reference_data")

  # Create multi-run backend - concatenate runs
  combined_data <- do.call(rbind, ref_data$multirun_data)
  backend <- matrix_backend(
    data_matrix = combined_data
  )

  backend_open(backend)

  # Test dimensions
  dims <- backend_get_dims(backend)
  expected_nrow <- sum(sapply(ref_data$multirun_data, nrow))
  expected_ncol <- ncol(ref_data$multirun_data[[1]]) # all runs have same ncol
  expect_equal(dims$spatial, c(expected_ncol, 1, 1))
  expect_equal(dims$time, expected_nrow)

  # Test run boundaries
  run_idx <- 1
  row_start <- 1
  for (run_data in ref_data$multirun_data) {
    row_end <- row_start + nrow(run_data) - 1

    subset <- backend_get_data(backend, rows = row_start:row_end)
    compare_golden(subset, run_data)

    row_start <- row_end + 1
    run_idx <- run_idx + 1
  }

  backend_close(backend)
})

test_that("backend metadata is consistent", {
  ref_data <- load_golden_data("reference_data")

  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  # Test backend structure
  expect_true(is.list(backend))
  expect_true("data_matrix" %in% names(backend))
  expect_true("mask" %in% names(backend))
})

test_that("backend validation works consistently", {
  ref_data <- load_golden_data("reference_data")

  # Valid backend
  valid_backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  expect_silent(fmridataset:::validate_backend(valid_backend))

  # Test invalid backends
  expect_error(
    matrix_backend(data_matrix = "not a matrix"),
    class = "fmridataset_error"
  )

  expect_error(
    matrix_backend(data_matrix = data.frame(a = 1:10)),
    "must be a matrix"
  )
})

test_that("backend print output matches snapshot", {
  skip_if(testthat::edition_get() < 3, "Snapshot tests require testthat edition 3")

  ref_data <- load_golden_data("reference_data")

  backend <- matrix_backend(
    data_matrix = ref_data$matrix_data
  )

  expect_snapshot({
    print(backend)
  })
})

test_that("mock NeuroVec backend works correctly", {
  skip_if_not_installed("withr")

  mock_vec <- load_golden_data("mock_neurvec")
  backend <- mock_vec
  class(backend) <- c("mock_neurovec_backend", class(mock_vec))

  ns <- asNamespace("fmridataset")
  registered <- character()

  register_backend_method <- function(generic, fn) {
    environment(fn) <- ns
    base::registerS3method(generic, "mock_neurovec_backend", fn, envir = ns)
    registered <<- c(registered, paste0(generic, ".mock_neurovec_backend"))
  }

  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns)
    rm(list = registered, envir = table)
  })

  register_backend_method("backend_open", function(backend, ...) backend)
  register_backend_method("backend_close", function(backend, ...) invisible(NULL))
  register_backend_method("backend_get_dims", function(backend, ...) {
    arr_dims <- dim(as.array(backend))
    list(
      spatial = as.numeric(arr_dims[1:3]),
      time = as.integer(arr_dims[4])
    )
  })
  register_backend_method("backend_get_mask", function(backend, ...) {
    arr_dims <- dim(as.array(backend))
    rep(TRUE, prod(arr_dims[1:3]))
  })
  register_backend_method("backend_get_metadata", function(backend, ...) {
    list(storage_format = "mock_neurovec", n_runs = 1)
  })
  register_backend_method("backend_get_data", function(backend, rows = NULL, cols = NULL, ...) {
    arr <- as.array(backend)
    arr_dims <- dim(arr)
    time_dim <- arr_dims[4]
    spatial_dims <- prod(arr_dims[1:3])
    mat <- matrix(aperm(arr, c(4, 1, 2, 3)), nrow = time_dim, ncol = spatial_dims)
    if (is.null(rows)) rows <- seq_len(nrow(mat))
    if (is.null(cols)) cols <- seq_len(ncol(mat))
    mat[rows, cols, drop = FALSE]
  })

  backend_open(backend)

  dims <- backend_get_dims(backend)
  expected_dims <- dim(as.array(mock_vec))
  expected_spatial <- as.numeric(expected_dims[1:3])
  expected_time <- as.integer(expected_dims[4])
  expect_equal(dims$spatial, expected_spatial)
  expect_equal(dims$time, expected_time)

  full_data <- backend_get_data(backend)
  expect_equal(dim(full_data), c(expected_time, prod(expected_spatial)))

  subset_data <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(subset_data), c(5, 10))

  backend_close(backend)
})

test_that("backend edge cases handle correctly", {
  # Single voxel
  single_voxel <- matrix(rnorm(50), nrow = 1, ncol = 50)
  backend_sv <- matrix_backend(data_matrix = single_voxel)

  backend_open(backend_sv)
  dims <- backend_get_dims(backend_sv)
  expect_equal(dims$time, 1)
  expect_equal(prod(dims$spatial), 50)

  data <- backend_get_data(backend_sv)
  compare_golden(data, single_voxel)
  backend_close(backend_sv)

  # Single timepoint
  single_time <- matrix(rnorm(100), nrow = 100, ncol = 1)
  backend_st <- matrix_backend(data_matrix = single_time)

  backend_open(backend_st)
  dims <- backend_get_dims(backend_st)
  expect_equal(dims$time, 100)
  expect_equal(prod(dims$spatial), 1)

  data <- backend_get_data(backend_st)
  compare_golden(data, single_time)
  backend_close(backend_st)
})
