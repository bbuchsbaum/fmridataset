test_that("backend registry integrates with dataset constructors", {
  # Test that dataset constructors work with the registry system
  
  # Test matrix_dataset backend integration
  mat <- matrix(rnorm(100 * 50), 100, 50)
  ds <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  expect_s3_class(ds, "matrix_dataset")
  expect_equal(nrow(get_data_matrix(ds)), 100)
  expect_equal(ncol(get_data_matrix(ds)), 50)
  
  # Test that we can create a matrix backend directly via registry
  backend <- create_backend("matrix", 
                           data_matrix = mat, 
                           mask = rep(TRUE, 50))
  expect_s3_class(backend, c("matrix_backend", "storage_backend"))
  
  # Test that built-in backends are properly registered
  expect_true(is_backend_registered("matrix"))
  expect_true(is_backend_registered("nifti"))
  expect_true(is_backend_registered("h5"))
  expect_true(is_backend_registered("latent"))
  expect_true(is_backend_registered("study"))
  expect_true(is_backend_registered("zarr"))
  
  # Test registry info for built-in backends
  nifti_info <- get_backend_registry("nifti")
  expect_equal(nifti_info$name, "nifti")
  expect_true(!is.null(nifti_info$factory))
  expect_true(!is.null(nifti_info$description))
  expect_true(grepl("NIfTI", nifti_info$description, ignore.case = TRUE))
})

test_that("custom backend can be used in dataset creation", {
  # Create a custom simple backend for testing
  simple_backend_factory <- function(data_matrix, mask = NULL, ...) {
    if (is.null(mask)) {
      mask <- rep(TRUE, ncol(data_matrix))
    }
    
    backend <- list(
      data_matrix = data_matrix,
      mask = mask,
      spatial_dims = c(ncol(data_matrix), 1, 1)
    )
    
    class(backend) <- c("simple_test_backend", "storage_backend")
    backend
  }
  
  # Register S3 methods
  assign("backend_open.simple_test_backend", function(backend) backend, envir = globalenv())
  assign("backend_close.simple_test_backend", function(backend) invisible(NULL), envir = globalenv())
  assign("backend_get_dims.simple_test_backend", function(backend) {
    list(spatial = backend$spatial_dims, time = nrow(backend$data_matrix))
  }, envir = globalenv())
  assign("backend_get_mask.simple_test_backend", function(backend) backend$mask, envir = globalenv())
  assign("backend_get_data.simple_test_backend", function(backend, rows = NULL, cols = NULL) {
    data <- backend$data_matrix[, backend$mask, drop = FALSE]
    if (!is.null(rows)) data <- data[rows, , drop = FALSE]
    if (!is.null(cols)) data <- data[, cols, drop = FALSE]
    data
  }, envir = globalenv())
  assign("backend_get_metadata.simple_test_backend", function(backend) {
    list(format = "simple_test")
  }, envir = globalenv())
  
  on.exit({
    unregister_backend("simple_test")
    cleanup_functions <- c("backend_open.simple_test_backend",
                          "backend_close.simple_test_backend", 
                          "backend_get_dims.simple_test_backend",
                          "backend_get_mask.simple_test_backend",
                          "backend_get_data.simple_test_backend", 
                          "backend_get_metadata.simple_test_backend")
    for (func in cleanup_functions) {
      if (exists(func, envir = globalenv())) {
        rm(list = func, envir = globalenv())
      }
    }
  })
  
  # Register the backend
  register_backend("simple_test", simple_backend_factory, "Simple test backend")
  
  # Test that it's registered
  expect_true(is_backend_registered("simple_test"))
  
  # Create test data
  test_data <- matrix(rnorm(50 * 20), 50, 20)
  
  # Create backend instance
  backend <- create_backend("simple_test", data_matrix = test_data)
  expect_s3_class(backend, c("simple_test_backend", "storage_backend"))
  
  # Test that backend works
  expect_equal(backend_get_dims(backend)$time, 50)
  expect_equal(prod(backend_get_dims(backend)$spatial), 20)
  expect_length(backend_get_mask(backend), 20)
  expect_true(all(backend_get_mask(backend)))
  
  # Test data access
  data <- backend_get_data(backend)
  expect_equal(dim(data), c(50, 20))
  
  # Test that the backend can be used in fmri_dataset
  dataset <- fmri_dataset(backend, TR = 2, run_length = 50)
  expect_s3_class(dataset, "fmri_file_dataset")
  expect_equal(n_timepoints(dataset), 50)
  
  # Test data access through dataset
  ds_data <- get_data_matrix(dataset)
  expect_equal(dim(ds_data), c(50, 20))
})

test_that("backward compatibility is maintained", {
  # Test that existing code still works without using the registry directly
  
  # Create matrix backend the old way
  mat <- matrix(rnorm(30 * 10), 30, 10)
  backend_old <- matrix_backend(mat, mask = rep(TRUE, 10))
  expect_s3_class(backend_old, c("matrix_backend", "storage_backend"))
  
  # Create the same backend through registry
  backend_new <- create_backend("matrix", data_matrix = mat, mask = rep(TRUE, 10))
  expect_s3_class(backend_new, c("matrix_backend", "storage_backend"))
  
  # Both should work the same way
  expect_equal(backend_get_dims(backend_old), backend_get_dims(backend_new))
  expect_equal(backend_get_mask(backend_old), backend_get_mask(backend_new))
  
  # Old dataset constructors should still work
  ds1 <- matrix_dataset(mat, TR = 2, run_length = 30)
  expect_s3_class(ds1, "matrix_dataset")
  expect_equal(nrow(get_data_matrix(ds1)), 30)
  expect_equal(ncol(get_data_matrix(ds1)), 10)
})