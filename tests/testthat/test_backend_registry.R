test_that("backend registry system works correctly", {
  # Clear registry for clean testing
  old_registry <- get_backend_registry()
  on.exit({
    # Restore original registry
    for (name in list_backend_names()) {
      unregister_backend(name)
    }
    for (name in names(old_registry)) {
      do.call(register_backend, old_registry[[name]][c("name", "factory", "description", "validate_function")])
    }
  })
  
  # Start with clean registry
  for (name in list_backend_names()) {
    unregister_backend(name)
  }
  
  expect_equal(length(list_backend_names()), 0)
  expect_false(is_backend_registered("test"))
  
  # Test basic registration
  test_factory <- function(data, ...) {
    backend <- list(data = data, ...)
    class(backend) <- c("test_backend", "storage_backend")
    backend
  }
  
  expect_true(register_backend("test", test_factory, "Test backend"))
  expect_true(is_backend_registered("test"))
  expect_equal(list_backend_names(), "test")
  
  # Test registration details
  reg_info <- get_backend_registry("test")
  expect_equal(reg_info$name, "test")
  expect_equal(reg_info$description, "Test backend")
  expect_identical(reg_info$factory, test_factory)
  expect_null(reg_info$validate_function)
  expect_true(inherits(reg_info$registered_at, "POSIXt"))
  
  # Test duplicate registration protection
  expect_error(register_backend("test", test_factory), "already registered")
  expect_true(register_backend("test", test_factory, overwrite = TRUE))
  
  # Test custom validation function
  test_validator <- function(backend) {
    if (!is.list(backend$data)) {
      stop("data must be a list")
    }
    TRUE
  }
  
  expect_true(register_backend("test_validated", test_factory, 
                              validate_function = test_validator))
  
  # Test unregistration
  expect_true(unregister_backend("test"))
  expect_false(is_backend_registered("test"))
  expect_false(unregister_backend("nonexistent"))
})

test_that("backend creation works correctly", {
  # Register a simple test backend
  test_factory <- function(value = 42, ...) {
    backend <- list(value = value, extra = list(...))
    class(backend) <- c("test_create_backend", "storage_backend")
    backend
  }
  
  # Add minimal S3 methods to satisfy validation
  assign("backend_open.test_create_backend", function(backend) backend, envir = globalenv())
  assign("backend_close.test_create_backend", function(backend) invisible(NULL), envir = globalenv())
  assign("backend_get_dims.test_create_backend", function(backend) {
    list(spatial = c(10, 10, 10), time = 100)
  }, envir = globalenv())
  assign("backend_get_mask.test_create_backend", function(backend) {
    rep(TRUE, 1000)
  }, envir = globalenv())
  assign("backend_get_data.test_create_backend", function(backend, rows = NULL, cols = NULL) {
    matrix(rnorm(100 * 1000), 100, 1000)
  }, envir = globalenv())
  assign("backend_get_metadata.test_create_backend", function(backend) {
    list(test = TRUE)
  }, envir = globalenv())
  
  on.exit({
    unregister_backend("test_create")
    if (exists("backend_open.test_create_backend", envir = globalenv())) {
      rm("backend_open.test_create_backend", envir = globalenv())
    }
    if (exists("backend_close.test_create_backend", envir = globalenv())) {
      rm("backend_close.test_create_backend", envir = globalenv())
    }
    if (exists("backend_get_dims.test_create_backend", envir = globalenv())) {
      rm("backend_get_dims.test_create_backend", envir = globalenv())
    }
    if (exists("backend_get_mask.test_create_backend", envir = globalenv())) {
      rm("backend_get_mask.test_create_backend", envir = globalenv())
    }
    if (exists("backend_get_data.test_create_backend", envir = globalenv())) {
      rm("backend_get_data.test_create_backend", envir = globalenv())
    }
    if (exists("backend_get_metadata.test_create_backend", envir = globalenv())) {
      rm("backend_get_metadata.test_create_backend", envir = globalenv())
    }
  })
  
  register_backend("test_create", test_factory, "Test creation backend")
  
  # Test basic creation
  backend <- create_backend("test_create")
  expect_s3_class(backend, c("test_create_backend", "storage_backend"))
  expect_equal(backend$value, 42)
  
  # Test creation with parameters
  backend2 <- create_backend("test_create", value = 100, extra_param = "test")
  expect_equal(backend2$value, 100)
  expect_equal(backend2$extra$extra_param, "test")
  
  # Test creation with validation disabled
  backend3 <- create_backend("test_create", validate = FALSE)
  expect_s3_class(backend3, c("test_create_backend", "storage_backend"))
  
  # Test error for unregistered backend
  expect_error(create_backend("nonexistent"), "not registered")
})

test_that("backend validation works correctly", {
  # Test with invalid backend factory
  bad_factory <- function(...) {
    list(bad = TRUE)  # Missing storage_backend class
  }
  
  register_backend("bad_test", bad_factory, overwrite = TRUE)
  on.exit(unregister_backend("bad_test"))
  
  # Should fail validation
  expect_error(create_backend("bad_test"), "storage_backend")
  
  # Test with custom validation
  good_factory <- function(data = NULL, ...) {
    backend <- list(data = data, ...)
    class(backend) <- c("validated_backend", "storage_backend")
    backend
  }
  
  # Add minimal S3 methods
  assign("backend_open.validated_backend", function(backend) backend, envir = globalenv())
  assign("backend_close.validated_backend", function(backend) invisible(NULL), envir = globalenv())
  assign("backend_get_dims.validated_backend", function(backend) {
    list(spatial = c(10, 10, 10), time = 100)
  }, envir = globalenv())
  assign("backend_get_mask.validated_backend", function(backend) {
    rep(TRUE, 1000)
  }, envir = globalenv())
  assign("backend_get_data.validated_backend", function(backend, rows = NULL, cols = NULL) {
    matrix(rnorm(100 * 1000), 100, 1000)
  }, envir = globalenv())
  assign("backend_get_metadata.validated_backend", function(backend) {
    list(test = TRUE)
  }, envir = globalenv())
  
  custom_validator <- function(backend) {
    if (is.null(backend$data)) {
      stop("data is required")
    }
    TRUE
  }
  
  on.exit({
    unregister_backend("validated_test")
    if (exists("backend_open.validated_backend", envir = globalenv())) {
      rm("backend_open.validated_backend", envir = globalenv())
    }
    if (exists("backend_close.validated_backend", envir = globalenv())) {
      rm("backend_close.validated_backend", envir = globalenv())
    }
    if (exists("backend_get_dims.validated_backend", envir = globalenv())) {
      rm("backend_get_dims.validated_backend", envir = globalenv())
    }
    if (exists("backend_get_mask.validated_backend", envir = globalenv())) {
      rm("backend_get_mask.validated_backend", envir = globalenv())
    }
    if (exists("backend_get_data.validated_backend", envir = globalenv())) {
      rm("backend_get_data.validated_backend", envir = globalenv())
    }
    if (exists("backend_get_metadata.validated_backend", envir = globalenv())) {
      rm("backend_get_metadata.validated_backend", envir = globalenv())
    }
  }, add = TRUE)
  
  register_backend("validated_test", good_factory, 
                  validate_function = custom_validator)
  
  # Should pass with data
  backend <- create_backend("validated_test", data = list(x = 1:10))
  expect_s3_class(backend, "validated_backend")
  
  # Should fail custom validation
  expect_error(create_backend("validated_test"), "data is required")
})

test_that("registry utility functions work correctly", {
  # Clear and set up test backends
  old_names <- list_backend_names()
  for (name in old_names) {
    unregister_backend(name)
  }
  
  on.exit({
    # Clean up
    for (name in list_backend_names()) {
      unregister_backend(name)
    }
    # Re-register built-ins
    register_builtin_backends()
  })
  
  # Test empty registry
  expect_equal(length(list_backend_names()), 0)
  expect_equal(length(get_backend_registry()), 0)
  
  # Register some test backends
  test_factory1 <- function(...) list(type = "test1", ...)
  test_factory2 <- function(...) list(type = "test2", ...)
  
  register_backend("test1", test_factory1, "First test")
  register_backend("test2", test_factory2, "Second test")
  
  # Test listing
  names <- list_backend_names()
  expect_equal(length(names), 2)
  expect_true(all(c("test1", "test2") %in% names))
  
  # Test getting all registrations
  all_regs <- get_backend_registry()
  expect_equal(length(all_regs), 2)
  expect_equal(names(all_regs), names)
  expect_equal(all_regs$test1$description, "First test")
  expect_equal(all_regs$test2$description, "Second test")
  
  # Test error for nonexistent backend
  expect_error(get_backend_registry("nonexistent"), "not registered")
})

test_that("built-in backends are registered correctly", {
  # Register built-ins (should be idempotent)
  expect_silent(register_builtin_backends())
  
  # Check that key built-in backends are registered
  builtin_names <- list_backend_names()
  expected_builtins <- c("nifti", "h5", "matrix", "latent", "study", "zarr")
  
  for (backend_name in expected_builtins) {
    expect_true(is_backend_registered(backend_name), 
                info = paste("Backend", backend_name, "should be registered"))
    
    reg_info <- get_backend_registry(backend_name)
    expect_false(is.null(reg_info$factory))
    expect_false(is.null(reg_info$description))
  }
  
  # Test that factories correspond to expected functions
  expect_identical(get_backend_registry("nifti")$factory, nifti_backend)
  expect_identical(get_backend_registry("h5")$factory, h5_backend)
  expect_identical(get_backend_registry("matrix")$factory, matrix_backend)
  expect_identical(get_backend_registry("latent")$factory, latent_backend)
  expect_identical(get_backend_registry("study")$factory, study_backend)
  expect_identical(get_backend_registry("zarr")$factory, zarr_backend)
})

test_that("input validation works correctly", {
  # Test invalid names
  expect_error(register_backend("", function() NULL), "non-empty character string")
  expect_error(register_backend(NULL, function() NULL), "non-empty character string")
  expect_error(register_backend(c("a", "b"), function() NULL), "non-empty character string")
  
  # Test invalid factory
  expect_error(register_backend("test", "not a function"), "factory must be a function")
  expect_error(register_backend("test", NULL), "factory must be a function")
  
  # Test invalid description
  expect_error(register_backend("test", function() NULL, c("a", "b")), 
               "description must be a character string")
  
  # Test invalid validate_function
  expect_error(register_backend("test", function() NULL, validate_function = "not a function"),
               "validate_function must be a function")
  
  # Test valid registration with NULL optional parameters
  expect_silent(register_backend("test_null", function() NULL))
  expect_true(is_backend_registered("test_null"))
  
  on.exit(unregister_backend("test_null"))
  
  # Test is_backend_registered with invalid input
  expect_false(is_backend_registered(NULL))
  expect_false(is_backend_registered(c("a", "b")))
  expect_false(is_backend_registered(""))
})

test_that("print method works correctly", {
  # Create a test registry
  test_registry <- list(
    test1 = list(
      name = "test1",
      description = "Test backend 1",
      registered_at = as.POSIXct("2024-01-01 12:00:00"),
      validate_function = NULL
    ),
    test2 = list(
      name = "test2", 
      description = "Test backend 2",
      registered_at = as.POSIXct("2024-01-01 13:00:00"),
      validate_function = function(x) TRUE
    )
  )
  class(test_registry) <- "backend_registry"
  
  # Test printing with backends
  output <- capture.output(print(test_registry))
  expect_true(any(grepl("Registered Storage Backends", output)))
  expect_true(any(grepl("Backend: test1", output)))
  expect_true(any(grepl("Backend: test2", output)))
  expect_true(any(grepl("Custom validation: Yes", output)))
  
  # Test printing empty registry
  empty_registry <- list()
  class(empty_registry) <- "backend_registry"
  output <- capture.output(print(empty_registry))
  expect_true(any(grepl("No backends registered", output)))
})