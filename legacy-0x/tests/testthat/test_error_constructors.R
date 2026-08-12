library(fmridataset)

test_that("error constructors create structured errors", {
  err <- fmridataset:::fmridataset_error_backend_io("oops", file = "f.h5", operation = "read")
  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_match(err$message, "oops")
  expect_equal(err$file, "f.h5")
  expect_equal(err$operation, "read")

  err2 <- fmridataset:::fmridataset_error_config("bad", parameter = "x", value = 1)
  expect_s3_class(err2, "fmridataset_error_config")
  expect_equal(err2$parameter, "x")
  expect_equal(err2$value, 1)
})


test_that("stop_fmridataset throws the constructed error", {
  expect_error(
    fmridataset:::stop_fmridataset(fmridataset:::fmridataset_error_config, "bad", parameter = "y"),
    class = "fmridataset_error_config"
  )
})

test_that("fmridataset_error base constructor works", {
  err <- fmridataset:::fmridataset_error("base error", class = "test_error")
  expect_s3_class(err, "test_error")
  expect_s3_class(err, "fmridataset_error")
  expect_s3_class(err, "error")
  expect_s3_class(err, "condition")
  expect_equal(err$message, "base error")
})

test_that("error constructors handle optional parameters", {
  # Backend I/O error with minimal parameters
  err1 <- fmridataset:::fmridataset_error_backend_io("minimal error")
  expect_s3_class(err1, "fmridataset_error_backend_io")
  expect_equal(err1$message, "minimal error")
  expect_null(err1$file)
  expect_null(err1$operation)

  # Backend I/O error with all parameters
  err2 <- fmridataset:::fmridataset_error_backend_io(
    "full error",
    file = "data.h5",
    operation = "write",
    additional_info = "extra context"
  )
  expect_equal(err2$file, "data.h5")
  expect_equal(err2$operation, "write")
  expect_equal(err2$additional_info, "extra context")

  # Config error with minimal parameters
  err3 <- fmridataset:::fmridataset_error_config("config issue")
  expect_s3_class(err3, "fmridataset_error_config")
  expect_null(err3$parameter)
  expect_null(err3$value)

  # Config error with all parameters
  err4 <- fmridataset:::fmridataset_error_config(
    "invalid config",
    parameter = "TR",
    value = -1,
    expected = "positive number"
  )
  expect_equal(err4$parameter, "TR")
  expect_equal(err4$value, -1)
  expect_equal(err4$expected, "positive number")
})

test_that("error constructors preserve class hierarchy", {
  backend_err <- fmridataset:::fmridataset_error_backend_io("io error")
  config_err <- fmridataset:::fmridataset_error_config("config error")

  # Both should inherit from fmridataset_error
  expect_true(inherits(backend_err, "fmridataset_error"))
  expect_true(inherits(config_err, "fmridataset_error"))

  # Both should inherit from error and condition
  expect_true(inherits(backend_err, "error"))
  expect_true(inherits(backend_err, "condition"))
  expect_true(inherits(config_err, "error"))
  expect_true(inherits(config_err, "condition"))

  # But should be distinguishable
  expect_false(inherits(backend_err, "fmridataset_error_config"))
  expect_false(inherits(config_err, "fmridataset_error_backend_io"))
})

test_that("stop_fmridataset handles complex error scenarios", {
  # Test with file list in backend error
  expect_error(
    fmridataset:::stop_fmridataset(
      fmridataset:::fmridataset_error_backend_io,
      message = "Multiple files failed",
      file = c("file1.nii", "file2.nii", "file3.nii"),
      operation = "read"
    ),
    class = "fmridataset_error_backend_io"
  )

  # Test with complex value in config error
  expect_error(
    fmridataset:::stop_fmridataset(
      fmridataset:::fmridataset_error_config,
      message = "Invalid dataset structure",
      parameter = "run_lengths",
      value = list(a = 1, b = "invalid")
    ),
    class = "fmridataset_error_config"
  )
})

test_that("error messages are preserved correctly", {
  test_message <- "This is a detailed error message with context"

  err1 <- fmridataset:::fmridataset_error_backend_io(test_message, file = "test.h5")
  expect_equal(err1$message, test_message)

  err2 <- fmridataset:::fmridataset_error_config(test_message, parameter = "mask")
  expect_equal(err2$message, test_message)

  # Test that stop_fmridataset preserves the message
  expect_error(
    fmridataset:::stop_fmridataset(
      fmridataset:::fmridataset_error_config,
      test_message,
      parameter = "test"
    ),
    test_message,
    fixed = TRUE
  )
})

test_that("error constructors handle edge case values", {
  # Test with NULL values
  err1 <- fmridataset:::fmridataset_error_config(
    "null value error",
    parameter = "important_param",
    value = NULL
  )
  expect_null(err1$value)
  expect_equal(err1$parameter, "important_param")

  # Test with empty strings
  err2 <- fmridataset:::fmridataset_error_backend_io(
    "empty file error",
    file = "",
    operation = ""
  )
  expect_equal(err2$file, "")
  expect_equal(err2$operation, "")

  # Test with complex objects
  complex_value <- list(
    matrix = matrix(1:4, 2, 2),
    list = list(a = 1, b = 2),
    function_ref = function(x) x + 1
  )

  err3 <- fmridataset:::fmridataset_error_config(
    "complex object error",
    parameter = "complex_param",
    value = complex_value
  )
  expect_identical(err3$value, complex_value)
})

test_that("error system integrates with base R error handling", {
  # Test that our errors can be caught by standard error handling
  result <- tryCatch(
    {
      fmridataset:::stop_fmridataset(
        fmridataset:::fmridataset_error_backend_io,
        "test error",
        file = "test.nii"
      )
    },
    error = function(e) {
      e
    }
  )

  expect_s3_class(result, "fmridataset_error_backend_io")
  expect_equal(result$message, "test error")

  # Test that specific error types can be caught
  backend_caught <- FALSE
  config_caught <- FALSE

  tryCatch(
    {
      fmridataset:::stop_fmridataset(
        fmridataset:::fmridataset_error_backend_io,
        "backend error"
      )
    },
    fmridataset_error_backend_io = function(e) {
      backend_caught <<- TRUE
    },
    fmridataset_error_config = function(e) {
      config_caught <<- TRUE
    }
  )

  expect_true(backend_caught)
  expect_false(config_caught)
})
