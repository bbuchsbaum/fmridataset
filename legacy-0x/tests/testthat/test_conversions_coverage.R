# Tests for R/conversions.R - coverage improvement

test_that("as.matrix_dataset.matrix_dataset returns identity", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  result <- as.matrix_dataset(ds)
  expect_identical(result, ds)
})

# Test as_delayed_array with matrix_backend
test_that("as_delayed_array.matrix_backend works", {
  skip_if_not_installed("DelayedArray")

  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)

  da <- as_delayed_array(backend)
  expect_true(inherits(da, "DelayedMatrix") || inherits(da, "DelayedArray"))
  expect_equal(dim(da), c(20, 10))
})

test_that("as_delayed_array.default errors on unknown class", {
  expect_error(as_delayed_array(42), "No as_delayed_array method")
})
