# Tests for R/mask_standards.R

test_that("mask_to_logical converts numeric vector", {
  mask <- c(1, 0, 1, 0, 1)
  result <- mask_to_logical(mask)
  expect_type(result, "logical")
  expect_equal(result, c(TRUE, FALSE, TRUE, FALSE, TRUE))
})

test_that("mask_to_logical converts logical vector (identity)", {
  mask <- c(TRUE, FALSE, TRUE)
  result <- mask_to_logical(mask)
  expect_type(result, "logical")
  expect_equal(result, mask)
})

test_that("mask_to_logical converts array", {
  mask <- array(c(1, 0, 0, 1, 1, 0, 1, 1), dim = c(2, 2, 2))
  result <- mask_to_logical(mask)
  expect_type(result, "logical")
  expect_length(result, 8)
  expect_equal(result, as.logical(as.vector(mask)))
})

test_that("mask_to_volume converts logical vector to 3D array", {
  mask_vec <- c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE)
  dims <- c(2, 2, 2)
  result <- mask_to_volume(mask_vec, dims)
  expect_true(is.array(result))
  expect_equal(dim(result), dims)
  expect_type(result, "logical")
  expect_equal(as.vector(result), mask_vec)
})

test_that("mask_to_volume errors on dimension mismatch", {
  mask_vec <- c(TRUE, FALSE, TRUE)
  dims <- c(2, 2, 2)
  expect_error(mask_to_volume(mask_vec, dims), "Mask length.*doesn't match")
})

test_that("mask_to_volume handles numeric input", {
  mask_vec <- c(1, 0, 1, 0, 1, 0, 1, 0)
  dims <- c(2, 2, 2)
  result <- mask_to_volume(mask_vec, dims)
  expect_true(is.array(result))
  expect_type(result, "logical")
  expect_equal(as.vector(result), as.logical(mask_vec))
})
