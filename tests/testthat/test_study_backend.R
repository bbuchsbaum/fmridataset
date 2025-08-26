context("study_backend")

test_that("constructor validates dimensions", {
  b1 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), spatial_dims = c(1, 2, 1))
  expect_error(study_backend(list(b1, b2)), "spatial dimensions")
})

test_that("constructor validates mask", {
  b1 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), mask = c(TRUE, TRUE), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), mask = c(TRUE, FALSE), spatial_dims = c(2, 1, 1))
  expect_error(study_backend(list(b1, b2)), "masks differ")
})

test_that("study_backend basic operations", {
  b1 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(21:40, nrow = 10, ncol = 2), spatial_dims = c(2, 1, 1))
  sb <- study_backend(list(b1, b2))
  dims <- backend_get_dims(sb)
  expect_equal(dims$time, 20)
  expect_equal(dims$spatial, c(2, 1, 1))
  mask <- backend_get_mask(sb)
  expect_equal(mask, rep(TRUE, 2))

  da <- backend_get_data(sb)
  expect_s4_class(da, "DelayedArray")
  expect_s4_class(da, "DelayedArray")
  expect_equal(dim(da), c(20, 2))

  sub <- backend_get_data(sb, rows = 1:5, cols = 1)
  expect_equal(dim(sub), c(5, 1))
})

test_that("empty backend allowed", {
  empty <- matrix_backend(matrix(numeric(), nrow = 0, ncol = 2), spatial_dims = c(2, 1, 1))
  sb <- study_backend(list(empty))
  expect_equal(backend_get_dims(sb)$time, 0)
})

test_that("strict='intersect' validates mask overlap", {
  b1 <- matrix_backend(matrix(1:10, nrow = 5, ncol = 2),
    mask = rep(TRUE, 2), spatial_dims = c(2, 1, 1)
  )
  b2 <- matrix_backend(matrix(11:20, nrow = 5, ncol = 2),
    mask = c(TRUE, FALSE), spatial_dims = c(2, 1, 1)
  )

  expect_error(
    study_backend(list(b1, b2), strict = "intersect"),
    "mask overlap <95%"
  )

  big_mask1 <- rep(TRUE, 100)
  big_mask2 <- c(rep(TRUE, 96), rep(FALSE, 4))
  bb1 <- matrix_backend(matrix(1:200, nrow = 2, ncol = 100),
    mask = big_mask1, spatial_dims = c(10, 10, 1)
  )
  bb2 <- matrix_backend(matrix(201:400, nrow = 2, ncol = 100),
    mask = big_mask2, spatial_dims = c(10, 10, 1)
  )

  sb <- study_backend(list(bb1, bb2), strict = "intersect")
  expect_equal(backend_get_mask(sb), big_mask1 & big_mask2)
})
