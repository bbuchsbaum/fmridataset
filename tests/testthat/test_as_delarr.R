suppressPackageStartupMessages({
  library(fmridataset)
})

test_that("test_as_delarr package exports are available", {
  expect_true(exists("matrix_backend", envir = asNamespace("fmridataset"), inherits = FALSE))
  expect_true(exists("as_delarr", envir = asNamespace("fmridataset"), inherits = FALSE))
})

test_that("as_delarr wraps matrix_backend", {

  mat <- matrix(1:12, nrow = 3, ncol = 4)
  backend <- matrix_backend(mat, mask = rep(TRUE, 4), spatial_dims = c(2, 2, 1))

  lazy <- as_delarr(backend)
  expect_true(inherits(lazy, "delarr"))
  expect_equal(dim(lazy), c(3, 4))
  expect_equal(as.matrix(lazy), backend_get_data(backend))
})

test_that("as_delarr stitches study backend rows", {
  b1 <- matrix_backend(matrix(1:10, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(11:20, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  study <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  lazy <- as_delarr(study)
  expect_true(inherits(lazy, "delarr"))
  expect_equal(dim(lazy), c(10, 2))

  subset_rows <- c(4:7)
  subset_cols <- c(1, 2)
  block <- lazy[subset_rows, subset_cols]
  expect_true(inherits(block, "delarr"))
  expect_equal(
    as.matrix(block),
    rbind(
      backend_get_data(b1)[subset_rows[subset_rows <= 5], , drop = FALSE],
      backend_get_data(b2)[subset_rows[subset_rows > 5] - 5, , drop = FALSE]
    )
  )
})

test_that("backend_get_data.study_backend returns delarr lazily", {
  b1 <- matrix_backend(matrix(1:6, nrow = 3, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(7:12, nrow = 3, ncol = 2), spatial_dims = c(2, 1, 1))
  sb <- study_backend(list(b1, b2), subject_ids = c("a", "b"))

  lazy <- backend_get_data(sb)
  expect_true(inherits(lazy, "delarr"))

  realised <- backend_get_data(sb, rows = 2:5, cols = 1)
  expect_true(is.matrix(realised))
  expect_equal(nrow(realised), 4)
  expect_equal(ncol(realised), 1)
  expected <- rbind(backend_get_data(b1)[2:3, 1, drop = FALSE], backend_get_data(b2)[1:2, 1, drop = FALSE])
  expect_equal(realised, expected)
})
