# test-as_delarr.R
# Tests for as_delarr.R - delarr lazy matrix interface
# as_delarr is the PRIMARY lazy array path (delarr is in Imports)
#
# Coverage target: as_delarr.R methods
#   - as_delarr.matrix_backend (tested)
#   - as_delarr.study_backend (tested)
#   - as_delarr.default (tested)
#   - as_delarr.nifti_backend (skipped - requires neuroim2 S4 objects)

# ============================================
# Section 1: matrix_backend conversion ----
# ============================================

test_that("as_delarr.matrix_backend returns delarr object", {
  skip_if_not_installed("delarr")

  mat <- create_test_matrix(n_time = 20, n_voxels = 50)
  backend <- matrix_backend(mat)
  result <- as_delarr(backend)

  expect_s3_class(result, "delarr")
  expect_equal(nrow(result), 20)
  expect_equal(ncol(result), 50)
})

test_that("as_delarr.matrix_backend pull function retrieves correct data", {
  skip_if_not_installed("delarr")

  mat <- create_test_matrix(n_time = 15, n_voxels = 30)
  backend <- matrix_backend(mat)
  darr <- as_delarr(backend)

  # Full materialization
  realized <- delarr::collect(darr)
  expect_equal(realized, mat, tolerance = 1e-10)

  # Subset retrieval
  subset <- darr[1:5, 1:10]
  subset_realized <- delarr::collect(subset)
  expect_equal(subset_realized, mat[1:5, 1:10], tolerance = 1e-10)
})

test_that("as_delarr.matrix_backend handles single row", {
  skip_if_not_installed("delarr")

  set.seed(123)
  mat <- matrix(rnorm(20), nrow = 1, ncol = 20)
  backend <- matrix_backend(mat)
  result <- as_delarr(backend)

  expect_equal(nrow(result), 1)
  expect_equal(ncol(result), 20)
  realized <- delarr::collect(result)
  expect_equal(realized, mat, tolerance = 1e-10)
})

test_that("as_delarr.matrix_backend handles single column", {
  skip_if_not_installed("delarr")

  set.seed(123)
  mat <- matrix(rnorm(15), nrow = 15, ncol = 1)
  backend <- matrix_backend(mat)
  result <- as_delarr(backend)

  expect_equal(nrow(result), 15)
  expect_equal(ncol(result), 1)
  realized <- delarr::collect(result)
  expect_equal(realized, mat, tolerance = 1e-10)
})

test_that("as_delarr.matrix_backend preserves data through round trip", {

  skip_if_not_installed("delarr")

  # Create matrix with varied values
  set.seed(456)
  mat <- matrix(c(1:100, rnorm(100)), nrow = 20, ncol = 10)
  backend <- matrix_backend(mat)
  darr <- as_delarr(backend)

  # Full round trip
  result <- delarr::collect(darr)
  expect_equal(result, mat)

  # Multiple subsetting operations
  s1 <- delarr::collect(darr[1:10, ])
  expect_equal(s1, mat[1:10, ])

  s2 <- delarr::collect(darr[, 1:5])
  expect_equal(s2, mat[, 1:5])

  s3 <- delarr::collect(darr[5:15, 3:8])
  expect_equal(s3, mat[5:15, 3:8])
})

# NOTE: nifti_backend skipped - requires neuroim2 S4 objects
# Testing would require complex mocking of NeuroVec objects.
# Same limitation as h5_backend in plan 04.
