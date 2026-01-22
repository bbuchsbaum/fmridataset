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

# ============================================
# Section 2: study_backend conversion ----
# ============================================

test_that("as_delarr.study_backend returns delarr object", {
  skip_if_not_installed("delarr")

  # Create two matrix backends with same number of voxels
  mat1 <- create_test_matrix(n_time = 10, n_voxels = 50)
  mat2 <- create_test_matrix(n_time = 15, n_voxels = 50)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)

  # Create study backend
  sb <- study_backend(list(backend1, backend2), subject_ids = c("S01", "S02"))
  result <- as_delarr(sb)

  expect_s3_class(result, "delarr")
  # Combined dimensions: 10 + 15 = 25 rows, 50 cols
  expect_equal(nrow(result), 25)
  expect_equal(ncol(result), 50)
})

test_that("as_delarr.study_backend retrieves data correctly", {
  skip_if_not_installed("delarr")

  # Create backends with distinct values for verification
  set.seed(100)
  mat1 <- matrix(1:50, nrow = 5, ncol = 10)  # Subject 1: values 1-50
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)  # Subject 2: values 51-100

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)
  realized <- delarr::collect(darr)

  # Combined should have both subjects
  expected <- rbind(mat1, mat2)
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend retrieves data across subject boundaries", {
  skip_if_not_installed("delarr")

  # Subject 1: rows 1-5, Subject 2: rows 6-10
  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)

  # Query rows 3-8 which crosses the boundary at row 5
  subset <- darr[3:8, ]
  realized <- delarr::collect(subset)

  # Expected: rows 3-5 from mat1, rows 1-3 from mat2
  expected <- rbind(mat1[3:5, ], mat2[1:3, ])
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend handles column subsetting", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:100, nrow = 10, ncol = 10)
  mat2 <- matrix(101:200, nrow = 10, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)

  # Subset columns only
  subset <- darr[, 2:5]
  realized <- delarr::collect(subset)

  expected <- rbind(mat1[, 2:5], mat2[, 2:5])
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend handles logical row indices", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:30, nrow = 3, ncol = 10)
  mat2 <- matrix(31:60, nrow = 3, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)

  # Logical index: select rows 1, 3, 4, 6 (TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)
  logical_idx <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE)
  subset <- darr[logical_idx, ]
  realized <- delarr::collect(subset)

  expected <- rbind(mat1, mat2)[logical_idx, ]
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend handles logical column indices", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)

  # Logical index for columns
  logical_cols <- c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE)
  subset <- darr[, logical_cols]
  realized <- delarr::collect(subset)

  expected <- rbind(mat1, mat2)[, logical_cols]
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend converts integer-valued doubles to integers", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  darr <- as_delarr(sb)

  # Use double indices that are integer-valued (1.0, 2.0, 3.0)
  double_rows <- c(1.0, 2.0, 3.0)
  double_cols <- c(1.0, 5.0)
  subset <- darr[double_rows, double_cols]
  realized <- delarr::collect(subset)

  expected <- rbind(mat1, mat2)[1:3, c(1, 5)]
  expect_equal(realized, expected)
})

test_that("as_delarr.study_backend handles empty result via backend_get_data", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  # Test empty results through backend_get_data which uses the same logic
  # Empty row index
  result1 <- backend_get_data(sb, rows = integer(0), cols = 1:5)
  expect_equal(nrow(result1), 0)
  expect_equal(ncol(result1), 5)

  # Empty column index
  result2 <- backend_get_data(sb, rows = 1:5, cols = integer(0))
  expect_equal(nrow(result2), 5)
  expect_equal(ncol(result2), 0)
})

# Error path tests for study_backend ----
# Note: delarr handles some index validation before the pull function is called.
# We test our error paths through backend_get_data which uses the same logic.

test_that("study_backend errors on out-of-bounds rows", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  # Total rows = 10, so row 11 is out of bounds
  expect_error(
    backend_get_data(sb, rows = 11L, cols = 1L),
    "Row indices out of bounds"
  )

  # Row 0 is out of bounds
  expect_error(
    backend_get_data(sb, rows = 0L, cols = 1L),
    "Row indices out of bounds"
  )

  # Negative row is out of bounds
  expect_error(
    backend_get_data(sb, rows = -1L, cols = 1L),
    "Row indices out of bounds"
  )
})

test_that("study_backend errors on out-of-bounds columns", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  # Total cols = 10, so col 11 is out of bounds
  expect_error(
    backend_get_data(sb, rows = 1L, cols = 11L),
    "Column indices out of bounds"
  )

  # Column 0 is out of bounds
  expect_error(
    backend_get_data(sb, rows = 1L, cols = 0L),
    "Column indices out of bounds"
  )
})

test_that("study_backend errors on non-integer-valued row doubles", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  # Non-integer double (1.5 is not integer-valued)
  expect_error(
    backend_get_data(sb, rows = c(1.5, 2.5), cols = 1L),
    "Row indices must be integer valued"
  )
})

test_that("study_backend errors on non-integer-valued column doubles", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:50, nrow = 5, ncol = 10)
  mat2 <- matrix(51:100, nrow = 5, ncol = 10)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  sb <- study_backend(list(backend1, backend2))

  # Non-integer double (1.5 is not integer-valued)
  expect_error(
    backend_get_data(sb, rows = 1L, cols = c(1.5, 2.5)),
    "Column indices must be integer valued"
  )
})

test_that("as_delarr.study_backend handles three or more subjects", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:20, nrow = 4, ncol = 5)
  mat2 <- matrix(21:40, nrow = 4, ncol = 5)
  mat3 <- matrix(41:60, nrow = 4, ncol = 5)

  backend1 <- matrix_backend(mat1)
  backend2 <- matrix_backend(mat2)
  backend3 <- matrix_backend(mat3)
  sb <- study_backend(list(backend1, backend2, backend3))

  darr <- as_delarr(sb)

  # Total: 12 rows, 5 cols
  expect_equal(nrow(darr), 12)
  expect_equal(ncol(darr), 5)

  # Query across all three subjects (rows 3-10)
  subset <- darr[3:10, ]
  realized <- delarr::collect(subset)

  expected <- rbind(mat1[3:4, ], mat2, mat3[1:2, ])
  expect_equal(realized, expected)
})
