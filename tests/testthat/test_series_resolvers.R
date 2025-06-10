library(testthat)

context("series resolver helpers")

create_test_dataset <- function() {
  mat <- matrix(1:40, nrow = 5, ncol = 8)
  backend <- matrix_backend(mat, mask = rep(TRUE, 8), spatial_dims = c(2,2,2))
  fmri_dataset(backend, TR = 1, run_length = 5)
}

test_that("resolve_selector handles NULL and indices", {
  dset <- create_test_dataset()
  expect_equal(resolve_selector(dset, NULL), 1:8)
  expect_equal(resolve_selector(dset, 2:3), as.integer(2:3))
})

test_that("resolve_selector handles coordinates and masks", {
  dset <- create_test_dataset()
  coords <- matrix(c(1,1,1,
                     2,1,1), ncol = 3, byrow = TRUE)
  expect_equal(resolve_selector(dset, coords), as.integer(1:2))

  mask <- array(FALSE, c(2,2,2))
  mask[1,1,1] <- TRUE
  mask[2,1,1] <- TRUE
  expect_equal(resolve_selector(dset, mask), as.integer(1:2))
})

test_that("resolve_timepoints handles basic cases", {
  dset <- create_test_dataset()
  expect_equal(resolve_timepoints(dset, NULL), 1:5)
  expect_equal(resolve_timepoints(dset, 1:2), as.integer(1:2))
  logical_sel <- c(TRUE, FALSE, TRUE, FALSE, FALSE)
  expect_equal(resolve_timepoints(dset, logical_sel), c(1L,3L))
})

test_that("all_timepoints returns full range", {
  dset <- create_test_dataset()
  expect_equal(all_timepoints(dset), 1:5)
})

