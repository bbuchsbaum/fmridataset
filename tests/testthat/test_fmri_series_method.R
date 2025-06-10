library(testthat)

create_test_dataset <- function() {
  mat <- matrix(1:40, nrow = 5, ncol = 8)
  backend <- matrix_backend(mat, mask = rep(TRUE, 8), spatial_dims = c(2,2,2))
  fmri_dataset(backend, TR = 1, run_length = 5)
}


test_that("fmri_series.fmri_dataset returns FmriSeries", {
  dset <- create_test_dataset()
  fs <- fmri_series(dset, selector = 3:5, timepoints = 2:4)
  expect_s4_class(fs, "FmriSeries")
  expect_equal(dim(fs), c(3, 3))
  expected <- dset$backend$data_matrix[2:4, 3:5]
  expect_equal(as.matrix(fs), expected)
  expect_equal(fs@voxel_info$voxel, 3:5)
  expect_equal(fs@temporal_info$timepoint, 2:4)
})

test_that("fmri_series can return DelayedMatrix", {
  dset <- create_test_dataset()
  dm <- fmri_series(dset, selector = 1:2, timepoints = 1:2, output = "DelayedMatrix")
  expect_s4_class(dm, "DelayedMatrix")
  expected <- dset$backend$data_matrix[1:2, 1:2]
  expect_equal(as.matrix(dm), expected)
})
