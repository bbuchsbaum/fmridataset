library(testthat)

create_test_dataset <- function() {
  mat <- matrix(as.double(1:40), nrow = 5, ncol = 8)
  backend <- matrix_backend(mat, mask = rep(TRUE, 8), spatial_dims = c(2,2,2))
  fmri_dataset(backend, TR = 1, run_length = 5)
}


test_that("fmri_series.fmri_dataset returns fmri_series", {
  dset <- create_test_dataset()
  fs <- fmri_series(dset, selector = 3:5, timepoints = 2:4)
  expect_s3_class(fs, "fmri_series")
  expect_equal(dim(fs$data), c(3, 3))
  expected <- dset$backend$data_matrix[2:4, 3:5]
  expect_equal(as.matrix(fs), expected)
  expect_equal(fs$voxel_info$voxel, 3:5)
  expect_equal(fs$temporal_info$timepoint, 2:4)
})

test_that("fmri_series can return DelayedMatrix", {
  dset <- create_test_dataset()
  dm <- fmri_series(dset, selector = 1:2, timepoints = 1:2, output = "DelayedMatrix")
  expect_s4_class(dm, "DelayedMatrix")
  expected <- dset$backend$data_matrix[1:2, 1:2]
  expect_equal(as.matrix(dm), expected)
})


test_that("as.matrix.fmri_series materialises data", {
  dset <- create_test_dataset()
  fs <- fmri_series(dset, selector = 1:4, timepoints = 1:3)
  expect_type(as.matrix(fs), "double")
  expect_equal(as.matrix(fs), dset$backend$data_matrix[1:3, 1:4])
})

test_that("as_tibble.fmri_series supports dplyr summarise", {
  skip_if_not_installed("dplyr")
  dset <- create_test_dataset()
  fs <- fmri_series(dset, selector = 1:2, timepoints = 1:4)
  tb <- as_tibble(fs)
  res <- dplyr::group_by(tb, voxel) %>% dplyr::summarise(mean_signal = mean(signal))
  expected <- colMeans(dset$backend$data_matrix[1:4, 1:2])
  expect_equal(res$mean_signal, as.numeric(expected))
})
