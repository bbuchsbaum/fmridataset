library(testthat)
library(fmridataset)

# Test slicewise_chunks

test_that("slicewise_chunks generates one mask per slice", {
  skip_if_not_installed("neuroim2")

  dims <- c(2, 2, 2, 1)
  scan <- neuroim2::NeuroVec(array(1:prod(dims), dims), space = neuroim2::NeuroSpace(dims))
  mask <- neuroim2::LogicalNeuroVol(array(TRUE, dims[1:3]), neuroim2::NeuroSpace(dims[1:3]))

  dset <- fmri_mem_dataset(list(scan), mask, TR = 1)
  slices <- fmridataset:::slicewise_chunks(dset)

  expect_length(slices, dims[3])
  expect_s3_class(slices[[1]], "NeuroVol")
  expect_true(all(slices[[1]][,,1] == 1))
  expect_true(all(slices[[1]][,,2] == 0))
})

# Test deprecated series() alias

test_that("series alias forwards with deprecation warning", {
  mat <- matrix(1:40, nrow = 5, ncol = 8)
  backend <- matrix_backend(mat, mask = rep(TRUE, 8), spatial_dims = c(2,2,2))
  dset <- fmri_dataset(backend, TR = 1, run_length = 5)

  expect_warning(fs_alias <- series(dset, selector = 1:2, timepoints = 1:3),
                 class = "lifecycle_warning_deprecated")
  fs_direct <- fmri_series(dset, selector = 1:2, timepoints = 1:3)
  expect_equal(as.matrix(fs_alias), as.matrix(fs_direct))
})

# Test exec_strategy for runwise and voxelwise

test_that("exec_strategy runwise and voxelwise", {
  mat <- matrix(1:100, nrow = 10, ncol = 10)
  backend <- matrix_backend(mat, mask = rep(TRUE,10), spatial_dims = c(10,1,1))
  dset <- fmri_dataset(backend, TR = 1, run_length = c(5,5))

  run_iter <- fmridataset:::exec_strategy("runwise")(dset)
  expect_equal(run_iter$nchunks, 2)
  ch1 <- run_iter$nextElem()
  ch2 <- run_iter$nextElem()
  expect_equal(ch1$row_ind, 1:5)
  expect_equal(ch2$row_ind, 6:10)

  vox_iter <- fmridataset:::exec_strategy("voxelwise")(dset)
  expect_equal(vox_iter$nchunks, 10)
  first_vox <- vox_iter$nextElem()
  expect_equal(ncol(first_vox$data), 1)
})
