library(testthat)

create_matrix_dataset <- function() {
  mat <- matrix(1:40, nrow = 5, ncol = 8)
  backend <- matrix_backend(mat, mask = rep(TRUE, 8), spatial_dims = c(2,2,2))
  fmri_dataset(backend, TR = 1, run_length = 5)
}

create_nifti_dataset <- function() {
  skip_if_not_installed("neuroim2")
  dims <- c(2, 2, 1, 5)
  data_array <- array(seq_len(prod(dims)), dims)
  mock_vec <- structure(
    data_array,
    class = c("DenseNeuroVec", "NeuroVec", "array"),
    space = structure(list(dim = dims[1:3], origin = c(0,0,0), spacing = c(1,1,1)),
                     class = "NeuroSpace")
  )
  mock_mask <- structure(array(TRUE, dims[1:3]),
                         class = c("LogicalNeuroVol", "NeuroVol", "array"),
                         dim = dims[1:3])
  backend <- nifti_backend(source = list(mock_vec),
                          mask_source = mock_mask,
                          preload = TRUE)
  fmri_dataset(backend, TR = 1, run_length = 5)
}


test_that("fmri_series works with multiple backends", {
  dset_mat <- create_matrix_dataset()
  dset_nifti <- create_nifti_dataset()

  fs_mat <- fmri_series(dset_mat, selector = 1:2, timepoints = 1:3)
  expected_mat <- dset_mat$backend$data_matrix[1:3, 1:2]
  expect_equal(as.matrix(fs_mat), expected_mat)

  fs_nifti <- fmri_series(dset_nifti, selector = 1:2, timepoints = 1:3)
  expected_nifti <- backend_get_data(dset_nifti$backend, rows = 1:3, cols = 1:2)
  expect_equal(as.matrix(fs_nifti), expected_nifti)
})


test_that("multi-subject ordering is preserved across backends", {
  dset1 <- create_matrix_dataset()
  dset2 <- create_nifti_dataset()
  study <- fmri_study_dataset(list(dset1, dset2), subject_ids = c("s1", "s2"))

  fs <- fmri_series(study, selector = 1:2, timepoints = 4:7)

  expected <- rbind(
    backend_get_data(dset1$backend, rows = 4:5, cols = 1:2),
    backend_get_data(dset2$backend, rows = 1:2, cols = 1:2)
  )
  expect_equal(as.matrix(fs), expected)
  expect_equal(as.character(fs@temporal_info$subject_id), c("s1", "s1", "s2", "s2"))
})


test_that("edge cases for selection are handled", {
  dset <- create_matrix_dataset()

  empty_sel <- fmri_series(dset, selector = integer(0), timepoints = 1:2)
  expect_equal(dim(empty_sel), c(2, 0))
  expect_equal(ncol(as.matrix(empty_sel)), 0)

  single_tp <- fmri_series(dset, selector = 1, timepoints = 3)
  expect_equal(dim(single_tp), c(1,1))
  expect_equal(as.matrix(single_tp), matrix(dset$backend$data_matrix[3,1], nrow=1))
})


test_that("tidyverse workflow on fmri_series output", {
  dset <- create_matrix_dataset()
  fs <- fmri_series(dset, selector = 1:2, timepoints = 1:4)
  tb <- as_tibble(fs)
  res <- dplyr::filter(tb, voxel == 1) %>% dplyr::summarise(mn = mean(signal))
  expect_equal(res$mn, mean(dset$backend$data_matrix[1:4, 1]))
})

test_that("subject mapping works with uneven run lengths", {
  d1 <- fmri_dataset(
    matrix_backend(matrix(1:6, nrow = 3, ncol = 2), mask = rep(TRUE,2), spatial_dims = c(2,1,1)),
    TR = 1, run_length = 3
  )
  d2 <- fmri_dataset(
    matrix_backend(matrix(7:10, nrow = 2, ncol = 2), mask = rep(TRUE,2), spatial_dims = c(2,1,1)),
    TR = 1, run_length = 2
  )
  study <- fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))

  fs <- fmri_series(study, selector = 1:2, timepoints = 1:5)
  expected <- rbind(
    d1$backend$data_matrix[1:3, 1:2],
    d2$backend$data_matrix[1:2, 1:2]
  )
  expect_equal(as.matrix(fs), expected)
  expect_equal(as.character(fs@temporal_info$subject_id), c("s1", "s1", "s1", "s2", "s2"))
})
