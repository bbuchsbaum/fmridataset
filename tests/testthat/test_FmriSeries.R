library(DelayedArray)

# Basic instantiation

test_that("fmri_series can be created and displayed", {
  mat <- DelayedArray(matrix(1:6, nrow = 3))
  vox_info <- data.frame(id = 1:ncol(mat))
  tmp_info <- data.frame(id = 1:nrow(mat))
  
  fs <- new_fmri_series(
    data = mat,
    voxel_info = vox_info,
    temporal_info = tmp_info,
    selection_info = list(selector = NULL),
    dataset_info = list(backend_type = "matrix_backend")
  )
  
  expect_s3_class(fs, "fmri_series")
  
  out <- capture.output(print(fs))
  expect_true(any(grepl("Orientation: time", out)))
})

test_that("fmri_series as.matrix works", {
  mat <- matrix(1:6, nrow = 3)
  delayed_mat <- DelayedArray(mat)
  vox_info <- data.frame(id = 1:ncol(delayed_mat))
  tmp_info <- data.frame(id = 1:nrow(delayed_mat))
  
  fs <- new_fmri_series(
    data = delayed_mat,
    voxel_info = vox_info,
    temporal_info = tmp_info,
    selection_info = list(),
    dataset_info = list()
  )
  
  result <- as.matrix(fs)
  expect_equal(result, mat)
  expect_true(is.matrix(result))
})

test_that("fmri_series as_tibble works", {
  skip_if_not_installed("tibble")
  
  mat <- matrix(1:6, nrow = 3)
  delayed_mat <- DelayedArray(mat)
  vox_info <- data.frame(voxel_id = 1:2, region = c("A", "B"))
  tmp_info <- data.frame(time = 1:3, condition = c("rest", "task", "rest"))
  
  fs <- new_fmri_series(
    data = delayed_mat,
    voxel_info = vox_info,
    temporal_info = tmp_info,
    selection_info = list(),
    dataset_info = list()
  )
  
  tbl <- as_tibble(fs)
  expect_s3_class(tbl, "tbl_df")
  expect_equal(nrow(tbl), 6)  # 3 timepoints x 2 voxels
  expect_true("signal" %in% names(tbl))
  expect_true("time" %in% names(tbl))
  expect_true("voxel_id" %in% names(tbl))
})

test_that("is.fmri_series works", {
  mat <- DelayedArray(matrix(1:6, nrow = 3))
  vox_info <- data.frame(id = 1:ncol(mat))
  tmp_info <- data.frame(id = 1:nrow(mat))
  
  fs <- new_fmri_series(
    data = mat,
    voxel_info = vox_info,
    temporal_info = tmp_info,
    selection_info = list(),
    dataset_info = list()
  )
  
  expect_true(is.fmri_series(fs))
  expect_false(is.fmri_series(mat))
  expect_false(is.fmri_series(list()))
})