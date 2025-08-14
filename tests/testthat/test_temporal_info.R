library(testthat)
library(fmridataset)

# Tests for build_temporal_info_lazy helpers

create_temp_dataset <- function() {
  Y <- matrix(1:10, nrow = 5, ncol = 2)
  matrix_dataset(Y, TR = 1, run_length = c(3, 2))
}

create_study_dataset <- function() {
  d1 <- matrix_dataset(matrix(1:10, nrow = 5, ncol = 2), TR = 1, run_length = 5)
  d2 <- matrix_dataset(matrix(11:20, nrow = 5, ncol = 2), TR = 1, run_length = 5)
  fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))
}


test_that("build_temporal_info_lazy.fmri_dataset returns correct metadata", {
  dset <- create_temp_dataset()
  info <- fmridataset:::build_temporal_info_lazy(dset, 1:5)
  expect_s3_class(info, "data.frame")
  expect_equal(info$run_id, c(1, 1, 1, 2, 2))
  expect_equal(info$timepoint, 1:5)
})


test_that("build_temporal_info_lazy.fmri_study_dataset includes subject mapping", {
  study <- create_study_dataset()
  info <- fmridataset:::build_temporal_info_lazy(study, 4:7)
  expect_s3_class(info, "data.frame")
  expect_equal(as.character(info$subject_id), c("s1", "s1", "s2", "s2"))
  expect_equal(info$run_id, c(1, 1, 2, 2))
  expect_equal(info$timepoint, 4:7)
})

