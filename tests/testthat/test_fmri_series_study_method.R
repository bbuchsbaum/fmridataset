library(testthat)

create_study_dataset <- function() {
  b1 <- matrix_backend(matrix(1:20, nrow = 5, ncol = 4), spatial_dims = c(2, 2, 1))
  d1 <- fmri_dataset(b1, TR = 1, run_length = 5)
  b2 <- matrix_backend(matrix(21:40, nrow = 5, ncol = 4), spatial_dims = c(2, 2, 1))
  d2 <- fmri_dataset(b2, TR = 1, run_length = 5)
  fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))
}


test_that("fmri_series.fmri_study_dataset returns valid FmriSeries", {
  study <- create_study_dataset()
  fs <- fmri_series(study, selector = 2:3, timepoints = 4:7)
  expect_s3_class(fs, "fmri_series")
  expect_equal(dim(fs), c(4, 2))

  expected <- rbind(
    study$backend$backends[[1]]$data_matrix[4:5, 2:3],
    study$backend$backends[[2]]$data_matrix[1:2, 2:3]
  )
  expect_equal(as.matrix(fs), expected)

  md <- fs$temporal_info
  run_ids <- fmrihrf::blockids(study$sampling_frame)
  expect_equal(md$timepoint, 4:7)
  expect_equal(md$run_id, run_ids[4:7])
  expect_equal(as.character(md$subject_id), c("s1", "s1", "s2", "s2"))
})
