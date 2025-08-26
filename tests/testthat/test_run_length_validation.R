library(testthat)

create_study <- function() {
  b1 <- matrix_backend(matrix(1:10, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(11:20, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  d1 <- fmri_dataset(b1, TR = 1, run_length = 5)
  d2 <- fmri_dataset(b2, TR = 1, run_length = 5)
  fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))
}

test_that("run length mismatch triggers error in metadata", {
  study <- create_study()
  study$sampling_frame$blocklens[1] <- 6
  expect_error(
    fmri_series(study, selector = 1, timepoints = 1:10),
    "run lengths inconsistent with backend dimensions"
  )
})
