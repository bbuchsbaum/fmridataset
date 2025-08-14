library(testthat)

create_alias_dataset <- function() {
  backend <- matrix_backend(matrix(1:20, nrow = 5, ncol = 4))
  fmri_dataset(backend, TR = 1, run_length = 5)
}

test_that("series() returns FmriSeries and shows deprecation", {
  dset <- create_alias_dataset()
  
  # Test that the function works and returns correct type
  res <- suppressWarnings(series(dset, selector = 1:2, timepoints = 1:2))
  expect_s3_class(res, "fmri_series")
  
  # Test that it produces the same result as fmri_series
  direct_res <- fmri_series(dset, selector = 1:2, timepoints = 1:2)
  expect_equal(as.matrix(res), as.matrix(direct_res))
  
  # Test that calling series() does generate a lifecycle warning
  # (We don't test the "once only" behavior as it's hard to test reliably in testthat)
  expect_warning(series(dset, selector = 1:2, timepoints = 1:2), "deprecated|lifecycle")
})
