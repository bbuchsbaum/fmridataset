library(testthat)

create_alias_dataset <- function() {
  matrix_dataset(matrix(1:20, nrow = 5, ncol = 4), TR = 1, run_length = 5)
}

test_that("series() warns once and returns FmriSeries", {
  dset <- create_alias_dataset()
  expect_warning(res1 <- series(dset, selector = 1:2, timepoints = 1:2),
                 "deprecated")
  expect_s4_class(res1, "FmriSeries")
  expect_warning(series(dset, selector = 1:2, timepoints = 1:2), NA)
})
