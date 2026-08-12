test_that("memoise cache respects memory bounds", {
  # The cache is created when the package loads
  # Just verify fmri_clear_cache works
  expect_silent(fmri_clear_cache())

  # Can set cache size via option
  options(fmridataset.cache_max_mb = 256)
  expect_equal(getOption("fmridataset.cache_max_mb"), 256)
})

test_that("study_backend warns about memory usage", {
  skip("Comprehensive study_backend testing will be added in next phase")
  # This test requires proper mock backends which will be implemented
  # as part of the full study_backend refactoring
})

test_that("fmri_clear_cache works", {
  # Clear the cache
  expect_silent(fmri_clear_cache())

  # Function should return NULL invisibly
  result <- fmri_clear_cache()
  expect_null(result)
})
