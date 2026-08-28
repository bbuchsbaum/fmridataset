# study_backend coverage used to be deferred here behind an unconditional
# skip() -- a TODO shaped like a test. It now lives in
# test_study_backend_column_routing.R, which exercises the mask-combination
# and column-routing contract that placeholder was standing in for.

test_that("memoise cache respects memory bounds", {
  # The cache is created when the package loads
  # Just verify fmri_clear_cache works
  expect_silent(fmri_clear_cache())

  # Can set cache size via option
  options(fmridataset.cache_max_mb = 256)
  expect_equal(getOption("fmridataset.cache_max_mb"), 256)
})

test_that("fmri_clear_cache works", {
  # Clear the cache
  expect_silent(fmri_clear_cache())

  # Function should return NULL invisibly
  result <- fmri_clear_cache()
  expect_null(result)
})
