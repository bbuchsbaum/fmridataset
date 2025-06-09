library(fmridataset)

# Tests for sampling_frame utilities

test_that("sampling_frame utilities work", {
  sf <- fmrihrf::sampling_frame(blocklens = c(10, 20, 30), TR = 2)

  expect_true(is.sampling_frame(sf))
  expect_equal(get_TR(sf), 2)
  expect_equal(get_run_lengths(sf), c(10, 20, 30))
  expect_equal(n_runs(sf), 3)
  expect_equal(n_timepoints(sf), 60)
  expect_equal(blocklens(sf), c(10, 20, 30))
  expect_equal(blockids(sf), c(rep(1, 10), rep(2, 20), rep(3, 30)))
  expect_equal(samples(sf), 1:60)
  # Note: fmrihrf global_onsets has different signature - needs onsets and blockids
  # expect_equal(global_onsets(sf), c(1, 11, 31))
  expect_equal(get_total_duration(sf), 120)
  expect_equal(get_run_duration(sf), c(20, 40, 60))
  expect_output(print(sf), "Sampling Frame")
})
