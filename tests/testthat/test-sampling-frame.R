test_that("sampling_frame constructor works correctly", {
  # Single run
  sf <- sampling_frame(TR = 2.0, run_lengths = 100)
  
  expect_s3_class(sf, "sampling_frame")
  expect_equal(sf$TR, c(2.0))
  expect_equal(sf$blocklens, c(100))
  expect_equal(sf$n_runs, 1)
  expect_equal(sf$total_timepoints, 100)
  
  # Multiple runs
  sf_multi <- sampling_frame(TR = 2.5, run_lengths = c(180, 160, 200))
  
  expect_equal(sf_multi$TR, c(2.5, 2.5, 2.5))
  expect_equal(sf_multi$blocklens, c(180, 160, 200))
  expect_equal(sf_multi$n_runs, 3)
  expect_equal(sf_multi$total_timepoints, 540)
  
  # Variable TR
  sf_var <- sampling_frame(TR = c(2.0, 2.5, 3.0), run_lengths = c(100, 80, 60))
  
  expect_equal(sf_var$TR, c(2.0, 2.5, 3.0))
  expect_equal(sf_var$blocklens, c(100, 80, 60))
  expect_equal(sf_var$n_runs, 3)
  expect_equal(sf_var$total_timepoints, 240)
})

test_that("sampling_frame validation works", {
  # TR and run_lengths length mismatch (when TR is vector)
  expect_error(
    sampling_frame(TR = c(2.0, 2.5), run_lengths = c(100, 80, 60)),
    "Length of TR"
  )
  
  # Negative TR
  expect_error(
    sampling_frame(TR = -1.0, run_lengths = 100),
    "TR values must be positive"
  )
  
  # Zero run length
  expect_error(
    sampling_frame(TR = 2.0, run_lengths = c(100, 0, 80)),
    "run_lengths must be positive"
  )
  
  # Empty run_lengths
  expect_error(
    sampling_frame(TR = 2.0, run_lengths = numeric(0)),
    "run_lengths cannot be empty"
  )
})

test_that("sampling_frame accessor methods work", {
  sf <- sampling_frame(TR = 2.0, run_lengths = c(100, 80, 120))
  
  # Basic accessors
  expect_equal(n_timepoints(sf), 300)
  expect_equal(n_runs(sf), 3)
  expect_equal(get_TR(sf), c(2.0, 2.0, 2.0))
  expect_equal(get_run_lengths(sf), c(100, 80, 120))
  
  # Run-specific timepoints
  expect_equal(n_timepoints(sf, run_id = 1), 100)
  expect_equal(n_timepoints(sf, run_id = 2), 80)
  expect_equal(n_timepoints(sf, run_id = c(1, 3)), 220)
  
  # Duration calculations
  expect_equal(get_total_duration(sf), 600)  # 300 * 2.0
  expect_equal(get_run_duration(sf, run_id = 1), 200)  # 100 * 2.0
  expect_equal(get_run_duration(sf, run_id = c(2, 3)), c(160, 240))  # 80*2.0, 120*2.0
})

test_that("sampling_frame with variable TR works", {
  sf <- sampling_frame(TR = c(2.0, 2.5, 1.5), run_lengths = c(100, 80, 120))
  
  expect_equal(get_TR(sf), c(2.0, 2.5, 1.5))
  expect_equal(get_total_duration(sf), 580)  # 100*2.0 + 80*2.5 + 120*1.5
  expect_equal(get_run_duration(sf, run_id = 2), 200)  # 80 * 2.5
})

test_that("sampling_frame fmrireg compatibility methods work", {
  sf <- sampling_frame(TR = 2.0, run_lengths = c(100, 80))
  
  # Test fmrireg-style accessors
  expect_equal(samples(sf), 1:180)
  expect_equal(blockids(sf), c(rep(1, 100), rep(2, 80)))
  expect_equal(blocklens(sf), c(100, 80))
  
  # Global onsets should account for TR and start_time (defaults to TR/2 = 1)
  # Run 1: starts at 1, goes to 1 + (100-1)*2 = 199
  # Run 2: starts at 1 + 100*2 = 201, goes to 201 + (80-1)*2 = 359
  expected_onsets <- c(seq(1, 199, 2), seq(201, 359, 2))
  expect_equal(global_onsets(sf), expected_onsets)
})

test_that("sampling_frame with start_time works", {
  sf <- sampling_frame(TR = 2.0, run_lengths = c(50, 50), start_time = 10)
  
  # Global onsets should start at start_time = 10
  # Run 1: starts at 10, goes to 10 + (50-1)*2 = 108
  # Run 2: starts at 10 + 50*2 = 110, goes to 110 + (50-1)*2 = 208
  expected_onsets <- c(seq(10, 108, 2), seq(110, 208, 2))
  expect_equal(global_onsets(sf), expected_onsets)
})

test_that("sampling_frame print method works", {
  sf <- sampling_frame(TR = 2.0, run_lengths = c(100, 80))
  
  expect_output(print(sf), "sampling_frame")
  expect_output(print(sf), "TR: 2")
  expect_output(print(sf), "Runs: 2")
  expect_output(print(sf), "Total timepoints: 180")
})

test_that("is.sampling_frame works", {
  sf <- sampling_frame(TR = 2.0, run_lengths = 100)
  
  expect_true(is.sampling_frame(sf))
  expect_false(is.sampling_frame(list(TR = 2.0)))
  expect_false(is.sampling_frame(data.frame(TR = 2.0)))
  expect_false(is.sampling_frame(NULL))
}) 