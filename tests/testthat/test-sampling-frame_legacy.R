# Legacy Sampling Frame Tests from fmrireg
# These tests ensure backward compatibility with the original fmrireg implementation

test_that("sampling_frame constructor works correctly", {
  # Basic construction
  sframe <- sampling_frame(blocklens = c(100, 100), TR = 2)
  expect_s3_class(sframe, "sampling_frame")
  expect_equal(length(sframe$blocklens), 2)
  expect_equal(sframe$TR, c(2, 2))
  expect_equal(sframe$start_time, c(1, 1))
  
  # Test with different TRs per block
  sframe2 <- sampling_frame(blocklens = c(100, 200), TR = c(2, 1.5))
  expect_equal(sframe2$TR, c(2, 1.5))
  
  # Test input validation
  expect_error(sampling_frame(blocklens = c(-1, 100), TR = 2), 
              "run_lengths must be positive")
  expect_error(sampling_frame(blocklens = c(100, 100), TR = -1), 
              "TR values must be positive")
  expect_error(sampling_frame(blocklens = c(100, 100), TR = 2, precision = 3),
              "Precision must be positive and less than")
})

test_that("samples.sampling_frame works correctly", {
  sframe <- sampling_frame(blocklens = c(100, 100), TR = 2)
  
  # Test basic functionality - our implementation returns timepoint indices
  samples_result <- samples(sframe)
  expect_equal(length(samples_result), 200)
  expect_equal(samples_result[1:5], c(1, 2, 3, 4, 5))  # Sequential indices
  
  # Test block selection
  block1_samples <- samples(sframe, blockids = 1)
  expect_equal(length(block1_samples), 100)
  expect_equal(block1_samples[1:5], c(1, 2, 3, 4, 5))
})

test_that("global_onsets works correctly", {
  sframe <- sampling_frame(blocklens = c(100, 100), TR = 2)
  
  # Test basic functionality with arguments
  onsets <- c(10, 20)
  blockids <- c(1, 2)
  global_times <- global_onsets(sframe, onsets, blockids)
  expect_equal(length(global_times), 2)
  expect_equal(global_times[1], 10)  # First block onset unchanged
  expect_equal(global_times[2], 220)  # Second block onset = 200 (block1 duration) + 20
  
  # Test without arguments - should return all timepoint onsets
  all_onsets <- global_onsets(sframe)
  expect_equal(length(all_onsets), 200)
  expect_equal(all_onsets[1], 1)  # Start time + 0 * TR
  expect_equal(all_onsets[2], 3)  # Start time + 1 * TR
})

test_that("print.sampling_frame works correctly", {
  sframe <- sampling_frame(blocklens = c(100, 100), TR = 2)
  expect_output(print(sframe), "sampling_frame")
  expect_output(print(sframe), "Runs")
  expect_output(print(sframe), "Total timepoints")
  expect_output(print(sframe), "TR")
})

test_that("sampling_frame handles edge cases", {
  # Single block
  single_block <- sampling_frame(blocklens = 100, TR = 2)
  expect_equal(length(single_block$blocklens), 1)
  expect_equal(length(samples(single_block)), 100)
  
  # Very short block
  short_block <- sampling_frame(blocklens = c(1, 1), TR = 2)
  expect_equal(length(samples(short_block)), 2)
  
  # Different start times
  custom_starts <- sampling_frame(blocklens = c(100, 100), 
                                TR = 2, 
                                start_time = c(0, 5))
  expect_equal(custom_starts$start_time, c(0, 5))
  
  # High precision
  high_prec <- sampling_frame(blocklens = c(10, 10), 
                            TR = 2, 
                            precision = 0.01)
  expect_equal(high_prec$precision, 0.01)
})

test_that("sampling_frame maintains temporal consistency", {
  sframe <- sampling_frame(blocklens = c(100, 100, 100), TR = 2)
  
  # Test global onsets temporal consistency
  all_onsets <- global_onsets(sframe)
  
  # Check uniform spacing within blocks (TR = 2)
  # First 100 timepoints should have spacing of 2
  expect_equal(diff(all_onsets[1:5]), rep(2, 4))
  
  # Second block should continue with proper spacing
  expect_equal(diff(all_onsets[101:105]), rep(2, 4))
  
  # Third block should continue with proper spacing  
  expect_equal(diff(all_onsets[201:205]), rep(2, 4))
})

test_that("blockids method works correctly", {
  sframe <- sampling_frame(blocklens = c(100, 100, 50), TR = 2)
  
  block_ids <- blockids(sframe)
  expect_equal(length(block_ids), 250)  # Total timepoints
  expect_equal(sum(block_ids == 1), 100)  # First block
  expect_equal(sum(block_ids == 2), 100)  # Second block  
  expect_equal(sum(block_ids == 3), 50)   # Third block
  
  # Check sequence
  expect_equal(block_ids[1:3], c(1, 1, 1))
  expect_equal(block_ids[99:101], c(1, 1, 2))
  expect_equal(block_ids[199:201], c(2, 2, 3))
})

test_that("blocklens method works correctly", {
  sframe <- sampling_frame(blocklens = c(100, 150, 75), TR = 2)
  
  lens <- blocklens(sframe)
  expect_equal(lens, c(100, 150, 75))
  expect_equal(length(lens), 3)
}) 