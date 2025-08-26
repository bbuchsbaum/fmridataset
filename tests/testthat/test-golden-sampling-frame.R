# Golden tests for sampling_frame objects

test_that("sampling_frame produces consistent output", {
  ref_data <- load_golden_data("reference_data")

  # Create sampling frame
  sframe <- fmrihrf::sampling_frame(
    TR = ref_data$metadata$TR,
    blocklens = ref_data$metadata$run_lengths
  )

  # Load expected
  expected <- load_golden_data("sampling_frame")

  # Test structure
  expect_s3_class(sframe, "sampling_frame")
  expect_equal(class(sframe), class(expected))

  # Test components
  expect_equal(sframe$TR, expected$TR)
  expect_equal(sframe$run_length, expected$run_length)
  expect_equal(sframe$nruns, expected$nruns)
})

test_that("sampling_frame accessors work consistently", {
  ref_data <- load_golden_data("reference_data")

  sframe <- fmrihrf::sampling_frame(
    TR = ref_data$metadata$TR,
    blocklens = ref_data$metadata$run_lengths
  )

  # Test all accessors
  expect_equal(blocklens(sframe), ref_data$metadata$run_lengths)
  expect_equal(
    get_total_duration(sframe),
    sum(ref_data$metadata$run_lengths) * ref_data$metadata$TR
  )
  expect_equal(n_timepoints(sframe), sum(ref_data$metadata$run_lengths))
  expect_equal(n_runs(sframe), length(ref_data$metadata$run_lengths))
})

test_that("sampling_frame print output matches snapshot", {
  sframe <- fmrihrf::sampling_frame(TR = 2, blocklens = c(100, 100, 150))

  expect_snapshot({
    print(sframe)
  })
})

test_that("single-run sampling_frame handles correctly", {
  # Single run
  sframe_single <- fmrihrf::sampling_frame(TR = 2.5, blocklens = 200)

  expect_equal(n_runs(sframe_single), 1)
  expect_equal(n_timepoints(sframe_single), 200)
  expect_equal(blocklens(sframe_single), 200)
  expect_equal(get_total_duration(sframe_single), 200 * 2.5)
})

test_that("sampling_frame with event onsets works", {
  ref_data <- load_golden_data("reference_data")

  # Create frame with events
  sframe <- fmrihrf::sampling_frame(
    TR = ref_data$metadata$TR,
    blocklens = sum(ref_data$metadata$run_lengths),
    event_table = ref_data$block_info
  )

  # Test event integration
  expect_true(!is.null(sframe$event_table))
  expect_equal(nrow(sframe$event_table), nrow(ref_data$block_info))
  expect_equal(sframe$event_table$onset, ref_data$block_info$onset)
})

test_that("sampling_frame validation works", {
  # Valid frames
  expect_silent(fmrihrf::sampling_frame(TR = 2, blocklens = 100))
  expect_silent(fmrihrf::sampling_frame(TR = 2.5, blocklens = c(100, 150)))

  # Invalid TR
  expect_error(
    fmrihrf::sampling_frame(TR = -1, blocklens = 100),
    "TR must be positive"
  )

  expect_error(
    fmrihrf::sampling_frame(TR = 0, blocklens = 100),
    "TR must be positive"
  )

  # Invalid run lengths
  expect_error(
    fmrihrf::sampling_frame(TR = 2, blocklens = numeric(0)),
    "blocklens"
  )

  expect_error(
    fmrihrf::sampling_frame(TR = 2, blocklens = c(100, -50)),
    "positive"
  )
})

test_that("sampling_frame conversion maintains consistency", {
  ref_data <- load_golden_data("reference_data")

  # Create dataset with sampling frame
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  # Extract sampling frame
  sframe <- dset$sampling_frame

  expect_s3_class(sframe, "sampling_frame")
  expect_equal(sframe$TR, ref_data$metadata$TR)
  expect_equal(sum(sframe$run_length), ncol(ref_data$matrix_data))
})
