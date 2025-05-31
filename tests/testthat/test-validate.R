test_that("validate_fmri_dataset passes for well-formed datasets", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Should pass validation
  expect_true(validate_fmri_dataset(dataset))
  expect_true(validate_fmri_dataset(dataset, check_data_load = TRUE))
})

test_that("validate_fmri_dataset detects object structure issues", {
  # Invalid class
  fake_dataset <- list(
    sampling_frame = sampling_frame(TR = 2.0, run_lengths = 100),
    metadata = list(dataset_type = "matrix"),
    data_cache = new.env()
  )
  
  expect_error(
    validate_fmri_dataset(fake_dataset),
    "Object is not of class 'fmri_dataset'"
  )
  
  # Missing required fields
  incomplete_dataset <- structure(list(), class = "fmri_dataset")
  expect_error(
    validate_fmri_dataset(incomplete_dataset),
    "Missing required fields"
  )
})

test_that("validate_fmri_dataset detects sampling frame issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Corrupt sampling frame
  dataset$sampling_frame$total_timepoints <- 999  # Wrong value
  
  expect_error(
    validate_fmri_dataset(dataset),
    "sampling_frame internal inconsistency"
  )
})

test_that("validate_fmri_dataset detects data source issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # No data source
  dataset$image_matrix <- NULL
  
  expect_error(
    validate_fmri_dataset(dataset),
    "No image data source found"
  )
  
  # Multiple data sources
  dataset$image_matrix <- test_matrix
  dataset$image_paths <- c("file1.nii", "file2.nii")
  
  expect_error(
    validate_fmri_dataset(dataset),
    "Multiple image data sources found"
  )
})

test_that("validate_fmri_dataset detects dimensional inconsistencies", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Create a valid dataset first
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Then manually corrupt the sampling frame to test validation
  dataset$sampling_frame$total_timepoints <- 120  # Wrong value
  dataset$sampling_frame$blocklens <- c(60, 60)   # Wrong values
  
  expect_error(
    validate_fmri_dataset(dataset),
    "Matrix dimension mismatch"
  )
})

test_that("validate_fmri_dataset detects mask compatibility issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  wrong_mask <- rep(TRUE, 15)  # Wrong length
  
  expect_error(
    fmri_dataset_create(
      images = test_matrix,
      mask = wrong_mask,
      TR = 2.0,
      run_lengths = c(50, 50)
    ),
    "Mask length"
  )
})

test_that("validate_fmri_dataset detects event table issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Events beyond temporal bounds
  bad_events <- data.frame(
    onset = c(10, 30, 250),  # 250 is beyond 200 seconds (100 * 2.0)
    duration = c(2, 2, 2),
    trial_type = c("A", "B", "A")
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = bad_events
  )
  
  expect_error(
    validate_fmri_dataset(dataset),
    "onset values beyond total duration"
  )
})

test_that("validate_fmri_dataset detects censor vector issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Create a valid dataset first
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Then manually corrupt the censor vector to test validation
  wrong_censor <- rep(TRUE, 80)  # Wrong length
  dataset$censor_vector <- wrong_censor
  
  expect_error(
    validate_fmri_dataset(dataset),
    "censor_vector length"
  )
})

test_that("validate_fmri_dataset detects metadata issues", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Corrupt metadata TR
  dataset$metadata$TR <- 3.0  # Different from sampling_frame
  
  expect_error(
    validate_fmri_dataset(dataset),
    "metadata\\$TR.*does not match sampling_frame\\$TR"
  )
})

test_that("validate_fmri_dataset verbose mode works", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Should print progress messages
  expect_output(
    validate_fmri_dataset(dataset, verbose = TRUE),
    "Validating fmri_dataset object"
  )
  
  expect_output(
    validate_fmri_dataset(dataset, verbose = TRUE),
    "All validation checks passed"
  )
})

test_that("validate_fmri_dataset handles edge cases", {
  set.seed(123)
  test_matrix <- matrix(rnorm(100), nrow = 50, ncol = 2)
  
  # Single run dataset
  dataset_single <- fmri_dataset_create(
    images = test_matrix,
    TR = 1.5,
    run_lengths = 50
  )
  
  expect_true(validate_fmri_dataset(dataset_single))
  
  # Dataset with no events or censoring
  expect_true(validate_fmri_dataset(dataset_single))
  
  # Dataset with events but no duration
  events_no_duration <- data.frame(
    onset = c(10, 30),
    trial_type = c("A", "B")
  )
  
  dataset_no_duration <- fmri_dataset_create(
    images = test_matrix,
    TR = 1.5,
    run_lengths = 50,
    event_table = events_no_duration
  )
  
  expect_true(validate_fmri_dataset(dataset_no_duration))
})

test_that("is.fmri_dataset works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_true(is.fmri_dataset(dataset))
  
  # Not fmri_dataset objects
  expect_false(is.fmri_dataset(list()))
  expect_false(is.fmri_dataset(data.frame()))
  expect_false(is.fmri_dataset(test_matrix))
  expect_false(is.fmri_dataset(NULL))
  expect_false(is.fmri_dataset("not_a_dataset"))
})

test_that("validate_fmri_dataset with realistic complex dataset", {
  set.seed(123)
  test_matrix <- matrix(rnorm(3000), nrow = 150, ncol = 20)
  
  # Complex events
  events <- data.frame(
    onset = seq(5, 145, 10),
    duration = rep(c(2, 3, 2), length.out = 15),
    trial_type = rep(c("stimulus", "response", "rest"), length.out = 15),
    response_time = runif(15, 0.5, 2.5)
  )
  
  # Censoring some timepoints
  censor_vector <- rep(TRUE, 150)
  censor_vector[c(20:25, 80:85, 120:125)] <- FALSE
  
  # Mask
  mask_vector <- runif(20) > 0.2
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50, 50),
    event_table = events,
    censor_vector = censor_vector,
    temporal_zscore = TRUE,
    voxelwise_detrend = TRUE,
    metadata = list(
      experiment = "complex_task",
      subject_id = "sub-001"
    )
  )
  
  # Should pass all validation
  expect_true(validate_fmri_dataset(dataset))
  expect_true(validate_fmri_dataset(dataset, check_data_load = TRUE, verbose = FALSE))
})

test_that("validate_fmri_dataset catches real-world edge cases", {
  set.seed(123)
  
  # Case 1: Censoring reduces timepoints but sampling_frame expects full count
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  heavy_censor <- rep(FALSE, 100)
  heavy_censor[1:20] <- TRUE  # Only keep 20 timepoints
  
  # This should create a dataset but validation might catch inconsistencies
  dataset_heavy_censor <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = heavy_censor
  )
  
  # Validation should still pass because the framework handles this correctly
  expect_true(validate_fmri_dataset(dataset_heavy_censor))
  
  # Case 2: Events with zero duration
  events_zero_duration <- data.frame(
    onset = c(10, 30, 50),
    duration = c(0, 2, 0),
    trial_type = c("spike", "block", "spike")
  )
  
  dataset_zero_duration <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = events_zero_duration
  )
  
  expect_true(validate_fmri_dataset(dataset_zero_duration))
}) 