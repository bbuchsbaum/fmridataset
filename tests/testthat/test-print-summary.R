test_that("print.fmri_dataset works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Should print without error and show key information
  expect_output(print(dataset), "fMRI Dataset")
  expect_output(print(dataset), "matrix")
  expect_output(print(dataset), "TR: 2")
  expect_output(print(dataset), "Runs: 2")
  expect_output(print(dataset), "100")  # timepoints
})

test_that("print.fmri_dataset shows different dataset types", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Matrix dataset
  dataset_matrix <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_output(print(dataset_matrix), "matrix")
  
  # Mock file dataset
  dataset_files <- dataset_matrix
  dataset_files$image_matrix <- NULL
  dataset_files$image_paths <- c("run1.nii.gz", "run2.nii.gz")
  dataset_files$metadata$dataset_type <- "file_vec"
  
  expect_output(print(dataset_files), "file_vec")
  expect_output(print(dataset_files), "NIfTI file")
})

test_that("print.fmri_dataset shows event information", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  events <- data.frame(
    onset = c(10, 30, 50),
    duration = c(2, 2, 2),
    trial_type = c("A", "B", "A")
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = events
  )
  
  expect_output(print(dataset), "3 events")
  expect_output(print(dataset), "3 variables")
})

test_that("print.fmri_dataset shows censoring information", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  censor_vector <- rep(TRUE, 100)
  censor_vector[c(10:15, 60:65)] <- FALSE
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = censor_vector
  )
  
  expect_output(print(dataset), "Censoring:")
  expect_output(print(dataset), "12/100")  # 12 censored out of 100
})

test_that("print.fmri_dataset shows masking information", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_output(print(dataset), "Mask:")
  expect_output(print(dataset), "7/10")  # 7 voxels kept out of 10
})

test_that("print.fmri_dataset shows preprocessing options", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    temporal_zscore = TRUE,
    voxelwise_detrend = TRUE
  )
  
  expect_output(print(dataset), "temporal z-score")
  expect_output(print(dataset), "voxelwise detrend")
})

test_that("summary.fmri_dataset works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Should print detailed summary without error
  expect_output(summary(dataset), "fMRI Dataset Summary")
  expect_output(summary(dataset), "BASIC INFORMATION")
  expect_output(summary(dataset), "TEMPORAL STRUCTURE")
  expect_output(summary(dataset), "SPATIAL INFORMATION")
})

test_that("summary.fmri_dataset with validation", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # With validation
  expect_output(summary(dataset, validate = TRUE), "VALIDATION REPORT")
  expect_output(summary(dataset, validate = TRUE), "PASSED")
  
  # Without validation
  output <- capture.output(summary(dataset, validate = FALSE))
  expect_false(any(grepl("VALIDATION REPORT", output)))
})

test_that("summary.fmri_dataset with data statistics", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # With data statistics
  expect_output(
    summary(dataset, include_data_stats = TRUE), 
    "DATA STATISTICS"
  )
  
  # Without data statistics
  output <- capture.output(summary(dataset, include_data_stats = FALSE))
  expect_false(any(grepl("DATA STATISTICS", output)))
})

test_that("summary.fmri_dataset with events", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  events <- data.frame(
    onset = c(10, 30, 50, 70),
    duration = c(2, 2, 2, 2),
    trial_type = c("stimulus", "response", "stimulus", "response")
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = events
  )
  
  expect_output(summary(dataset), "EVENT TABLE ANALYSIS")
  expect_output(summary(dataset), "Number of Events.*4")
  expect_output(summary(dataset), "stimulus.*2")  # Trial type counts
})

test_that("summary.fmri_dataset shows memory information", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_output(summary(dataset), "MEMORY & CACHING")
  expect_output(summary(dataset), "Object Size")
})

test_that("summary.fmri_dataset shows temporal statistics", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1500), nrow = 150, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 60, 40)
  )
  
  expect_output(summary(dataset), "Run Length Statistics")
  expect_output(summary(dataset), "Mean.*50")  # Mean run length
  expect_output(summary(dataset), "Individual Lengths.*50, 60, 40")
})

test_that("summary.fmri_dataset handles complex datasets", {
  set.seed(123)
  test_matrix <- matrix(rnorm(2000), nrow = 100, ncol = 20)
  
  events <- data.frame(
    onset = seq(5, 95, 10),
    duration = rep(c(2, 3), length.out = 10),
    trial_type = rep(c("A", "B"), length.out = 10)
  )
  
  censor_vector <- rep(TRUE, 100)
  censor_vector[c(20:25, 70:75)] <- FALSE
  
  mask_vector <- runif(20) > 0.3
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = events,
    censor_vector = censor_vector,
    temporal_zscore = TRUE,
    metadata = list(
      experiment = "complex_task",
      subject_id = "sub-001"
    )
  )
  
  # Should handle all components
  expect_output(summary(dataset), "complex_task")
  expect_output(summary(dataset), "sub-001")
  expect_output(summary(dataset), "EVENT TABLE ANALYSIS")
  expect_output(summary(dataset), "temporal z-score")
})

test_that("print and summary return invisibly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Should return the dataset object invisibly
  expect_identical(
    suppressMessages(print(dataset)), 
    dataset
  )
  
  expect_identical(
    suppressMessages(summary(dataset)), 
    dataset
  )
})

test_that("print handles edge cases gracefully", {
  set.seed(123)
  
  # Very small dataset
  small_matrix <- matrix(rnorm(20), nrow = 10, ncol = 2)
  small_dataset <- fmri_dataset_create(
    images = small_matrix,
    TR = 1.0,
    run_lengths = 10
  )
  
  expect_output(print(small_dataset), "fMRI Dataset")
  
  # Single voxel dataset
  single_voxel <- matrix(rnorm(100), nrow = 100, ncol = 1)
  single_dataset <- fmri_dataset_create(
    images = single_voxel,
    TR = 2.0,
    run_lengths = 100
  )
  
  expect_output(print(single_dataset), "1.*voxel")
}) 