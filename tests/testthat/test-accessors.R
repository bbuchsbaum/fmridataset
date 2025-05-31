test_that("get_data_matrix works with matrix datasets", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Get full data matrix
  data_matrix <- get_data_matrix(dataset)
  expect_equal(data_matrix, test_matrix)
  expect_equal(dim(data_matrix), c(100, 10))
  
  # Get data for specific run
  run1_data <- get_data_matrix(dataset, run_id = 1)
  expect_equal(run1_data, test_matrix[1:50, ])
  
  run2_data <- get_data_matrix(dataset, run_id = 2)
  expect_equal(run2_data, test_matrix[51:100, ])
  
  # Get data for multiple runs
  both_runs_data <- get_data_matrix(dataset, run_id = c(1, 2))
  expect_equal(both_runs_data, test_matrix)
})

test_that("get_data_matrix applies masking correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  masked_data <- get_data_matrix(dataset)
  expected_masked <- test_matrix[, mask_vector]
  
  expect_equal(masked_data, expected_masked)
  expect_equal(ncol(masked_data), sum(mask_vector))
})

test_that("get_data_matrix applies censoring correctly", {
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
  
  censored_data <- get_data_matrix(dataset)
  expected_censored <- test_matrix[censor_vector, ]
  
  expect_equal(censored_data, expected_censored)
  expect_equal(nrow(censored_data), sum(censor_vector))
})

test_that("get_data_matrix applies preprocessing correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    temporal_zscore = TRUE,
    voxelwise_detrend = TRUE
  )
  
  # With preprocessing
  processed_data <- get_data_matrix(dataset, apply_preprocessing = TRUE)
  expect_false(identical(processed_data, test_matrix))
  
  # Without preprocessing
  raw_data <- get_data_matrix(dataset, apply_preprocessing = FALSE)
  expect_equal(raw_data, test_matrix)
})

test_that("sampling_frame accessors work correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(40, 60)
  )
  
  sf <- get_sampling_frame(dataset)
  expect_s3_class(sf, "sampling_frame")
  
  # Test accessor functions
  expect_equal(get_TR(dataset), c(2.0, 2.0))
  expect_equal(get_run_lengths(dataset), c(40, 60))
  expect_equal(get_num_runs(dataset), 2)
  expect_equal(get_num_timepoints(dataset), 100)
  expect_equal(get_num_timepoints(dataset, run_id = 1), 40)
  expect_equal(get_num_timepoints(dataset, run_id = 2), 60)
})

test_that("event table accessors work correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  events <- data.frame(
    onset = c(10, 30, 50, 70),
    duration = c(2, 2, 2, 2),
    trial_type = c("A", "B", "A", "B")
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    event_table = events
  )
  
  retrieved_events <- get_event_table(dataset)
  expect_equal(retrieved_events, tibble::as_tibble(events))
  
  # Dataset without events
  dataset_no_events <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_null(get_event_table(dataset_no_events))
})

test_that("censor vector accessors work correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  censor_vector <- rep(TRUE, 100)
  censor_vector[c(20:25, 70:75)] <- FALSE
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = censor_vector
  )
  
  retrieved_censor <- get_censor_vector(dataset)
  expect_equal(retrieved_censor, censor_vector)
  
  # Dataset without censoring
  dataset_no_censor <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_null(get_censor_vector(dataset_no_censor))
})

test_that("metadata accessors work correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  extra_metadata <- list(
    experiment = "task_switching",
    subject_id = "sub-002"
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    temporal_zscore = TRUE,
    metadata = extra_metadata
  )
  
  # Get all metadata
  metadata <- get_metadata(dataset)
  expect_equal(metadata$experiment, "task_switching")
  expect_equal(metadata$subject_id, "sub-002")
  expect_true(metadata$matrix_options$temporal_zscore)
  
  # Get specific metadata field
  expect_equal(get_metadata(dataset, "experiment"), "task_switching")
  expect_equal(get_metadata(dataset, "dataset_type"), "matrix")
  
  # Dataset type accessor
  expect_equal(get_dataset_type(dataset), "matrix")
})

test_that("get_num_voxels works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Without mask
  dataset_no_mask <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_equal(get_num_voxels(dataset_no_mask), 10)
  
  # With mask
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset_with_mask <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_equal(get_num_voxels(dataset_with_mask), sum(mask_vector))
})

test_that("accessor invariance across dataset types works", {
  # This tests that accessors return consistent results regardless of dataset_type
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Matrix dataset
  dataset_matrix <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Mock memory_vec dataset
  dataset_memory <- dataset_matrix
  dataset_memory$image_matrix <- NULL
  dataset_memory$image_objects <- list(test_matrix[1:50, ], test_matrix[51:100, ])
  dataset_memory$metadata$dataset_type <- "memory_vec"
  
  # Should get same results from accessors
  expect_equal(get_TR(dataset_matrix), get_TR(dataset_memory))
  expect_equal(get_run_lengths(dataset_matrix), get_run_lengths(dataset_memory))
  expect_equal(get_num_runs(dataset_matrix), get_num_runs(dataset_memory))
  expect_equal(get_num_timepoints(dataset_matrix), get_num_timepoints(dataset_memory))
})

test_that("get_mask_volume works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  # Get mask as vector
  mask_vec <- get_mask_volume(dataset, as_vector = TRUE)
  expect_equal(mask_vec, mask_vector)
  
  # Dataset without mask
  dataset_no_mask <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_null(get_mask_volume(dataset_no_mask))
})

test_that("get_image_source_type works correctly", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Matrix dataset
  dataset_matrix <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_equal(get_image_source_type(dataset_matrix), "matrix")
  
  # Mock file dataset
  dataset_files <- dataset_matrix
  dataset_files$image_matrix <- NULL
  dataset_files$image_paths <- c("file1.nii", "file2.nii")
  dataset_files$metadata$dataset_type <- "file_vec"
  
  expect_equal(get_image_source_type(dataset_files), "character")
}) 