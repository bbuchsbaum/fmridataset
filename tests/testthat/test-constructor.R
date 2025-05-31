test_that("fmri_dataset_create works with matrix input", {
  # Create test matrix
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Basic matrix dataset
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_s3_class(dataset, "fmri_dataset")
  expect_equal(get_dataset_type(dataset), "matrix")
  expect_equal(dataset$image_matrix, test_matrix)
  expect_null(dataset$image_paths)
  expect_null(dataset$image_objects)
  
  # Check sampling frame
  sf <- get_sampling_frame(dataset)
  expect_equal(get_TR(sf), c(2.0, 2.0))
  expect_equal(get_run_lengths(sf), c(50, 50))
  expect_equal(n_timepoints(sf), 100)
})

test_that("fmri_dataset_create works with file paths", {
  # Skip if no test data available
  skip_if_not(file.exists("test_data"), "Test data not available")
  
  # Create mock file paths (would need actual test files)
  file_paths <- c("run1.nii.gz", "run2.nii.gz")
  
  dataset <- fmri_dataset_create(
    images = file_paths,
    TR = 2.5,
    run_lengths = c(180, 180),
    base_path = "/tmp"
  )
  
  expect_equal(get_dataset_type(dataset), "file_vec")
  expect_equal(length(dataset$image_paths), 2)
  expect_null(dataset$image_matrix)
  expect_null(dataset$image_objects)
})

test_that("fmri_dataset_create validates inputs correctly", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # TR validation
  expect_error(
    fmri_dataset_create(images = test_matrix, TR = -1.0, run_lengths = 100),
    "TR values must be positive"
  )
  
  # Run lengths validation
  expect_error(
    fmri_dataset_create(images = test_matrix, TR = 2.0, run_lengths = c(50, 0)),
    "run_lengths must be positive"
  )
  
  # Dimension mismatch
  expect_error(
    fmri_dataset_create(images = test_matrix, TR = 2.0, run_lengths = c(60, 60)),
    "Sum of run_lengths \\(120\\) does not match"
  )
})

test_that("fmri_dataset_create handles event tables", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Data frame event table
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
  
  expect_equal(get_event_table(dataset), tibble::as_tibble(events))
})

test_that("fmri_dataset_create handles censoring", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Logical censoring vector
  censor_vector <- rep(TRUE, 100)
  censor_vector[c(10:15, 60:65)] <- FALSE
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = censor_vector
  )
  
  expect_equal(get_censor_vector(dataset), censor_vector)
  
  # Numeric censoring vector
  censor_numeric <- ifelse(censor_vector, 1, 0)
  
  dataset_numeric <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    censor_vector = censor_numeric
  )
  
  expect_equal(get_censor_vector(dataset_numeric), censor_numeric)
})

test_that("fmri_dataset_create handles masking", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  # Logical mask vector
  mask_vector <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = mask_vector,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_equal(dataset$mask_vector, mask_vector)
  expect_null(dataset$mask_path)
  expect_null(dataset$mask_object)
})

test_that("fmri_dataset_create handles preprocessing options", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    temporal_zscore = TRUE,
    voxelwise_detrend = TRUE
  )
  
  metadata <- get_metadata(dataset)
  expect_true(metadata$matrix_options$temporal_zscore)
  expect_true(metadata$matrix_options$voxelwise_detrend)
})

test_that("fmri_dataset_create handles extra metadata", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  extra_metadata <- list(
    experiment = "working_memory",
    subject_id = "sub-001",
    session_date = Sys.Date()
  )
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50),
    metadata = extra_metadata
  )
  
  metadata <- get_metadata(dataset)
  expect_equal(metadata$experiment, "working_memory")
  expect_equal(metadata$subject_id, "sub-001")
  expect_equal(metadata$session_date, Sys.Date())
})

test_that("fmri_dataset_create data cache is initialized", {
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  
  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(50, 50)
  )
  
  expect_true(is.environment(dataset$data_cache))
  expect_equal(ls(dataset$data_cache), character(0))  # Should be empty initially
})

test_that("fmri_dataset_create validates dataset_type determination", {
  # Matrix type
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
  dataset_matrix <- fmri_dataset_create(test_matrix, TR = 2.0, run_lengths = 100)
  expect_equal(get_dataset_type(dataset_matrix), "matrix")
  
  # File type (mocked)
  dataset_files <- structure(list(), class = "fmri_dataset")
  dataset_files$image_paths <- c("file1.nii", "file2.nii")
  dataset_files$metadata <- list(dataset_type = "file_vec")
  expect_equal(get_dataset_type(dataset_files), "file_vec")
  
  # Memory type (mocked)
  dataset_memory <- structure(list(), class = "fmri_dataset")
  dataset_memory$image_objects <- list("obj1", "obj2")
  dataset_memory$metadata <- list(dataset_type = "memory_vec")
  expect_equal(get_dataset_type(dataset_memory), "memory_vec")
}) 