test_that("validate_fmri_dataset_structure detects multiple image sources", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)

  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = 100
  )

  # Introduce additional image source
  dataset$image_paths <- c("img1.nii.gz")

  expect_error(
    validate_fmri_dataset_structure(dataset),
    "Exactly one image source must be populated"
  )
})


test_that("validate_fmri_dataset_structure detects multiple mask sources", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)

  dataset <- fmri_dataset_create(
    images = test_matrix,
    mask = rep(TRUE, 10),
    TR = 2.0,
    run_lengths = 100
  )

  # Introduce additional mask source
  dataset$mask_path <- "mask.nii.gz"

  expect_error(
    validate_fmri_dataset_structure(dataset),
    "At most one mask source"
  )
})


test_that("validate_fmri_dataset_structure requires sampling_frame", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)

  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = 100
  )

  dataset$sampling_frame <- NULL

  expect_error(
    validate_fmri_dataset_structure(dataset),
    "sampling_frame is required"
  )
})


test_that("validate_fmri_dataset_structure checks dataset_type consistency", {
  set.seed(123)
  test_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)

  dataset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = 100
  )

  dataset$metadata$dataset_type <- "file_vec"

  expect_error(
    validate_fmri_dataset_structure(dataset),
    "dataset_type 'file_vec' requires character images input"
  )
})

