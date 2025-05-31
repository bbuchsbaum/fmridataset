# Tests for as.fmri_dataset with BIDS projects

# Helper to skip tests when package is not installed
skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE),
              paste("Package", pkg, "not available"))
}

## Test conversion from bidser mock project

test_that("as.fmri_dataset converts mock BIDS project", {
  skip_if_not_installed("bidser")

  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    task = c("rest", "rest"),
    stringsAsFactors = FALSE
  )

  mock_bids <- bidser::create_mock_bids(
    project_name = "dataset_test",
    participants = c("sub-01"),
    file_structure = file_structure_df
  )

  dset <- as.fmri_dataset(
    mock_bids,
    subject_id = "01",
    task_id = "rest",
    preload_data = FALSE
  )

  expect_s3_class(dset, "fmri_dataset")
  expect_equal(get_dataset_type(dset), "bids_file")
  meta <- get_metadata(dset)
  expect_equal(meta$bids_info$subject_id, "01")
  expect_equal(meta$bids_info$task_id, "rest")
  expect_true(meta$TR > 0)
})

## Test preload behavior

test_that("preloading returns bids_mem dataset", {
  skip_if_not_installed("bidser")

  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    task = c("rest", "rest"),
    stringsAsFactors = FALSE
  )

  mock_bids <- bidser::create_mock_bids(
    project_name = "dataset_test_preload",
    participants = c("sub-01"),
    file_structure = file_structure_df
  )

  dset <- as.fmri_dataset(
    mock_bids,
    subject_id = "01",
    task_id = "rest",
    preload_data = TRUE
  )

  expect_equal(get_dataset_type(dset), "bids_mem")
  meta <- get_metadata(dset)
  expect_true(meta$file_options$preload)
})

## Test informative error on missing scans

test_that("missing scans trigger informative error", {
  skip_if_not_installed("bidser")

  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    task = c("rest", "rest"),
    stringsAsFactors = FALSE
  )

  mock_bids <- bidser::create_mock_bids(
    project_name = "dataset_test_missing",
    participants = c("sub-01"),
    file_structure = file_structure_df
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "02", task_id = "rest"),
    "No raw functional scans found for subject"
  )
})

## Test error message for missing packages

test_that("missing package produces informative error", {
  expect_error(
    check_package_available("totallyfakepkg", "testing", error = TRUE),
    "Package 'totallyfakepkg' is required for testing but is not installed"
  )
})

