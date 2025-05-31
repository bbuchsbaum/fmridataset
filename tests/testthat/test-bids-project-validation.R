# Validation tests for as.fmri_dataset.bids_project
# Skip if bidser package not available
skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE),
              paste("Package", pkg, "not available"))
}

 test_that("argument checks catch invalid inputs", {
  skip_if_not_installed("bidser")

  # minimal mock project
  file_structure_df <- data.frame(
    subid = "sub-01",
    datatype = "func",
    suffix = "bold",
    fmriprep = FALSE,
    task = "rest",
    stringsAsFactors = FALSE
  )
  mock_bids <- bidser::create_mock_bids(
    project_name = "validation_proj",
    participants = "sub-01",
    file_structure = file_structure_df
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "01", task_id = c("rest", "task")),
    "task_id must be a single, non-NA character string or NULL"
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "01", session_id = NA_character_),
    "session_id must be a single, non-NA character string or NULL"
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "01", run_ids = c(1, -1)),
    "run_ids must be a numeric vector of positive integers or NULL"
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "01", image_type = c("raw", "preproc")),
    "image_type must be a single, non-NA character string"
  )

  expect_error(
    as.fmri_dataset(mock_bids, subject_id = "01", event_table_source = c("auto", "events")),
    "event_table_source must be a single, non-NA character string"
  )
})
