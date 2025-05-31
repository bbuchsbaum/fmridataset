skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE),
              paste("Package", pkg, "not available"))
}


test_that("BIDS datasets set dataset_type correctly", {
  skip_if_not_installed("bidser")
  skip_if_not_installed("neuroim2")

  file_structure_df <- data.frame(
    subid = c("sub-01"),
    datatype = c("func"),
    suffix = c("bold"),
    fmriprep = c(FALSE),
    task = c("rest"),
    stringsAsFactors = FALSE
  )

  mock_bids <- bidser::create_mock_bids(
    project_name = "dataset_type_test",
    participants = c("sub-01"),
    file_structure = file_structure_df
  )

  dset_file <- as.fmri_dataset(
    mock_bids,
    subject_id = "01",
    task_id = "rest",
    preload_data = FALSE
  )

  expect_equal(get_dataset_type(dset_file), "bids_file")

  dset_mem <- as.fmri_dataset(
    mock_bids,
    subject_id = "01",
    task_id = "rest",
    preload_data = TRUE
  )

  expect_equal(get_dataset_type(dset_mem), "bids_mem")
})
