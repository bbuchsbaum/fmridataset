skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE),
              paste("Package", pkg, "not available"))
}

context("BIDS interface backend helpers")

test_that("bidser backend discovery helpers work", {
  skip_if_not_installed("bidser")

  fs <- data.frame(
    subid = c("sub-01"),
    datatype = c("func"),
    suffix = c("bold"),
    fmriprep = c(FALSE),
    task = c("rest"),
    stringsAsFactors = FALSE
  )

  mock_bids <- bidser::create_mock_bids(
    project_name = "backend_test",
    participants = c("sub-01"),
    file_structure = fs
  )

  backend <- bids_backend("bidser", backend_config = list(prefer_preproc = FALSE))

  scans <- backend$find_scans(mock_bids, list(subjects = "01"))
  expect_true(is.character(scans))

  subs <- discover_subjects(backend, mock_bids)
  expect_true(length(subs) >= 1)
})

