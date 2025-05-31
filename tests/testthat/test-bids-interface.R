# Tests for bids_backend and bids_query interfaces

# Helper to create a minimal custom backend
mock_backend <- function() {
  bids_backend(
    "custom",
    backend_config = list(
      find_scans = function(bids_root, filters) NULL,
      read_metadata = function(scan_path) list(),
      get_run_info = function(paths) list()
    )
  )
}

# ---------------------------------------------------------------------------
# bids_backend ----------------------------------------------------------------

test_that("bids_backend errors on unsupported backend", {
  expect_error(bids_backend("unknown"), "Unsupported backend_type")
})

test_that("bids_backend custom requires required functions", {
  expect_error(
    bids_backend("custom", backend_config = list(find_scans = function() NULL)),
    "Custom backend requires"
  )
})

test_that("bids_backend custom returns object with functions", {
  backend <- mock_backend()
  expect_s3_class(backend, c("bids_backend", "bids_backend_custom"))
  expect_true(is.function(backend$find_scans))
  expect_true(is.function(backend$read_metadata))
  expect_true(is.function(backend$get_run_info))
  expect_true(backend$validate_bids())
})

test_that("bids_backend bidser loads when package available", {
  local_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    bidser_find_scans = function(...) "scans",
    bidser_read_metadata = function(...) list(meta = TRUE),
    bidser_get_run_info = function(...) list(info = TRUE),
    bidser_find_derivatives = function(...) "derivs",
    bidser_validate_bids = function(...) TRUE,
    .package = "fmridataset"
  )
  backend <- bids_backend("bidser")
  expect_s3_class(backend, c("bids_backend", "bids_backend_bidser"))
  expect_equal(backend$find_scans("root", list()), "scans")
  expect_true(backend$validate_bids("root"))
})

test_that("bids_backend bidser errors when package missing", {
  local_mocked_bindings(requireNamespace = function(pkg, quietly = TRUE) FALSE,
                        .package = "fmridataset")
  expect_error(bids_backend("bidser"), "bidser package is required")
})

# ---------------------------------------------------------------------------
# bids_query -----------------------------------------------------------------

test_that("bids_query constructs query with custom backend", {
  backend <- mock_backend()
  q <- bids_query("/bids", backend = backend)
  expect_s3_class(q, "bids_query")
  expect_identical(q$backend, backend)
  expect_equal(q$bids_root, "/bids")
})

test_that("bids_query chain methods accumulate filters", {
  backend <- mock_backend()
  q <- bids_query("/bids", backend = backend)
  q <- subject(q, "01")
  q <- task(q, "rest")
  q <- session(q, "1")
  q <- run(q, "2")
  q <- derivatives(q, "fmriprep")
  q <- space(q, "MNI")

  expect_equal(q$filters$subjects, "01")
  expect_equal(q$filters$tasks, "rest")
  expect_equal(q$filters$sessions, "1")
  expect_equal(q$filters$runs, "2")
  expect_equal(q$filters$derivatives, "fmriprep")
  expect_equal(q$filters$spaces, "MNI")
})

test_that("bids_query without backend fails when none available", {
  local_mocked_bindings(requireNamespace = function(pkg, quietly = TRUE) FALSE,
                        .package = "fmridataset")
  expect_error(bids_query("/bids"), "No compatible BIDS backend")
})

test_that("as.fmri_dataset.bids_query errors due to placeholder", {
  backend <- mock_backend()
  q <- bids_query("/bids", backend = backend)
  expect_error(as.fmri_dataset(q, subject_id = "01"),
               "Sophisticated BIDS extraction not yet implemented")
})

