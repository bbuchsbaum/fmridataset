# Tests for bids_h5_scan_backend and h5_shared_connection
#
# These tests cover the foundational layer: the scan backend contract,
# ref-counted shared connection, and event/confound/censor helpers.
# Tests requiring a real HDF5 file are skipped unless hdf5r is available.

# ==============================================================================
# Helpers
# ==============================================================================

make_mock_conn <- function(ref_count = 0L) {
  conn <- new.env(parent = emptyenv())
  conn$file      <- "fake_study.h5"
  conn$ref_count <- as.integer(ref_count)
  conn$handle    <- list(is_valid = TRUE)
  conn$acquire   <- function() { conn$ref_count <- conn$ref_count + 1L; invisible(NULL) }
  conn$release   <- function() {
    conn$ref_count <- conn$ref_count - 1L
    if (conn$ref_count < 0L) conn$ref_count <- 0L
    invisible(NULL)
  }
  class(conn) <- "h5_shared_connection"
  conn
}

make_scan_backend <- function(conn = make_mock_conn(),
                               path = "/scans/sub-01_task-nback_run-01",
                               n_parcels = 100L,
                               n_time    = 200L,
                               metadata  = list(subject = "01", task = "nback", tr = 2.0)) {
  bids_h5_scan_backend(conn, path, n_parcels, n_time, metadata)
}

# ==============================================================================
# h5_shared_connection
# ==============================================================================

test_that("h5_shared_connection requires hdf5r", {
  skip_if_not_installed("hdf5r")
  # If hdf5r is installed, opening a non-existent file should error clearly
  expect_error(
    h5_shared_connection("does_not_exist.h5"),
    "not found"
  )
})

test_that("h5_shared_connection is class h5_shared_connection", {
  conn <- make_mock_conn()
  expect_s3_class(conn, "h5_shared_connection")
})

test_that("mock connection acquire/release tracks ref_count", {
  conn <- make_mock_conn()
  expect_equal(conn$ref_count, 0L)
  conn$acquire()
  conn$acquire()
  expect_equal(conn$ref_count, 2L)
  conn$release()
  expect_equal(conn$ref_count, 1L)
  conn$release()
  expect_equal(conn$ref_count, 0L)
})

# ==============================================================================
# bids_h5_scan_backend constructor
# ==============================================================================

test_that("bids_h5_scan_backend has correct classes", {
  b <- make_scan_backend()
  expect_s3_class(b, "bids_h5_scan_backend")
  expect_s3_class(b, "storage_backend")
})

test_that("bids_h5_scan_backend stores fields correctly", {
  conn <- make_mock_conn()
  b <- bids_h5_scan_backend(
    conn, "/scans/sub-02_task-rest_run-01",
    n_parcels = 50L, n_time = 150L,
    metadata = list(subject = "02", task = "rest", tr = 1.5)
  )
  expect_equal(b$n_parcels, 50L)
  expect_equal(b$n_time,    150L)
  expect_equal(b$scan_group_path, "/scans/sub-02_task-rest_run-01")
  expect_false(b$is_open)
  expect_equal(b$metadata$subject, "02")
  expect_equal(b$metadata$tr, 1.5)
})

test_that("bids_h5_scan_backend rejects invalid connection", {
  not_a_conn <- list(file = "x.h5")
  expect_error(
    bids_h5_scan_backend(not_a_conn, "/scans/x", 10L, 20L),
    "h5_shared_connection"
  )
})

test_that("bids_h5_scan_backend rejects n_parcels < 1", {
  conn <- make_mock_conn()
  expect_error(
    bids_h5_scan_backend(conn, "/scans/x", 0L, 20L),
    "n_parcels"
  )
})

test_that("bids_h5_scan_backend rejects n_time < 1", {
  conn <- make_mock_conn()
  expect_error(
    bids_h5_scan_backend(conn, "/scans/x", 10L, 0L),
    "n_time"
  )
})

# ==============================================================================
# backend_get_dims
# ==============================================================================

test_that("backend_get_dims returns parcellated feature-space dims", {
  b    <- make_scan_backend(n_parcels = 80L, n_time = 300L)
  dims <- backend_get_dims(b)
  expect_named(dims, c("spatial", "time"))
  expect_equal(dims$spatial, c(80L, 1L, 1L))
  expect_equal(dims$time,    300L)
})

test_that("backend_get_dims spatial satisfies prod == K", {
  b    <- make_scan_backend(n_parcels = 120L, n_time = 100L)
  dims <- backend_get_dims(b)
  expect_equal(prod(dims$spatial), 120L)
})

# ==============================================================================
# backend_get_mask
# ==============================================================================

test_that("backend_get_mask returns all-TRUE logical vector of length K", {
  b    <- make_scan_backend(n_parcels = 60L)
  mask <- backend_get_mask(b)
  expect_type(mask, "logical")
  expect_length(mask, 60L)
  expect_true(all(mask))
})

test_that("mask length matches prod(spatial dims)", {
  b    <- make_scan_backend(n_parcels = 45L)
  dims <- backend_get_dims(b)
  mask <- backend_get_mask(b)
  expect_equal(length(mask), prod(dims$spatial))
})

# ==============================================================================
# validate_backend
# ==============================================================================

test_that("validate_backend passes on opened bids_h5_scan_backend", {
  b <- make_scan_backend()
  b <- backend_open(b)
  on.exit(backend_close(b))
  expect_true(validate_backend(b))
})

test_that("validate_backend passes without opening (dims/mask don't require open state)", {
  b <- make_scan_backend()
  # validate_backend calls backend_get_dims and backend_get_mask directly;
  # neither requires the file handle in parcel mode
  expect_true(validate_backend(b))
})

# ==============================================================================
# backend_open / backend_close (idempotency)
# ==============================================================================

test_that("backend_open sets is_open and increments ref_count", {
  conn <- make_mock_conn()
  b    <- make_scan_backend(conn = conn)
  b    <- backend_open(b)
  expect_true(b$is_open)
  expect_equal(conn$ref_count, 1L)
})

test_that("repeated backend_open is idempotent (ref_count stays at 1)", {
  conn <- make_mock_conn()
  b    <- make_scan_backend(conn = conn)
  b    <- backend_open(b)
  b    <- backend_open(b)   # should be a no-op
  expect_equal(conn$ref_count, 1L)
})

test_that("backend_close clears is_open and decrements ref_count", {
  conn <- make_mock_conn()
  b    <- make_scan_backend(conn = conn)
  b    <- backend_open(b)
  backend_close(b)
  expect_false(b$is_open)
  expect_equal(conn$ref_count, 0L)
})

test_that("repeated backend_close is idempotent (ref_count stays at 0)", {
  conn <- make_mock_conn()
  b    <- make_scan_backend(conn = conn)
  b    <- backend_open(b)
  backend_close(b)
  backend_close(b)  # should be a no-op
  expect_equal(conn$ref_count, 0L)
})

test_that("multiple backends share ref_count correctly", {
  conn <- make_mock_conn()
  b1   <- make_scan_backend(conn = conn, path = "/scans/scan1", n_parcels = 50L, n_time = 100L)
  b2   <- make_scan_backend(conn = conn, path = "/scans/scan2", n_parcels = 50L, n_time = 120L)

  b1 <- backend_open(b1)
  b2 <- backend_open(b2)
  expect_equal(conn$ref_count, 2L)

  backend_close(b1)
  expect_equal(conn$ref_count, 1L)

  backend_close(b2)
  expect_equal(conn$ref_count, 0L)
})

# ==============================================================================
# backend_get_metadata
# ==============================================================================

test_that("backend_get_metadata returns compression_mode and scan fields", {
  meta <- list(subject = "01", task = "nback", session = "pre", run = "01", tr = 2.0)
  b    <- make_scan_backend(metadata = meta)
  m    <- backend_get_metadata(b)

  expect_equal(m$compression_mode, "parcellated")
  expect_equal(m$n_parcels,        100L)
  expect_equal(m$subject,          "01")
  expect_equal(m$task,             "nback")
  expect_equal(m$tr,               2.0)
  expect_equal(m$format,           "bids_h5")
})

# ==============================================================================
# backend_get_data (requires real HDF5 — test with hdf5r if available)
# ==============================================================================

test_that("backend_get_data reads [T, K] matrix from HDF5", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  # Write a minimal test HDF5 file
  h5f <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)

  K <- 30L; T <- 50L
  expected <- matrix(rnorm(T * K), nrow = T, ncol = K)

  scans_grp <- h5f$create_group("scans")
  scan_grp  <- scans_grp$create_group("sub-01_task-test_run-01")
  data_grp  <- scan_grp$create_group("data")
  data_grp$create_dataset("summary_data", robj = expected)
  h5f$close_all()

  # Now read via backend
  conn <- h5_shared_connection(tmp)
  b    <- bids_h5_scan_backend(conn, "/scans/sub-01_task-test_run-01", K, T)
  conn$acquire()  # simulate backend_open without environment mutation

  mat <- backend_get_data(b)
  conn$release()

  expect_equal(dim(mat), c(T, K))
  expect_equal(mat, expected, tolerance = 1e-6)
})

test_that("backend_get_data respects rows and cols subsetting", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  K <- 20L; T <- 40L
  full_mat <- matrix(seq_len(T * K), nrow = T, ncol = K)

  h5f <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp  <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")
  scan_grp$create_group("data")$create_dataset("summary_data", robj = full_mat)
  h5f$close_all()

  conn <- h5_shared_connection(tmp)
  b    <- bids_h5_scan_backend(conn, "/scans/sub-01_task-test_run-01", K, T)
  conn$acquire()

  row_idx <- c(1L, 3L, 5L)
  col_idx <- c(2L, 4L)
  sub <- backend_get_data(b, rows = row_idx, cols = col_idx)
  conn$release()

  expect_equal(dim(sub), c(3L, 2L))
  expect_equal(sub, full_mat[row_idx, col_idx, drop = FALSE], tolerance = 1e-6)
})

# ==============================================================================
# Event helpers (h5_write_events / h5_read_events)
# ==============================================================================

test_that("h5_write_events and h5_read_events round-trip correctly", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  events <- data.frame(
    onset      = c(0.0, 10.0, 20.0),
    duration   = c(2.0, 2.0, 2.0),
    trial_type = c("face", "house", "face"),
    stringsAsFactors = FALSE
  )

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  h5_write_events(scan_grp, events)
  result <- h5_read_events(scan_grp)
  h5f$close_all()

  expect_equal(nrow(result), 3L)
  expect_equal(result$onset,      events$onset)
  expect_equal(result$duration,   events$duration)
  expect_equal(result$trial_type, events$trial_type)
})

test_that("h5_read_events returns NULL when no events group exists", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  result <- h5_read_events(scan_grp)
  h5f$close_all()

  expect_null(result)
})

test_that("h5_write_events handles NULL events gracefully", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  expect_invisible(h5_write_events(scan_grp, NULL))
  h5f$close_all()
})

# ==============================================================================
# Confound helpers
# ==============================================================================

test_that("h5_write_confounds and h5_read_confounds round-trip correctly", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  confounds <- data.frame(
    motion_x = rnorm(50),
    motion_y = rnorm(50),
    csf      = rnorm(50)
  )

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  h5_write_confounds(scan_grp, confounds)
  result <- h5_read_confounds(scan_grp)
  h5f$close_all()

  expect_equal(nrow(result), 50L)
  expect_equal(ncol(result), 3L)
  expect_equal(names(result), c("motion_x", "motion_y", "csf"))
  expect_equal(result$motion_x, confounds$motion_x, tolerance = 1e-10)
})

test_that("h5_read_confounds returns NULL when absent", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  expect_null(h5_read_confounds(scan_grp))
  h5f$close_all()
})

# ==============================================================================
# Censor helpers
# ==============================================================================

test_that("h5_write_censor and h5_read_censor round-trip correctly", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  censor <- c(0L, 0L, 1L, 0L, 1L, 0L, 0L, 0L, 1L, 0L)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  h5_write_censor(scan_grp, censor)
  result <- h5_read_censor(scan_grp)
  h5f$close_all()

  expect_equal(result, censor)
})

test_that("h5_write_censor accepts logical vectors", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  censor_logical <- c(FALSE, FALSE, TRUE, FALSE, TRUE)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  h5_write_censor(scan_grp, censor_logical)
  result <- h5_read_censor(scan_grp)
  h5f$close_all()

  expect_equal(result, as.integer(censor_logical))
})

test_that("h5_read_censor returns NULL when absent", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-test_run-01")

  expect_null(h5_read_censor(scan_grp))
  h5f$close_all()
})

# ==============================================================================
# Scan metadata helpers
# ==============================================================================

test_that("h5_write_scan_metadata and h5_read_scan_metadata round-trip", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  meta <- list(subject = "01", task = "nback", run = "01", tr = 2.0)

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-nback_run-01")

  h5_write_scan_metadata(scan_grp, meta)
  result <- h5_read_scan_metadata(scan_grp)
  h5f$close_all()

  expect_equal(result$subject, "01")
  expect_equal(result$task,    "nback")
  expect_equal(result$tr,      2.0)
})

test_that("h5_write_scan_metadata skips NULL values", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  meta <- list(subject = "01", session = NULL, task = "rest")

  h5f      <- hdf5r::H5File$new(tmp, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)
  scan_grp <- h5f$create_group("scans")$create_group("sub-01_task-rest_run-01")

  expect_invisible(h5_write_scan_metadata(scan_grp, meta))
  result <- h5_read_scan_metadata(scan_grp)
  h5f$close_all()

  # session should not be stored (was NULL)
  expect_false("session" %in% names(result))
  expect_equal(result$subject, "01")
})

# ==============================================================================
# print methods
# ==============================================================================

test_that("print.bids_h5_scan_backend produces output", {
  b <- make_scan_backend()
  expect_output(print(b), "bids_h5_scan_backend")
  expect_output(print(b), "100")
  expect_output(print(b), "200")
})

test_that("print.h5_shared_connection produces output", {
  conn <- make_mock_conn()
  expect_output(print(conn), "h5_shared_connection")
  expect_output(print(conn), "fake_study.h5")
})
