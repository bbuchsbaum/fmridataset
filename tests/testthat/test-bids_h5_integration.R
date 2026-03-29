# Integration tests for the BIDS H5 reader, accessor, and subset API.
#
# These tests build minimal HDF5 files directly (without needing bidser or
# fmristore) so the round-trip reader path can be exercised in CI with only
# hdf5r installed.  Tests requiring the full compress_bids_study() writer
# are tagged with skip_if_not_installed("bidser") / skip_if_not_installed("fmristore").

skip_if_not_installed("hdf5r")

# ============================================================
# Helpers: build a minimal valid BIDS H5 file
# ============================================================

#' Write a minimal BIDS H5 archive for testing
#'
#' @param file   Output path.
#' @param scans  Named list of scan specs:
#'   list(scan_name = list(subject, task, run, session, n_time, data_matrix))
#' @param n_parcels Integer. K parcels (column count of data matrices).
#' @param tr     Numeric. TR in seconds.
#' @param cluster_ids Integer vector of parcel IDs (length K).
make_test_h5 <- function(file, scans, n_parcels, tr = 2.0,
                          cluster_ids = seq_len(n_parcels)) {
  h5 <- hdf5r::H5File$new(file, mode = "w")
  on.exit(tryCatch(h5$close_all(), error = function(e) NULL), add = TRUE)

  # Root attributes
  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "parcellated")
  h5$create_attr("writer_version",   "test")

  # /parcellation/
  parc_grp <- h5$create_group("parcellation")
  parc_grp$create_dataset("cluster_ids", robj = as.integer(cluster_ids))
  # Minimal cluster_map (just K integers)
  parc_grp$create_dataset("cluster_map", robj = as.integer(cluster_ids))

  # /bids/
  bids_grp <- h5$create_group("bids")
  unique_tasks    <- unique(vapply(scans, `[[`, character(1), "task"))
  unique_sessions <- unique(unlist(lapply(scans, function(s) s$session %||% "")))
  unique_sessions <- unique_sessions[nzchar(unique_sessions)]
  bids_grp$create_dataset("tasks",    robj = unique_tasks)
  bids_grp$create_dataset("name",     robj = "test_study")
  bids_grp$create_dataset("space",    robj = "MNI152NLin2009cAsym")
  bids_grp$create_dataset("pipeline", robj = "fmriprep")
  if (length(unique_sessions) > 0) {
    bids_grp$create_dataset("sessions", robj = unique_sessions)
  }

  # /spatial/ (minimal)
  spat_grp <- h5$create_group("spatial")
  hdr_grp  <- spat_grp$create_group("header")
  hdr_grp$create_dataset("dim", robj = c(n_parcels, 1L, 1L))
  hdr_grp$create_dataset("pixdim", robj = c(1.0, 1.0, 1.0))

  # /scans/
  scans_grp <- h5$create_group("scans")
  scan_names_vec   <- character(length(scans))
  n_time_vec       <- integer(length(scans))
  subject_vec      <- character(length(scans))
  task_vec         <- character(length(scans))
  run_vec          <- character(length(scans))
  session_vec      <- character(length(scans))
  has_events_vec   <- integer(length(scans))
  has_confounds_vec <- integer(length(scans))

  for (i in seq_along(scans)) {
    spec      <- scans[[i]]
    sname     <- names(scans)[[i]]
    scan_names_vec[[i]] <- sname
    n_time              <- spec$n_time
    n_time_vec[[i]]     <- n_time
    subject_vec[[i]]    <- spec$subject
    task_vec[[i]]       <- spec$task
    run_vec[[i]]        <- spec$run %||% "01"
    session_vec[[i]]    <- spec$session %||% ""

    sg <- scans_grp$create_group(sname)
    dg <- sg$create_group("data")

    # Data matrix [T, K]
    mat <- if (!is.null(spec$data_matrix)) {
      spec$data_matrix
    } else {
      matrix(rnorm(n_time * n_parcels), nrow = n_time, ncol = n_parcels)
    }
    dg$create_dataset("summary_data", robj = mat)

    # Metadata
    md_grp <- sg$create_group("metadata")
    md_grp$create_dataset("subject", robj = spec$subject)
    md_grp$create_dataset("task",    robj = spec$task)
    md_grp$create_dataset("run",     robj = run_vec[[i]])
    md_grp$create_dataset("tr",      robj = tr)
    if (nzchar(session_vec[[i]])) {
      md_grp$create_dataset("session", robj = session_vec[[i]])
    }

    # Events
    if (!is.null(spec$events)) {
      ev <- spec$events
      ev_grp <- sg$create_group("events")
      ev_grp$create_attr("n_events", nrow(ev))
      for (cn in names(ev)) {
        vals <- ev[[cn]]
        if (is.factor(vals)) vals <- as.character(vals)
        if (!is.character(vals)) vals <- as.double(vals)
        ev_grp$create_dataset(cn, robj = vals)
      }
      has_events_vec[[i]] <- 1L
    }

    # Confounds
    if (!is.null(spec$confounds)) {
      cf_grp <- sg$create_group("confounds")
      mat_cf <- as.matrix(spec$confounds)
      storage.mode(mat_cf) <- "double"
      ds_cf <- cf_grp$create_dataset("data", robj = mat_cf)
      ds_cf$create_attr("names", names(spec$confounds))
      has_confounds_vec[[i]] <- 1L
    }

    # Censor
    h5_write_censor(sg, rep(0L, n_time))
  }

  # /scan_index/
  si <- h5$create_group("scan_index")
  time_offset <- c(0L, cumsum(n_time_vec[-length(n_time_vec)]))
  si$create_dataset("scan_name",      robj = scan_names_vec)
  si$create_dataset("subject",        robj = subject_vec)
  si$create_dataset("task",           robj = task_vec)
  si$create_dataset("run",            robj = run_vec)
  si$create_dataset("session",        robj = session_vec)
  si$create_dataset("n_time",         robj = n_time_vec)
  si$create_dataset("time_offset",    robj = time_offset)
  si$create_dataset("has_events",     robj = has_events_vec)
  si$create_dataset("has_confounds",  robj = has_confounds_vec)

  h5$close_all()
  file
}

# Default test scans used by most tests
make_default_scans <- function(K = 20L, T1 = 50L, T2 = 60L) {
  events1 <- data.frame(
    onset      = c(0, 10, 20),
    duration   = c(2, 2, 2),
    trial_type = c("face", "house", "face"),
    stringsAsFactors = FALSE
  )
  events2 <- data.frame(
    onset      = c(0, 15),
    duration   = c(2, 2),
    trial_type = c("house", "house"),
    stringsAsFactors = FALSE
  )
  confounds1 <- data.frame(
    motion_x = rnorm(T1),
    motion_y = rnorm(T1)
  )

  list(
    "sub-01_task-nback_run-01" = list(
      subject = "01", task = "nback", run = "01", n_time = T1,
      events = events1, confounds = confounds1
    ),
    "sub-01_task-nback_run-02" = list(
      subject = "01", task = "nback", run = "02", n_time = T2,
      events = events2
    ),
    "sub-02_task-nback_run-01" = list(
      subject = "02", task = "nback", run = "01", n_time = T1,
      events = events1
    ),
    "sub-02_task-rest_run-01" = list(
      subject = "02", task = "rest", run = "01", n_time = T2
    )
  )
}


# ============================================================
# bids_h5_dataset() — reader
# ============================================================

test_that("bids_h5_dataset returns bids_h5_study_dataset", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)

  study <- bids_h5_dataset(tmp)
  expect_s3_class(study, "bids_h5_study_dataset")
  expect_s3_class(study, "fmri_study_dataset")
  expect_s3_class(study, "fmri_dataset")
})

test_that("bids_h5_dataset rejects missing file", {
  expect_error(bids_h5_dataset("does_not_exist.h5"), regexp = "does_not_exist")
})

test_that("bids_h5_dataset rejects file with wrong format attribute", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  h5 <- hdf5r::H5File$new(tmp, mode = "w")
  h5$create_attr("format", "something_else")
  h5$close_all()
  expect_error(bids_h5_dataset(tmp), regexp = "format")
})

test_that("bids_h5_dataset validates compression_mode", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  h5 <- hdf5r::H5File$new(tmp, mode = "w")
  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "lna")   # not yet supported
  # Need minimal scan_index and parcellation to get past earlier checks
  h5$create_group("parcellation")$create_dataset("cluster_ids", robj = 1L)
  si <- h5$create_group("scan_index")
  si$create_dataset("scan_name",   robj = "sub-01_task-test_run-01")
  si$create_dataset("subject",     robj = "01")
  si$create_dataset("task",        robj = "test")
  si$create_dataset("run",         robj = "01")
  si$create_dataset("session",     robj = "")
  si$create_dataset("n_time",      robj = 10L)
  si$create_dataset("time_offset", robj = 0L)
  si$create_dataset("has_events",     robj = 0L)
  si$create_dataset("has_confounds",  robj = 0L)
  h5$close_all()
  expect_error(bids_h5_dataset(tmp), regexp = "parcellated|lna")
})


# ============================================================
# scan_manifest
# ============================================================

test_that("scan_manifest returns tibble with expected columns", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  m <- scan_manifest(study)
  expect_s3_class(m, "data.frame")
  expected_cols <- c("scan_name", "subject", "task", "session", "run",
                     "n_time", "time_offset", "has_events", "has_confounds")
  expect_true(all(expected_cols %in% names(m)))
  expect_equal(nrow(m), 4L)
})

test_that("scan_manifest n_time values match what was written", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(T1 = 50L, T2 = 60L), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  m <- scan_manifest(study)
  n_times <- sort(unique(m$n_time))
  expect_equal(n_times, c(50L, 60L))
})


# ============================================================
# participants / tasks / sessions
# ============================================================

test_that("participants returns correct subject IDs", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  p <- participants(study)
  expect_equal(sort(p), c("01", "02"))
})

test_that("tasks returns correct task names", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  t <- tasks(study)
  expect_equal(sort(t), c("nback", "rest"))
})

test_that("sessions returns NULL when no sessions", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  expect_null(sessions(study))
})

test_that("sessions returns session names when present", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  scans_with_session <- list(
    "sub-01_ses-pre_task-nback_run-01" = list(
      subject = "01", task = "nback", run = "01", session = "pre", n_time = 40L
    ),
    "sub-01_ses-post_task-nback_run-01" = list(
      subject = "01", task = "nback", run = "01", session = "post", n_time = 40L
    )
  )

  make_test_h5(tmp, scans_with_session, n_parcels = 15L)
  study <- bids_h5_dataset(tmp)

  sess <- sessions(study)
  expect_equal(sort(sess), c("post", "pre"))
})


# ============================================================
# parcellation_info
# ============================================================

test_that("parcellation_info returns list with n_parcels", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 25L
  make_test_h5(tmp, make_default_scans(), n_parcels = K)
  study <- bids_h5_dataset(tmp)

  pinfo <- parcellation_info(study)
  expect_type(pinfo, "list")
  expect_equal(pinfo$n_parcels, K)
  expect_length(pinfo$cluster_ids, K)
})


# ============================================================
# get_data_matrix — [T, K] shape
# ============================================================

test_that("get_data_matrix returns [T_total, K] matrix", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 20L; T1 <- 30L; T2 <- 40L
  scans <- make_default_scans(K = K, T1 = T1, T2 = T2)
  make_test_h5(tmp, scans, n_parcels = K)
  study <- bids_h5_dataset(tmp)

  # get_data_matrix may return a delarr (lazy); materialize for inspection
  mat <- as.matrix(get_data_matrix(study))
  # Total timepoints: sub-01 has T1+T2=70, sub-02 has T1+T2=70 → 140
  expect_equal(ncol(mat), K)
  expect_equal(nrow(mat), T1 + T2 + T1 + T2)
})

test_that("get_data_matrix returns correct K columns matching n_parcels", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 33L
  scans <- list(
    "sub-01_task-rest_run-01" = list(
      subject = "01", task = "rest", run = "01", n_time = 50L
    )
  )
  make_test_h5(tmp, scans, n_parcels = K)
  study <- bids_h5_dataset(tmp)

  mat <- as.matrix(get_data_matrix(study))
  expect_equal(ncol(mat), K)
  expect_equal(nrow(mat), 50L)
})

test_that("get_data_matrix values round-trip correctly", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 10L; T <- 20L

  known_mat <- matrix(as.double(seq_len(T * K)), nrow = T, ncol = K)
  scans <- list(
    "sub-01_task-test_run-01" = list(
      subject = "01", task = "test", run = "01", n_time = T,
      data_matrix = known_mat
    )
  )
  make_test_h5(tmp, scans, n_parcels = K)
  study <- bids_h5_dataset(tmp)

  mat <- as.matrix(get_data_matrix(study))
  expect_equal(dim(mat), c(T, K))
  expect_equal(mat, known_mat, tolerance = 1e-6)
})


# ============================================================
# event_table
# ============================================================

test_that("event_table is populated from stored events", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  et <- study$event_table
  expect_s3_class(et, "data.frame")
  expect_gt(nrow(et), 0L)
  # Check that onset, duration, trial_type are present
  expect_true("onset"      %in% names(et))
  expect_true("duration"   %in% names(et))
  expect_true("trial_type" %in% names(et))
})

test_that("event_table has task and run columns added by reader", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  et <- study$event_table
  expect_true("task"       %in% names(et))
  expect_true("run"        %in% names(et))
  expect_true("subject_id" %in% names(et))
})


# ============================================================
# get_confounds
# ============================================================

test_that("get_confounds returns tibble for single scan with confounds", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  cf <- get_confounds(study, scan_name = "sub-01_task-nback_run-01")
  expect_s3_class(cf, "data.frame")
  expect_equal(ncol(cf), 2L)    # motion_x, motion_y
  expect_equal(nrow(cf), 50L)   # T1 = 50
  expect_true("motion_x" %in% names(cf))
})

test_that("get_confounds returns NULL for scan with no confounds", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  # sub-02_task-rest_run-01 has no confounds
  cf <- get_confounds(study, scan_name = "sub-02_task-rest_run-01")
  expect_null(cf)
})

test_that("get_confounds by subject returns named list", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  # sub-01 has confounds only on run-01
  cf <- get_confounds(study, subject = "01")
  expect_type(cf, "list")
  expect_named(cf)
})


# ============================================================
# subset_bids_h5
# ============================================================

test_that("subset_bids_h5 by task returns only matching scans", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  nback <- subset_bids_h5(study, task = "nback")
  expect_s3_class(nback, "bids_h5_study_dataset")
  m <- scan_manifest(nback)
  expect_true(all(m$task == "nback"))
  expect_equal(nrow(m), 3L)  # 2 runs for sub-01 + 1 run for sub-02
})

test_that("subset_bids_h5 by subject returns only matching scans", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  sub01 <- subset_bids_h5(study, subject = "01")
  expect_s3_class(sub01, "bids_h5_study_dataset")
  m <- scan_manifest(sub01)
  expect_true(all(m$subject == "01"))
})

test_that("subset_bids_h5 by task + subject returns correct intersection", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  sub <- subset_bids_h5(study, task = "nback", subject = "02")
  m <- scan_manifest(sub)
  expect_equal(nrow(m), 1L)
  expect_equal(m$subject, "02")
  expect_equal(m$task, "nback")
})

test_that("subset_bids_h5 errors when no scans match", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  expect_error(subset_bids_h5(study, task = "nonexistent"), regexp = "no scans")
})

test_that("subset shares the same H5 connection (ref-count stable)", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  expect_true(study$h5_connection$handle$is_valid)
  sub <- subset_bids_h5(study, task = "nback")
  # Same underlying handle
  expect_true(identical(study$h5_connection, sub$h5_connection))
  expect_true(sub$h5_connection$handle$is_valid)
})

test_that("subset_bids_h5 result supports get_data_matrix", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 15L; T1 <- 30L; T2 <- 45L
  make_test_h5(tmp, make_default_scans(K = K, T1 = T1, T2 = T2), n_parcels = K)
  study <- bids_h5_dataset(tmp)

  # sub-02 nback only (1 scan, T1 timepoints)
  sub <- subset_bids_h5(study, task = "nback", subject = "02")
  mat <- as.matrix(get_data_matrix(sub))
  expect_equal(ncol(mat), K)
  expect_equal(nrow(mat), T1)
})


# ============================================================
# data_chunks
# ============================================================

test_that("data_chunks iterates over study without error", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 20L
  make_test_h5(tmp, make_default_scans(K = K), n_parcels = K)
  study <- bids_h5_dataset(tmp)

  chunks <- data_chunks(study, nchunks = 2L)
  expect_false(is.null(chunks))

  # Collect all chunks
  collected <- list()
  while (TRUE) {
    chunk <- tryCatch(iterators::nextElem(chunks), error = function(e) NULL)
    if (is.null(chunk)) break
    collected <- c(collected, list(chunk))
  }
  expect_gt(length(collected), 0L)
})


# ============================================================
# get_TR
# ============================================================

test_that("get_TR returns correct TR value", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L, tr = 1.5)
  study <- bids_h5_dataset(tmp)

  expect_equal(get_TR(study), 1.5)
})


# ============================================================
# n_runs
# ============================================================

test_that("n_runs returns total number of runs across study", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  # Total: sub-01 has 2 nback runs, sub-02 has 1 nback + 1 rest = 4 scans total
  expect_equal(n_runs(study), 4L)
})


# ============================================================
# print method
# ============================================================

test_that("print.bids_h5_study_dataset produces output", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  expect_output(print(study), "bids_h5_study_dataset")
  expect_output(print(study), "parcellated")
  expect_output(print(study), "20")   # n_parcels
})


# ============================================================
# subject_ids (inherited from fmri_study_dataset)
# ============================================================

test_that("subject_ids returns all subjects", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  sids <- subject_ids(study)
  expect_equal(sort(sids), c("01", "02"))
})


# ============================================================
# study_to_group
# ============================================================

test_that("study_to_group returns fmri_group with correct subjects", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  make_test_h5(tmp, make_default_scans(), n_parcels = 20L)
  study <- bids_h5_dataset(tmp)

  grp <- study_to_group(study)
  expect_s3_class(grp, "fmri_group")
  expect_equal(nrow(grp$subjects), 2L)
})


# ============================================================
# validate_backend on scan backends inside study
# ============================================================

test_that("all scan backends pass validate_backend", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)
  K <- 20L
  make_test_h5(tmp, make_default_scans(K = K), n_parcels = K)
  study <- bids_h5_dataset(tmp)

  for (sn in names(study$.scan_backends)) {
    b <- study$.scan_backends[[sn]]
    expect_true(validate_backend(b), label = paste("validate_backend:", sn))
  }
})
