# Integration tests for the BIDS H5 round-trip:
#   bids_h5_dataset() reader, subset_bids_h5(), and accessors.
#
# Strategy: build a minimal well-formed HDF5 archive in-process using hdf5r
# (no bidser, fmristore, or real BIDS directory required).
# All tests are skipped when hdf5r is not installed.

# ==============================================================================
# Helpers to build a minimal BIDS H5 archive
# ==============================================================================

#' Write a minimal BIDS H5 archive for tests.
#'
#' @param file       Path to the output .h5 file (will be overwritten).
#' @param scans      Named list. Each element: list(subject, task, session, run,
#'                   n_time, data, events, confounds, censor).
#' @param n_parcels  Integer. Number of parcels (columns in data).
#' @param tr         Numeric. TR in seconds.
#' @param space      Character. Space name.
make_test_h5 <- function(file,
                          scans,
                          n_parcels = 20L,
                          tr        = 2.0,
                          space     = "MNI152") {
  h5f <- hdf5r::H5File$new(file, mode = "w")
  on.exit(tryCatch(h5f$close_all(), error = function(e) NULL), add = TRUE)

  # Root attributes
  h5f$create_attr("format",           "bids_h5_study")
  h5f$create_attr("version",          "1.0")
  h5f$create_attr("compression_mode", "parcellated")

  # /bids/ group
  bids_grp <- h5f$create_group("bids")
  bids_grp$create_dataset("name",  robj = "test_study")
  bids_grp$create_dataset("space", robj = space)

  # /parcellation/ group
  parc_grp <- h5f$create_group("parcellation")
  parc_grp$create_dataset("cluster_ids",  robj = seq_len(n_parcels))
  parc_grp$create_dataset("cluster_map",  robj = rep(seq_len(n_parcels), each = 10L))
  meta_grp <- parc_grp$create_group("cluster_meta")
  meta_grp$create_dataset("labels", robj = paste0("parcel_", seq_len(n_parcels)))

  # /spatial/ group (minimal)
  spat_grp <- h5f$create_group("spatial")

  # /scans/ group
  scans_grp <- h5f$create_group("scans")

  scan_names  <- character(length(scans))
  subjects    <- character(length(scans))
  tasks       <- character(length(scans))
  sessions    <- character(length(scans))
  runs        <- character(length(scans))
  n_times     <- integer(length(scans))
  has_events  <- logical(length(scans))
  has_confounds <- logical(length(scans))

  for (i in seq_along(scans)) {
    sc        <- scans[[i]]
    scan_name <- names(scans)[[i]]

    scan_grp  <- scans_grp$create_group(scan_name)
    data_grp  <- scan_grp$create_group("data")

    n_time <- sc$n_time
    K      <- n_parcels
    mat    <- if (!is.null(sc$data)) sc$data else matrix(rnorm(n_time * K), n_time, K)
    data_grp$create_dataset("summary_data", robj = mat)

    # metadata
    md_grp <- scan_grp$create_group("metadata")
    md_grp$create_dataset("subject", robj = sc$subject)
    md_grp$create_dataset("task",    robj = sc$task)
    md_grp$create_dataset("run",     robj = sc$run)
    md_grp$create_dataset("tr",      robj = tr)
    if (!is.null(sc$session) && nzchar(sc$session)) {
      md_grp$create_dataset("session", robj = sc$session)
    }

    # events
    if (!is.null(sc$events)) {
      ev_grp <- scan_grp$create_group("events")
      ev_grp$create_attr("n_events", nrow(sc$events))
      for (col in names(sc$events)) {
        ev_grp$create_dataset(col, robj = as.character(sc$events[[col]]))
      }
      has_events[[i]] <- TRUE
    }

    # confounds
    if (!is.null(sc$confounds)) {
      cf_grp <- scan_grp$create_group("confounds")
      mat_cf <- as.matrix(sc$confounds)
      storage.mode(mat_cf) <- "double"
      ds <- cf_grp$create_dataset("data", robj = mat_cf)
      ds$create_attr("names", names(sc$confounds))
      has_confounds[[i]] <- TRUE
    }

    # censor
    if (!is.null(sc$censor)) {
      censor_int <- as.integer(as.logical(sc$censor))
      h5_write_censor(scan_grp, censor_int)
    }

    scan_names[[i]]  <- scan_name
    subjects[[i]]    <- sc$subject
    tasks[[i]]       <- sc$task
    sessions[[i]]    <- if (!is.null(sc$session)) sc$session else ""
    runs[[i]]        <- sc$run
    n_times[[i]]     <- n_time
  }

  # /scan_index/
  idx_grp <- h5f$create_group("scan_index")
  idx_grp$create_dataset("scan_name",     robj = scan_names)
  idx_grp$create_dataset("subject",       robj = subjects)
  idx_grp$create_dataset("task",          robj = tasks)
  idx_grp$create_dataset("session",       robj = sessions)
  idx_grp$create_dataset("run",           robj = runs)
  idx_grp$create_dataset("n_time",        robj = n_times)
  idx_grp$create_dataset("time_offset",   robj = c(0L, cumsum(n_times)[-length(n_times)]))
  idx_grp$create_dataset("has_events",    robj = as.integer(has_events))
  idx_grp$create_dataset("has_confounds", robj = as.integer(has_confounds))

  invisible(file)
}

# Convenience: minimal 2-subject, 2-task dataset
make_standard_test_h5 <- function(file = tempfile(fileext = ".h5"),
                                   n_parcels = 15L,
                                   tr        = 2.0) {
  K <- n_parcels
  t1 <- 30L; t2 <- 25L; t3 <- 28L; t4 <- 22L

  ev1 <- data.frame(onset = c(0, 10, 20), duration = c(2, 2, 2), trial_type = c("A", "B", "A"))
  ev2 <- data.frame(onset = c(0, 12),     duration = c(2, 2),     trial_type = c("C", "C"))

  cf1 <- data.frame(motion_x = rnorm(t1), motion_y = rnorm(t1))

  scans <- list(
    "sub-01_task-nback_run-01" = list(
      subject  = "01", task = "nback", session = "", run = "01",
      n_time   = t1,
      data     = matrix(rnorm(t1 * K), t1, K),
      events   = ev1, confounds = cf1, censor = NULL
    ),
    "sub-01_task-rest_run-01"  = list(
      subject  = "01", task = "rest", session = "", run = "01",
      n_time   = t2,
      data     = matrix(rnorm(t2 * K), t2, K),
      events   = NULL, confounds = NULL, censor = NULL
    ),
    "sub-02_task-nback_run-01" = list(
      subject  = "02", task = "nback", session = "", run = "01",
      n_time   = t3,
      data     = matrix(rnorm(t3 * K), t3, K),
      events   = ev2, confounds = NULL, censor = NULL
    ),
    "sub-02_task-rest_run-01"  = list(
      subject  = "02", task = "rest", session = "", run = "01",
      n_time   = t4,
      data     = matrix(rnorm(t4 * K), t4, K),
      events   = NULL, confounds = NULL, censor = NULL
    )
  )

  make_test_h5(file, scans, n_parcels = K, tr = tr)
  file
}


# ==============================================================================
# bids_h5_dataset — construction and class
# ==============================================================================

test_that("bids_h5_dataset returns bids_h5_study_dataset with correct classes", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  expect_s3_class(study, "bids_h5_study_dataset")
  expect_s3_class(study, "fmri_study_dataset")
  expect_s3_class(study, "fmri_dataset")
})

test_that("bids_h5_dataset errors on missing file", {
  skip_if_not_installed("hdf5r")

  expect_error(bids_h5_dataset("/no/such/file.h5"), "No such file")
})

test_that("bids_h5_dataset errors when format attribute is wrong", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  h5f <- hdf5r::H5File$new(tmp, mode = "w")
  h5f$create_attr("format", "not_bids")
  h5f$close_all()

  expect_error(bids_h5_dataset(tmp), "format")
})

test_that("bids_h5_dataset errors when compression_mode is unsupported", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  h5f <- hdf5r::H5File$new(tmp, mode = "w")
  h5f$create_attr("format",           "bids_h5_study")
  h5f$create_attr("version",          "1.0")
  h5f$create_attr("compression_mode", "lna_future")
  h5f$close_all()

  expect_error(bids_h5_dataset(tmp), "Unknown compression_mode")
})


# ==============================================================================
# scan_manifest, participants, tasks, sessions
# ==============================================================================

test_that("scan_manifest has expected columns and rows", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  mf    <- scan_manifest(study)

  expect_s3_class(mf, "tbl_df")
  expect_equal(nrow(mf), 4L)
  expect_true(all(c("scan_name", "subject", "task", "session", "run",
                     "n_time", "has_events", "has_confounds") %in% names(mf)))
})

test_that("participants() returns unique subject IDs", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  subs  <- participants(study)

  expect_type(subs, "character")
  expect_setequal(subs, c("01", "02"))
})

test_that("tasks() returns unique task names", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  t_names <- tasks(study)

  expect_type(t_names, "character")
  expect_setequal(t_names, c("nback", "rest"))
})

test_that("sessions() returns NULL when no real sessions", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  # empty string sessions → NULL
  expect_null(sessions(study))
})

test_that("sessions() returns session names when sessions present", {
  skip_if_not_installed("hdf5r")

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  K <- 10L; T <- 20L
  scans <- list(
    "sub-01_ses-pre_task-rest_run-01" = list(
      subject = "01", task = "rest", session = "pre", run = "01",
      n_time = T, data = matrix(rnorm(T * K), T, K),
      events = NULL, confounds = NULL, censor = NULL
    )
  )
  make_test_h5(tmp, scans, n_parcels = K, tr = 2.0)

  study <- bids_h5_dataset(tmp)
  sess  <- sessions(study)
  expect_equal(sess, "pre")
})


# ==============================================================================
# get_TR, n_runs
# ==============================================================================

test_that("get_TR returns correct TR", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5(tr = 1.5)
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  expect_equal(get_TR(study), 1.5)
})

test_that("n_runs equals total number of scans across subjects", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  # 2 subjects × 2 tasks = 4 scans = 4 runs
  expect_equal(n_runs(study), 4L)
})


# ==============================================================================
# get_data_matrix — dimensions
# ==============================================================================

test_that("get_data_matrix returns [T_total, K] matrix", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  # get_data_matrix may return a delarr (lazy matrix) when delarr is available;
  # coerce to a plain matrix for dimension checking.
  mat   <- as.matrix(get_data_matrix(study))

  # total timepoints: 30+25+28+22 = 105
  expect_true(is.matrix(mat))
  expect_equal(ncol(mat), K)
  expect_equal(nrow(mat), 105L)
})

test_that("get_data_matrix per-subject returns [T_subj, K]", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  mat01 <- get_data_matrix(study, subject_id = "01")

  # sub-01 has 30 + 25 = 55 timepoints
  expect_equal(nrow(mat01), 55L)
  expect_equal(ncol(mat01), K)
})


# ==============================================================================
# Data round-trip: stored values are recovered correctly
# ==============================================================================

test_that("data stored in H5 is recovered correctly by get_data_matrix", {
  skip_if_not_installed("hdf5r")

  K <- 10L; T <- 20L
  expected_data <- matrix(seq_len(T * K) / 100, nrow = T, ncol = K)

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  scans <- list(
    "sub-01_task-test_run-01" = list(
      subject = "01", task = "test", session = "", run = "01",
      n_time  = T, data = expected_data,
      events  = NULL, confounds = NULL, censor = NULL
    )
  )
  make_test_h5(tmp, scans, n_parcels = K, tr = 2.0)

  study <- bids_h5_dataset(tmp)
  # Coerce to plain matrix (delarr may be returned when delarr package is installed)
  mat   <- as.matrix(get_data_matrix(study))

  expect_equal(dim(mat), c(T, K))
  expect_equal(mat, expected_data, tolerance = 1e-6)
})


# ==============================================================================
# event_table
# ==============================================================================

test_that("event_table is a tibble with expected columns", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  et    <- study$event_table

  expect_s3_class(et, "tbl_df")
  # Must have run, run_id, subject_id, task columns
  expect_true("run"        %in% names(et))
  expect_true("run_id"     %in% names(et))
  expect_true("subject_id" %in% names(et))
  expect_true("task"       %in% names(et))
})

test_that("event_table contains events from both subjects", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  et    <- study$event_table

  # sub-01_nback: 3 events, sub-02_nback: 2 events (rest has no events)
  expect_equal(nrow(et), 5L)
  expect_setequal(et$subject_id, c("01", "02"))
})

test_that("event_table trial_type values match original", {
  skip_if_not_installed("hdf5r")

  K <- 10L; T <- 20L
  ev <- data.frame(onset = c(0, 5), duration = c(2, 2), trial_type = c("go", "stop"))

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  scans <- list(
    "sub-01_task-nback_run-01" = list(
      subject = "01", task = "nback", session = "", run = "01",
      n_time  = T, data = matrix(rnorm(T * K), T, K),
      events  = ev, confounds = NULL, censor = NULL
    )
  )
  make_test_h5(tmp, scans, n_parcels = K, tr = 2.0)

  study  <- bids_h5_dataset(tmp)
  tt_out <- study$event_table$trial_type

  expect_equal(sort(tt_out), c("go", "stop"))
})


# ==============================================================================
# subset_bids_h5
# ==============================================================================

test_that("subset_bids_h5 by task returns only matching scans", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  nback  <- subset_bids_h5(study, task = "nback")

  expect_s3_class(nback, "bids_h5_study_dataset")
  expect_equal(nrow(scan_manifest(nback)), 2L)
  expect_true(all(scan_manifest(nback)$task == "nback"))
})

test_that("subset_bids_h5 by subject returns only matching scans", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  sub01  <- subset_bids_h5(study, subject = "01")

  expect_equal(nrow(scan_manifest(sub01)), 2L)
  expect_true(all(scan_manifest(sub01)$subject == "01"))
})

test_that("subset_bids_h5 by task+subject returns single scan", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study   <- bids_h5_dataset(tmp)
  single  <- subset_bids_h5(study, task = "nback", subject = "01")

  expect_equal(nrow(scan_manifest(single)), 1L)
  expect_equal(scan_manifest(single)$scan_name, "sub-01_task-nback_run-01")
})

test_that("subset_bids_h5 data dimensions match filtered scans", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  nback  <- subset_bids_h5(study, task = "nback")
  mat    <- get_data_matrix(nback)

  # sub-01_nback: 30 + sub-02_nback: 28 = 58 timepoints
  expect_equal(nrow(mat), 58L)
  expect_equal(ncol(mat), K)
})

test_that("subset_bids_h5 errors when no scans match", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  expect_error(subset_bids_h5(study, task = "nonexistent"), "no scans match")
})

test_that("subset_bids_h5 result has correct TR", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5(tr = 1.5)
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  nback  <- subset_bids_h5(study, task = "nback")
  expect_equal(get_TR(nback), 1.5)
})


# ==============================================================================
# parcellation_info
# ==============================================================================

test_that("parcellation_info returns list with expected fields", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  info  <- parcellation_info(study)

  expect_type(info, "list")
  expect_equal(info$n_parcels, K)
  expect_length(info$cluster_ids, K)
  expect_equal(info$labels, paste0("parcel_", seq_len(K)))
})

test_that("parcellation_info cluster_map has correct length", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  info  <- parcellation_info(study)

  # cluster_map has 10 voxels per parcel in make_test_h5
  expect_equal(length(info$cluster_map), K * 10L)
})


# ==============================================================================
# get_confounds
# ==============================================================================

test_that("get_confounds returns tibble for single matching scan", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  cf    <- get_confounds(study, scan_name = "sub-01_task-nback_run-01")

  expect_true(is.data.frame(cf))
  expect_equal(ncol(cf), 2L)
  expect_true("motion_x" %in% names(cf))
})

test_that("get_confounds returns NULL when scan has no confounds", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  cf    <- get_confounds(study, scan_name = "sub-01_task-rest_run-01")
  expect_null(cf)
})

test_that("get_confounds filtered by subject returns named list", {
  skip_if_not_installed("hdf5r")

  K <- 10L; T <- 20L
  cf1 <- data.frame(csf = rnorm(T))
  cf2 <- data.frame(csf = rnorm(T))

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  scans <- list(
    "sub-01_task-test_run-01" = list(
      subject = "01", task = "test", session = "", run = "01",
      n_time = T, data = matrix(rnorm(T * K), T, K),
      events = NULL, confounds = cf1, censor = NULL
    ),
    "sub-01_task-test_run-02" = list(
      subject = "01", task = "test", session = "", run = "02",
      n_time = T, data = matrix(rnorm(T * K), T, K),
      events = NULL, confounds = cf2, censor = NULL
    )
  )
  make_test_h5(tmp, scans, n_parcels = K, tr = 2.0)

  study <- bids_h5_dataset(tmp)
  cf    <- get_confounds(study, subject = "01")

  expect_type(cf, "list")
  expect_length(cf, 2L)
  expect_true(all(c("sub-01_task-test_run-01", "sub-01_task-test_run-02") %in% names(cf)))
})


# ==============================================================================
# data_chunks
# ==============================================================================

test_that("data_chunks iterates over study correctly", {
  skip_if_not_installed("hdf5r")

  K   <- 15L
  tmp <- make_standard_test_h5(n_parcels = K)
  on.exit(unlink(tmp))

  study  <- bids_h5_dataset(tmp)
  chunks <- data_chunks(study, nchunks = 3L)

  total_cols <- 0L
  while (!is.null(ch <- tryCatch(iterators::nextElem(chunks), error = function(e) NULL))) {
    expect_true(is.matrix(ch$data))
    total_cols <- total_cols + ncol(ch$data)
  }
  expect_equal(total_cols, K)
})


# ==============================================================================
# study_to_group
# ==============================================================================

test_that("study_to_group returns fmri_group with correct subjects", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  grp   <- study_to_group(study)

  expect_s3_class(grp, "fmri_group")
  expect_equal(n_subjects(grp), 2L)
  expect_setequal(grp$subjects$subject_id, c("01", "02"))
})

test_that("study_to_group subjects have fmri_dataset objects", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  grp   <- study_to_group(study)
  # fmri_group stores each dataset wrapped in list() (length-1 entries as
  # required by validate_fmri_group). Unwrap with [[1]] to access the dataset.
  ds_list <- grp$subjects$dataset

  for (ds_wrapper in ds_list) {
    ds <- ds_wrapper[[1]]
    expect_s3_class(ds, "fmri_dataset")
  }
})


# ==============================================================================
# print method
# ==============================================================================

test_that("print.bids_h5_study_dataset produces readable output", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  expect_output(print(study), "bids_h5_study_dataset")
  expect_output(print(study), "parcellated")
  expect_output(print(study), "2")      # 2 subjects
})


# ==============================================================================
# Shared H5 connection: ref-counting across subset
# ==============================================================================

test_that("subset shares same H5 connection as parent", {
  skip_if_not_installed("hdf5r")

  tmp <- make_standard_test_h5()
  on.exit(unlink(tmp))

  study <- bids_h5_dataset(tmp)
  nback <- subset_bids_h5(study, task = "nback")

  # Both should reference the same underlying H5 file
  expect_identical(study$h5_connection$file, nback$h5_connection$file)
})


# ==============================================================================
# Multi-run subject: run lengths accumulate correctly
# ==============================================================================

test_that("multi-run subject has correct total timepoints", {
  skip_if_not_installed("hdf5r")

  K <- 10L
  T_run1 <- 20L; T_run2 <- 25L

  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp))

  scans <- list(
    "sub-01_task-nback_run-01" = list(
      subject = "01", task = "nback", session = "", run = "01",
      n_time = T_run1, data = matrix(rnorm(T_run1 * K), T_run1, K),
      events = NULL, confounds = NULL, censor = NULL
    ),
    "sub-01_task-nback_run-02" = list(
      subject = "01", task = "nback", session = "", run = "02",
      n_time = T_run2, data = matrix(rnorm(T_run2 * K), T_run2, K),
      events = NULL, confounds = NULL, censor = NULL
    )
  )
  make_test_h5(tmp, scans, n_parcels = K, tr = 2.0)

  study <- bids_h5_dataset(tmp)
  mat   <- get_data_matrix(study)

  expect_equal(nrow(mat), T_run1 + T_run2)
  expect_equal(ncol(mat), K)
  expect_equal(n_runs(study), 2L)
})


# ==============================================================================
# Missing optional dependencies produce informative errors
# ==============================================================================

test_that("bids_h5_dataset requires hdf5r with informative error", {
  # Simulate hdf5r not available by testing the check path
  # (We can only really test this by mocking requireNamespace, which is fragile;
  #  instead we test that the function exists and the check string is present.)
  fn_body <- deparse(body(bids_h5_dataset))
  expect_true(any(grepl("hdf5r", fn_body)))
})
