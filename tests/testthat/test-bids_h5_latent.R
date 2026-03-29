# Tests for latent-mode BIDS H5 archives
# Builds minimal H5 files directly with hdf5r — no bidser or fmrilatent required.

skip_if_not_installed("hdf5r")
skip_if_not_installed("tibble")

# ============================================================
# Helper: build a minimal latent-mode H5 file
# ============================================================

.make_latent_h5 <- function(path,
                              n_scans      = 2L,
                              n_time_each  = c(10L, 12L),
                              K            = 5L,
                              V            = 20L,
                              tr           = 2.0,
                              encoding_family = "pca",
                              encoding_params = '{"n_components":5}') {
  h5 <- hdf5r::H5File$new(path, mode = "w")
  on.exit(if (h5$is_valid) h5$close_all(), add = TRUE)

  # Root attributes
  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "latent")

  # /latent_meta/
  lm <- h5$create_group("latent_meta")
  lm$create_dataset("n_components",    robj = as.integer(K))
  lm$create_dataset("encoding_family", robj = encoding_family)
  lm$create_dataset("encoding_params", robj = encoding_params)

  # /spatial/  (minimal: just a placeholder)
  sp <- h5$create_group("spatial")
  sp$create_dataset("mask", robj = rep(TRUE, V))

  # /scan_index/ — parallel arrays
  subjects  <- paste0("sub-0", seq_len(n_scans))
  tasks     <- rep("rest", n_scans)
  sessions  <- rep("", n_scans)
  runs      <- paste0("0", seq_len(n_scans))
  scan_names <- paste0("sub-0", seq_len(n_scans), "_task-rest_run-0", seq_len(n_scans))
  n_time_vec <- as.integer(n_time_each)
  time_offsets <- c(0L, cumsum(n_time_vec)[-length(n_time_vec)])

  si <- h5$create_group("scan_index")
  si$create_dataset("scan_name",     robj = scan_names)
  si$create_dataset("subject",       robj = subjects)
  si$create_dataset("task",          robj = tasks)
  si$create_dataset("session",       robj = sessions)
  si$create_dataset("run",           robj = runs)
  si$create_dataset("n_time",        robj = n_time_vec)
  si$create_dataset("time_offset",   robj = time_offsets)
  si$create_dataset("has_events",    robj = rep(FALSE, n_scans))
  si$create_dataset("has_confounds", robj = rep(FALSE, n_scans))

  # /scans/<name>/ for each scan
  scans_grp <- h5$create_group("scans")
  for (i in seq_len(n_scans)) {
    sname <- scan_names[[i]]
    T_i   <- n_time_vec[[i]]

    sg <- scans_grp$create_group(sname)

    # metadata
    md <- sg$create_group("metadata")
    md$create_dataset("subject", robj = subjects[[i]])
    md$create_dataset("task",    robj = tasks[[i]])
    md$create_dataset("run",     robj = runs[[i]])
    md$create_dataset("tr",      robj = tr)

    # data
    dg <- sg$create_group("data")

    basis    <- matrix(rnorm(T_i * K), nrow = T_i, ncol = K)
    loadings <- matrix(rnorm(V * K),   nrow = V,   ncol = K)
    offset   <- rnorm(V)

    dg$create_dataset("basis",    robj = basis)
    dg$create_dataset("loadings", robj = loadings)
    dg$create_dataset("offset",   robj = offset)
  }

  invisible(path)
}

# ============================================================
# Helper: build a minimal parcellated-mode H5 file
# ============================================================

.make_parcellated_h5 <- function(path,
                                   n_scans     = 2L,
                                   n_time_each = c(10L, 12L),
                                   K           = 5L,
                                   tr          = 2.0) {
  h5 <- hdf5r::H5File$new(path, mode = "w")
  on.exit(if (h5$is_valid) h5$close_all(), add = TRUE)

  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "parcellated")

  # /parcellation/
  pg <- h5$create_group("parcellation")
  pg$create_dataset("cluster_ids", robj = seq_len(K))

  subjects   <- paste0("sub-0", seq_len(n_scans))
  tasks      <- rep("rest", n_scans)
  sessions   <- rep("", n_scans)
  runs       <- paste0("0", seq_len(n_scans))
  scan_names <- paste0("sub-0", seq_len(n_scans), "_task-rest_run-0", seq_len(n_scans))
  n_time_vec <- as.integer(n_time_each)
  time_offsets <- c(0L, cumsum(n_time_vec)[-length(n_time_vec)])

  si <- h5$create_group("scan_index")
  si$create_dataset("scan_name",     robj = scan_names)
  si$create_dataset("subject",       robj = subjects)
  si$create_dataset("task",          robj = tasks)
  si$create_dataset("session",       robj = sessions)
  si$create_dataset("run",           robj = runs)
  si$create_dataset("n_time",        robj = n_time_vec)
  si$create_dataset("time_offset",   robj = time_offsets)
  si$create_dataset("has_events",    robj = rep(FALSE, n_scans))
  si$create_dataset("has_confounds", robj = rep(FALSE, n_scans))

  scans_grp <- h5$create_group("scans")
  for (i in seq_len(n_scans)) {
    sname <- scan_names[[i]]
    T_i   <- n_time_vec[[i]]

    sg <- scans_grp$create_group(sname)

    md <- sg$create_group("metadata")
    md$create_dataset("subject", robj = subjects[[i]])
    md$create_dataset("task",    robj = tasks[[i]])
    md$create_dataset("run",     robj = runs[[i]])
    md$create_dataset("tr",      robj = tr)

    dg <- sg$create_group("data")
    summary_data <- matrix(rnorm(T_i * K), nrow = T_i, ncol = K)
    dg$create_dataset("summary_data", robj = summary_data)
  }

  invisible(path)
}


# ============================================================
# Reader tests
# ============================================================

test_that("bids_h5_dataset() opens a latent-mode archive correctly", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)

  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = 5L, V = 20L)

  ds <- bids_h5_dataset(h5_path)
  expect_s3_class(ds, "bids_h5_study_dataset")
  expect_equal(ds$compression_mode, "latent")
})

test_that("compression_mode is 'latent' on opened object", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path)
  ds <- bids_h5_dataset(h5_path)
  expect_identical(ds$compression_mode, "latent")
})

test_that("participants(), tasks(), sessions() work on latent archive", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L)

  ds <- bids_h5_dataset(h5_path)
  expect_length(participants(ds), 2L)
  expect_identical(tasks(ds), "rest")
  expect_null(sessions(ds))
})

test_that("scan_manifest has correct rows and columns", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 3L, n_time_each = c(8L, 9L, 10L))

  ds <- bids_h5_dataset(h5_path)
  m  <- scan_manifest(ds)
  expect_equal(nrow(m), 3L)
  expect_true(all(c("scan_name", "subject", "task", "n_time") %in% names(m)))
  expect_equal(m$n_time, c(8L, 9L, 10L))
})

test_that("get_data_matrix() returns [T_total, K] for latent archive", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  K <- 5L
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = K, V = 20L)

  ds  <- bids_h5_dataset(h5_path)
  mat <- get_data_matrix(ds)
  expect_equal(nrow(mat), 22L)   # 10 + 12
  expect_equal(ncol(mat), K)
})

test_that("get_TR() returns correct TR for latent archive", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, tr = 1.5)
  ds <- bids_h5_dataset(h5_path)
  expect_equal(get_TR(ds), 1.5)
})


# ============================================================
# Accessor tests
# ============================================================

test_that("encoding_info() returns correct family, params and K", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, K = 7L,
                   encoding_family = "ica",
                   encoding_params = '{"n_components":7,"whiten":true}')

  ds   <- bids_h5_dataset(h5_path)
  info <- encoding_info(ds)

  expect_equal(info$encoding_family, "ica")
  expect_equal(info$n_components, 7L)
  expect_true(is.list(info$encoding_params))
  expect_equal(info$encoding_params$n_components, 7)
})

test_that("get_loadings(study, scan_name=...) returns [V, K] matrix", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  K <- 5L; V <- 20L
  .make_latent_h5(h5_path, n_scans = 2L, K = K, V = V)

  ds     <- bids_h5_dataset(h5_path)
  sname  <- scan_manifest(ds)$scan_name[[1]]
  L      <- get_loadings(ds, scan_name = sname)

  expect_true(is.matrix(L))
  expect_equal(nrow(L), V)
  expect_equal(ncol(L), K)
})

test_that("get_loadings(study) returns named list of all scans", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L, K = 5L, V = 20L)

  ds  <- bids_h5_dataset(h5_path)
  lst <- get_loadings(ds)

  expect_type(lst, "list")
  expect_length(lst, 2L)
  expect_equal(names(lst), scan_manifest(ds)$scan_name)
  for (L in lst) {
    expect_true(is.matrix(L))
    expect_equal(ncol(L), 5L)
    expect_equal(nrow(L), 20L)
  }
})

test_that("reconstruct_voxels() returns [T, V] matrix", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  K <- 5L; V <- 20L; T_i <- 10L
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(T_i, 12L), K = K, V = V)

  ds    <- bids_h5_dataset(h5_path)
  sname <- scan_manifest(ds)$scan_name[[1]]
  recon <- reconstruct_voxels(ds, scan_name = sname)

  expect_true(is.matrix(recon))
  expect_equal(nrow(recon), T_i)
  expect_equal(ncol(recon), V)
})

test_that("reconstruct_voxels() rows subset works", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = 5L, V = 20L)

  ds    <- bids_h5_dataset(h5_path)
  sname <- scan_manifest(ds)$scan_name[[1]]
  recon <- reconstruct_voxels(ds, scan_name = sname, rows = 1:5)

  expect_equal(nrow(recon), 5L)
  expect_equal(ncol(recon), 20L)
})

test_that("reconstruct_voxels() voxels subset works", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = 5L, V = 20L)

  ds    <- bids_h5_dataset(h5_path)
  sname <- scan_manifest(ds)$scan_name[[1]]
  recon <- reconstruct_voxels(ds, scan_name = sname, voxels = 1:8)

  expect_equal(nrow(recon), 10L)
  expect_equal(ncol(recon), 8L)
})

test_that("parcellation_info() returns NULL for latent mode", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path)
  ds <- bids_h5_dataset(h5_path)
  expect_null(parcellation_info(ds))
})


# ============================================================
# Subset tests
# ============================================================

test_that("subset_bids_h5 by task works on latent archives", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L)

  ds  <- bids_h5_dataset(h5_path)
  sub <- subset_bids_h5(ds, task = "rest")

  expect_s3_class(sub, "bids_h5_study_dataset")
  expect_equal(nrow(scan_manifest(sub)), 2L)
})

test_that("subset_bids_h5 preserves compression_mode", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path, n_scans = 2L)

  ds  <- bids_h5_dataset(h5_path)
  sub <- subset_bids_h5(ds, subject = participants(ds)[[1]])

  expect_identical(sub$compression_mode, "latent")
})

test_that("get_data_matrix on subset returns correct dims", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  K <- 5L
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = K, V = 20L)

  ds     <- bids_h5_dataset(h5_path)
  sub_p  <- participants(ds)[[1]]
  sub    <- subset_bids_h5(ds, subject = sub_p)
  mat    <- get_data_matrix(sub)

  expect_equal(ncol(mat), K)
  expect_equal(nrow(mat), 10L)   # first subject has 10 timepoints
})


# ============================================================
# Error tests
# ============================================================

test_that("get_loadings() on parcellated archive errors", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_parcellated_h5(h5_path, K = 5L)

  ds <- bids_h5_dataset(h5_path)
  expect_error(get_loadings(ds), regexp = "latent-mode")
})

test_that("reconstruct_voxels() on parcellated archive errors", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_parcellated_h5(h5_path, K = 5L)

  ds <- bids_h5_dataset(h5_path)
  expect_error(reconstruct_voxels(ds, scan_name = "anything"), regexp = "latent-mode")
})

test_that("unknown compression_mode errors on open", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)

  # Write a file with a bogus compression_mode
  h5 <- hdf5r::H5File$new(h5_path, mode = "w")
  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "bogus_mode")
  h5$close_all()

  expect_error(bids_h5_dataset(h5_path), regexp = "Unknown compression_mode")
})

test_that("get_loadings() errors for unknown scan_name", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  .make_latent_h5(h5_path)
  ds <- bids_h5_dataset(h5_path)
  expect_error(get_loadings(ds, scan_name = "nonexistent_scan"), regexp = "not found")
})


# ============================================================
# data_chunks test
# ============================================================

test_that("data_chunks() works on latent-mode study", {
  h5_path <- tempfile(fileext = ".h5")
  on.exit(unlink(h5_path), add = TRUE)
  K <- 5L
  .make_latent_h5(h5_path, n_scans = 2L, n_time_each = c(10L, 12L), K = K, V = 20L)

  ds     <- bids_h5_dataset(h5_path)
  chunks <- data_chunks(ds, nchunks = 2L)

  expect_false(is.null(chunks))

  # Collect all chunks and check dimensions
  all_data <- list()
  while (!is.null(chunk <- tryCatch(iterators::nextElem(chunks), error = function(e) NULL))) {
    all_data <- c(all_data, list(chunk$data))
  }
  expect_true(length(all_data) >= 1L)

  total_cols <- sum(vapply(all_data, ncol, integer(1L)))
  expect_equal(total_cols, K)
})
