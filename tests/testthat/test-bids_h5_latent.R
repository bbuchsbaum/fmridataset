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


# ============================================================
# Shared template tests
# ============================================================

.make_template_h5 <- function(path,
                                n_scans      = 2L,
                                n_time_each  = c(10L, 12L),
                                K            = 5L,
                                V            = 20L,
                                tr           = 2.0) {
  h5 <- hdf5r::H5File$new(path, mode = "w")
  on.exit(if (h5$is_valid) h5$close_all(), add = TRUE)

  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", "latent")

  # /latent_meta/ with shared template
  lm <- h5$create_group("latent_meta")
  lm$create_dataset("n_components",       robj = as.integer(K))
  lm$create_dataset("encoding_family",    robj = "shared_template")
  lm$create_dataset("encoding_params",    robj = "{}")
  lm$create_dataset("has_shared_template", robj = TRUE)

  # /latent_meta/template/ — shared loadings
  tpl <- lm$create_group("template")
  template_loadings <- matrix(rnorm(V * K), nrow = V, ncol = K)
  tpl$create_dataset("loadings", robj = template_loadings)
  tpl$create_dataset("meta", robj = '{"basis_spec":"slepian","k":5}')

  # /spatial/
  sp <- h5$create_group("spatial")
  sp_hdr <- sp$create_group("header")
  sp_hdr$create_dataset("dim",    robj = c(V, 1L, 1L))
  sp_hdr$create_dataset("pixdim", robj = c(2.0, 2.0, 2.0))
  sp$create_dataset("mask",         robj = rep(1L, V))
  sp$create_dataset("voxel_coords", robj = matrix(seq_len(V * 3L), ncol = 3L))

  # /scans/ — per-scan basis only, NO loadings
  h5$create_group("scans")
  subjects <- c("01", "02")[seq_len(n_scans)]
  scan_names <- paste0("sub-", subjects, "_task-rest_run-01")

  for (i in seq_len(n_scans)) {
    nt <- n_time_each[i]
    sg <- h5[["scans"]]$create_group(scan_names[i])
    dg <- sg$create_group("data")
    dg$create_dataset("basis", robj = matrix(rnorm(nt * K), nrow = nt, ncol = K))
    # NO loadings or offset — template mode

    # events
    eg <- sg$create_group("events")
    hdf5r::h5attr(eg, "n_events") <- 2L
    eg$create_dataset("onset",      robj = c(0.0, 5.0))
    eg$create_dataset("duration",   robj = c(1.0, 1.0))
    eg$create_dataset("trial_type", robj = c("A", "B"))

    # censor
    sg$create_dataset("censor", robj = rep(0L, nt))

    # metadata
    mg <- sg$create_group("metadata")
    mg$create_dataset("subject", robj = subjects[i])
    mg$create_dataset("task",    robj = "rest")
    mg$create_dataset("run",     robj = "01")
    mg$create_dataset("tr",      robj = tr)
  }

  # /scan_index/
  si <- h5$create_group("scan_index")
  si$create_dataset("scan_name",    robj = scan_names)
  si$create_dataset("subject",      robj = subjects)
  si$create_dataset("task",         robj = rep("rest", n_scans))
  si$create_dataset("session",      robj = rep("", n_scans))
  si$create_dataset("run",          robj = rep("01", n_scans))
  si$create_dataset("n_time",       robj = as.integer(n_time_each))
  si$create_dataset("time_offset",  robj = c(0L, cumsum(n_time_each[-n_scans])))
  si$create_dataset("has_events",   robj = rep(TRUE, n_scans))
  si$create_dataset("has_confounds", robj = rep(FALSE, n_scans))

  invisible(list(template_loadings = template_loadings))
}


test_that("shared template: reader opens correctly", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  info <- .make_template_h5(tmp)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  expect_s3_class(study, "bids_h5_study_dataset")
  expect_equal(study$compression_mode, "latent")
})

test_that("shared template: get_data_matrix returns correct dims", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  K <- 5L
  n_time <- c(10L, 12L)
  .make_template_h5(tmp, K = K, n_time_each = n_time)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  mat <- get_data_matrix(study)
  expect_equal(nrow(mat), sum(n_time))
  expect_equal(ncol(mat), K)
})

test_that("shared template: get_loadings falls back to template", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  K <- 5L; V <- 20L
  info <- .make_template_h5(tmp, K = K, V = V)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  # Single scan — should get template loadings
  loadings <- get_loadings(study, scan_name = "sub-01_task-rest_run-01")
  expect_equal(dim(loadings), c(V, K))
  expect_equal(loadings, info$template_loadings)

  # All scans — each should get same template loadings
  all_loadings <- get_loadings(study)
  expect_length(all_loadings, 2L)
  expect_equal(all_loadings[[1]], info$template_loadings)
  expect_equal(all_loadings[[2]], info$template_loadings)
})

test_that("shared template: reconstruct_voxels works", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  K <- 5L; V <- 20L; nt <- c(10L, 12L)
  info <- .make_template_h5(tmp, K = K, V = V, n_time_each = nt)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  recon <- reconstruct_voxels(study, scan_name = "sub-01_task-rest_run-01")
  expect_equal(dim(recon), c(nt[1], V))

  # Subset rows
  recon_sub <- reconstruct_voxels(study, scan_name = "sub-01_task-rest_run-01",
                                   rows = 1:3)
  expect_equal(dim(recon_sub), c(3L, V))

  # Subset voxels
  recon_vox <- reconstruct_voxels(study, scan_name = "sub-01_task-rest_run-01",
                                   voxels = c(1L, 5L, 10L))
  expect_equal(dim(recon_vox), c(nt[1], 3L))
})

test_that("shared template: encoding_info reports template", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  .make_template_h5(tmp, K = 5L)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  info <- encoding_info(study)
  expect_equal(info$encoding_family, "shared_template")
  expect_true(info$has_shared_template)
  expect_equal(info$n_components, 5L)
  expect_is(info$template_meta, "list")
})

test_that("shared template: subset preserves template fallback", {
  tmp <- tempfile(fileext = ".h5")
  on.exit(unlink(tmp), add = TRUE)

  K <- 5L; V <- 20L
  info <- .make_template_h5(tmp, K = K, V = V)
  study <- bids_h5_dataset(tmp)
  on.exit(study$h5_connection$release(), add = TRUE)

  sub <- subset_bids_h5(study, subject = "01")
  loadings <- get_loadings(sub, scan_name = "sub-01_task-rest_run-01")
  expect_equal(dim(loadings), c(V, K))
  expect_equal(loadings, info$template_loadings)
})
