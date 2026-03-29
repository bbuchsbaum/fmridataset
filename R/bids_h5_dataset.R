#' BIDS H5 Dataset Reader
#'
#' @description
#' Opens a BIDS HDF5 archive written by \code{compress_bids_study()} and
#' returns a \code{bids_h5_study_dataset} object that is a subclass of
#' \code{fmri_study_dataset}. The study-level object exposes the full
#' \code{fmridataset} API (data_chunks, as_delarr, get_data_matrix, etc.)
#' together with BIDS-specific accessors for participants, tasks, sessions,
#' the scan manifest, parcellation metadata, and confound regressors.
#'
#' @details
#' Internally the reader:
#' \enumerate{
#'   \item Opens the H5 file via a shared, ref-counted connection.
#'   \item Reads the \code{/scan_index/} table to build the scan manifest.
#'   \item Creates one lightweight \code{bids_h5_scan_backend} per scan.
#'   \item Groups scan backends by subject; multi-run subjects get a nested
#'         \code{study_backend} over their scan backends.
#'   \item Composes per-subject \code{fmri_dataset} objects into a
#'         \code{fmri_study_dataset} via \code{fmri_study_dataset()}.
#'   \item Wraps the result as a \code{bids_h5_study_dataset} with the scan
#'         manifest, shared H5 connection, and a flat list of per-scan
#'         backends (used by \code{subset_bids_h5()}).
#' }
#'
#' Parcellated data lives in feature-space (K parcel columns). ROI/sphere/
#' voxel selectors do not apply; use \code{index_selector()} to select
#' parcels by column index.
#'
#' @name bids-h5-dataset
#' @importFrom stats setNames
#' @keywords internal
NULL


# ============================================================
# Internal helpers
# ============================================================

.bind_data_frames <- function(dfs) {
  dfs <- Filter(function(df) !is.null(df) && nrow(df) > 0L, dfs)
  if (length(dfs) == 0L) {
    return(NULL)
  }

  dfs <- lapply(dfs, function(df) {
    out <- as.data.frame(df, stringsAsFactors = FALSE)
    rownames(out) <- NULL
    out
  })

  all_names <- unique(unlist(lapply(dfs, names), use.names = FALSE))

  aligned <- lapply(dfs, function(df) {
    missing <- setdiff(all_names, names(df))
    for (nm in missing) {
      df[[nm]] <- NA
    }
    df[all_names]
  })

  out <- do.call(rbind, aligned)
  rownames(out) <- NULL
  out
}

.make_bids_h5_scan_backends <- function(manifest, h5_connection, n_features, tr,
                                        compression_mode = "parcellated") {
  setNames(
    lapply(seq_len(nrow(manifest)), function(i) {
      row <- manifest[i, ]

      bids_h5_scan_backend(
        h5_connection    = h5_connection,
        scan_group_path  = paste0("/scans/", row$scan_name),
        n_features       = n_features,
        n_time           = row$n_time,
        metadata         = list(
          subject = row$subject,
          task    = row$task,
          session = if (!is.na(row$session) && nzchar(row$session)) row$session else NULL,
          run     = row$run,
          tr      = tr
        ),
        compression_mode = compression_mode
      )
    }),
    manifest$scan_name
  )
}

#' Read scan index from open H5 file
#' @keywords internal
.read_scan_index <- function(h5) {
  idx_grp <- h5[["scan_index"]]
  fields <- idx_grp$ls()$name

  read_field <- function(name) {
    if (name %in% fields) idx_grp[[name]]$read() else NULL
  }

  scan_name     <- read_field("scan_name")
  subject       <- read_field("subject")
  task          <- read_field("task")
  session       <- read_field("session")
  run           <- read_field("run")
  n_time        <- read_field("n_time")
  time_offset   <- read_field("time_offset")
  has_events    <- read_field("has_events")
  has_confounds <- read_field("has_confounds")

  n <- length(scan_name)

  # Coerce types
  if (is.null(session))      session      <- rep(NA_character_, n)
  if (is.null(has_events))   has_events   <- rep(FALSE, n)
  if (is.null(has_confounds)) has_confounds <- rep(FALSE, n)
  if (is.null(time_offset))  time_offset  <- c(0L, cumsum(as.integer(n_time)))[seq_len(n)]

  tibble::tibble(
    scan_name     = as.character(scan_name),
    subject       = as.character(subject),
    task          = as.character(task),
    session       = as.character(session),
    run           = as.character(run),
    n_time        = as.integer(n_time),
    time_offset   = as.integer(time_offset),
    has_events    = as.logical(has_events),
    has_confounds = as.logical(has_confounds)
  )
}

#' Build per-subject fmri_dataset from a set of scan rows and scan backends
#'
#' @param scan_rows Rows of scan_manifest for this subject (tibble).
#' @param scan_backends Named list of bids_h5_scan_backend, keyed by scan_name.
#' @param h5 The open H5File handle (for reading events/censor).
#' @param tr Numeric TR in seconds.
#' @param subject_id Character subject ID for event_table annotation.
#' @keywords internal
.build_subject_dataset <- function(scan_rows, scan_backends, h5, tr, subject_id) {
  n_scans <- nrow(scan_rows)

  # Collect per-scan events, censor, run lengths
  events_list  <- vector("list", n_scans)
  censor_parts <- vector("list", n_scans)
  run_lengths  <- integer(n_scans)

  for (i in seq_len(n_scans)) {
    row   <- scan_rows[i, ]
    sname <- row$scan_name
    grp   <- h5[[paste0("scans/", sname)]]

    run_lengths[i] <- row$n_time

    # Events
    if (row$has_events) {
      ev <- h5_read_events(grp)
      if (!is.null(ev) && nrow(ev) > 0) {
        ev$run      <- row$run      # BIDS run label
        ev$run_id   <- i            # sequential int within subject
        ev$subject_id <- subject_id
        ev$task     <- row$task
        if (!is.na(row$session) && nzchar(row$session)) {
          ev$session <- row$session
        }
        events_list[[i]] <- ev
      }
    }

    # Censor
    censor_val <- h5_read_censor(grp)
    if (is.null(censor_val)) {
      censor_parts[[i]] <- rep(0L, row$n_time)
    } else {
      censor_parts[[i]] <- censor_val
    }
  }

  combined_events <- .bind_data_frames(events_list)
  if (is.null(combined_events)) {
    combined_events <- data.frame()
  }
  censor <- unlist(censor_parts)

  # Backend: single scan or study_backend for multiple runs
  backends_for_subject <- lapply(scan_rows$scan_name, function(sn) scan_backends[[sn]])
  if (n_scans == 1L) {
    subj_backend <- backends_for_subject[[1]]
  } else {
    subj_backend <- study_backend(
      backends    = backends_for_subject,
      subject_ids = seq_len(n_scans)
    )
  }

  # Open the backend (idempotent due to ref-counting)
  subj_backend <- backend_open(subj_backend)

  frame <- fmrihrf::sampling_frame(blocklens = run_lengths, TR = tr)

  ds <- list(
    backend        = subj_backend,
    nruns          = n_scans,
    event_table    = tibble::as_tibble(combined_events),
    sampling_frame = frame,
    censor         = censor
  )
  class(ds) <- c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
  ds
}

#' Compose scan backends + manifest into a bids_h5_study_dataset
#'
#' Shared by bids_h5_dataset() and subset_bids_h5(). Takes the full manifest
#' and a flat named list of scan_backends (keyed by scan_name), builds the
#' subject-level datasets, and returns a bids_h5_study_dataset.
#'
#' @param manifest Tibble — the scan manifest (subset of the full one).
#' @param scan_backends Named list of bids_h5_scan_backend objects.
#' @param h5 The open hdf5r H5File handle.
#' @param h5_connection The h5_shared_connection (stored on the result).
#' @param tr Numeric TR.
#' @param bids_meta Named list: space, pipeline, name (from /bids/).
#' @param compression_mode Character. Either "parcellated" or "latent".
#' @keywords internal
.compose_bids_h5_study_dataset <- function(manifest, scan_backends,
                                            h5, h5_connection,
                                            tr, bids_meta,
                                            compression_mode = "parcellated") {
  subjects_in_manifest <- unique(manifest$subject)

  datasets    <- vector("list", length(subjects_in_manifest))
  subject_ids <- subjects_in_manifest

  for (i in seq_along(subjects_in_manifest)) {
    sid       <- subjects_in_manifest[[i]]
    subj_rows <- manifest[manifest$subject == sid, ]
    datasets[[i]] <- .build_subject_dataset(
      scan_rows    = subj_rows,
      scan_backends = scan_backends,
      h5           = h5,
      tr           = tr,
      subject_id   = sid
    )
  }

  # Build the study dataset via the standard constructor
  study <- fmri_study_dataset(datasets, subject_ids = subject_ids)

  # Wrap as bids_h5_study_dataset subclass with extra fields
  study$scan_manifest    <- manifest
  study$h5_connection    <- h5_connection
  study$compression_mode <- compression_mode
  study$.bids_metadata   <- bids_meta
  study$.scan_backends   <- scan_backends

  class(study) <- c("bids_h5_study_dataset", "fmri_study_dataset", "fmri_dataset", "list")
  study
}


# ============================================================
# Main reader constructor
# ============================================================

#' Open a BIDS HDF5 Study Archive
#'
#' @description
#' Opens a BIDS HDF5 archive created by \code{compress_bids_study()} and
#' returns a \code{bids_h5_study_dataset} that is a subclass of
#' \code{fmri_study_dataset}. All standard \pkg{fmridataset} methods
#' (\code{get_data_matrix}, \code{data_chunks}, \code{as_delarr}, etc.) work
#' on the returned object.
#'
#' @param file Character string. Path to the \code{.h5} BIDS archive.
#' @param preload Logical. Reserved for future use (ignored in Phase 1).
#'
#' @return A \code{bids_h5_study_dataset} object (subclass of
#'   \code{fmri_study_dataset}).
#'
#' @seealso \code{\link{compress_bids_study}}, \code{\link{subset_bids_h5}},
#'   \code{\link{participants}}, \code{\link{tasks}}, \code{\link{sessions}}
#'
#' @export
bids_h5_dataset <- function(file, preload = FALSE) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Package 'hdf5r' is required to open BIDS H5 archives but is not installed.",
      parameter = "file"
    )
  }

  file <- normalizePath(file, mustWork = TRUE)

  # Open shared connection (validates file exists, opens H5File)
  h5_connection <- h5_shared_connection(file)
  h5            <- h5_connection$handle

  # Validate root attributes
  if (!h5$attr_exists("format")) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf("'%s' does not look like a BIDS H5 archive: missing 'format' attribute.", file),
      file = file, operation = "validate"
    )
  }
  fmt <- h5$attr_open("format")$read()
  if (!identical(fmt, "bids_h5_study")) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf("Unsupported archive format '%s' (expected 'bids_h5_study').", fmt),
      file = file, operation = "validate"
    )
  }

  if (h5$attr_exists("version")) {
    ver <- h5$attr_open("version")$read()
    if (!startsWith(as.character(ver), "1.")) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = sprintf("Unsupported BIDS H5 schema version '%s' (only 1.x is supported).", ver),
        file = file, operation = "validate"
      )
    }
  }

  comp_mode <- if (h5$attr_exists("compression_mode")) {
    h5$attr_open("compression_mode")$read()
  } else {
    "parcellated"
  }
  if (!comp_mode %in% c("parcellated", "latent")) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf(
        "Unknown compression_mode '%s' (expected 'parcellated' or 'latent').",
        comp_mode
      ),
      file = file, operation = "validate"
    )
  }

  # Read scan manifest
  manifest <- .read_scan_index(h5)
  if (nrow(manifest) == 0L) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = "BIDS H5 archive contains no scans in /scan_index/.",
      file = file, operation = "read"
    )
  }

  # Number of features depends on compression mode
  if (comp_mode == "parcellated") {
    if (!h5$exists("parcellation/cluster_ids")) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = "BIDS H5 archive is missing /parcellation/cluster_ids.",
        file = file, operation = "read"
      )
    }
    cluster_ids <- h5[["parcellation/cluster_ids"]]$read()
    n_features  <- length(cluster_ids)
  } else {
    # latent mode: K from /latent_meta/n_components
    if (!h5$exists("latent_meta/n_components")) {
      stop_fmridataset(
        fmridataset_error_backend_io,
        message = "Latent-mode archive is missing /latent_meta/n_components.",
        file = file, operation = "read"
      )
    }
    n_features <- as.integer(h5[["latent_meta/n_components"]]$read())
  }

  # TR from first scan's metadata
  first_scan_name <- manifest$scan_name[[1]]
  tr_path <- paste0("scans/", first_scan_name, "/metadata/tr")
  if (!h5$exists(tr_path)) {
    stop_fmridataset(
      fmridataset_error_backend_io,
      message = sprintf("Cannot find TR in '%s'.", tr_path),
      file = file, operation = "read"
    )
  }
  tr <- as.numeric(h5[[tr_path]]$read())

  # Read /bids/ metadata (best effort)
  bids_meta <- list(space = NULL, pipeline = NULL, name = NULL)
  if (h5$exists("bids")) {
    bids_grp <- h5[["bids"]]
    for (key in c("space", "pipeline", "name")) {
      if (bids_grp$exists(key)) {
        bids_meta[[key]] <- tryCatch(
          as.character(bids_grp[[key]]$read()),
          error = function(e) NULL
        )
      }
    }
  }

  # Create per-scan backends (all share same h5_connection)
  scan_backends <- .make_bids_h5_scan_backends(
    manifest         = manifest,
    h5_connection    = h5_connection,
    n_features       = n_features,
    tr               = tr,
    compression_mode = comp_mode
  )

  .compose_bids_h5_study_dataset(
    manifest         = manifest,
    scan_backends    = scan_backends,
    h5               = h5,
    h5_connection    = h5_connection,
    tr               = tr,
    bids_meta        = bids_meta,
    compression_mode = comp_mode
  )
}


# ============================================================
# subset_bids_h5
# ============================================================

#' Subset a BIDS H5 Study Dataset
#'
#' @description
#' Filters a \code{bids_h5_study_dataset} by task, subject, session, and/or
#' run using standard (non-NSE) evaluation. Returns a new
#' \code{bids_h5_study_dataset} built from the matching scans, sharing the
#' same underlying HDF5 file handle.
#'
#' @param x A \code{bids_h5_study_dataset} object.
#' @param task Character vector of task names to keep, or \code{NULL} for all.
#' @param subject Character vector of subject IDs to keep, or \code{NULL} for all.
#' @param session Character vector of session names to keep, or \code{NULL} for all.
#' @param run Character vector of BIDS run labels to keep, or \code{NULL} for all.
#'
#' @return A new \code{bids_h5_study_dataset} containing only the matching scans.
#'
#' @export
subset_bids_h5 <- function(x,
                            task    = NULL,
                            subject = NULL,
                            session = NULL,
                            run     = NULL) {
  if (!inherits(x, "bids_h5_study_dataset")) {
    stop("'x' must be a bids_h5_study_dataset object.", call. = FALSE)
  }

  manifest <- x$scan_manifest
  keep      <- rep(TRUE, nrow(manifest))

  if (!is.null(task))    keep <- keep & (manifest$task    %in% task)
  if (!is.null(subject)) keep <- keep & (manifest$subject %in% subject)
  if (!is.null(run))     keep <- keep & (manifest$run     %in% run)
  if (!is.null(session)) {
    keep <- keep & (manifest$session %in% session)
  }

  sub_manifest <- manifest[keep, ]

  if (nrow(sub_manifest) == 0L) {
    stop("subset_bids_h5: no scans match the provided filters.", call. = FALSE)
  }

  n_features <- if (!is.null(x$.scan_backends) && length(x$.scan_backends) > 0L) {
    x$.scan_backends[[1]]$n_features
  } else if (x$compression_mode == "latent") {
    as.integer(x$h5_connection$handle[["latent_meta/n_components"]]$read())
  } else {
    length(x$h5_connection$handle[["parcellation/cluster_ids"]]$read())
  }

  sub_backends <- .make_bids_h5_scan_backends(
    manifest         = sub_manifest,
    h5_connection    = x$h5_connection,
    n_features       = n_features,
    tr               = get_TR(x),
    compression_mode = x$compression_mode
  )

  h5 <- x$h5_connection$handle

  .compose_bids_h5_study_dataset(
    manifest         = sub_manifest,
    scan_backends    = sub_backends,
    h5               = h5,
    h5_connection    = x$h5_connection,
    tr               = get_TR(x),
    bids_meta        = x$.bids_metadata,
    compression_mode = x$compression_mode
  )
}


# ============================================================
# S3 accessors for bids_h5_study_dataset
# ============================================================

#' @rdname participants
#' @method participants bids_h5_study_dataset
#' @export
participants.bids_h5_study_dataset <- function(x, ...) {
  unique(x$scan_manifest$subject)
}

#' @rdname tasks
#' @method tasks bids_h5_study_dataset
#' @export
tasks.bids_h5_study_dataset <- function(x, ...) {
  unique(x$scan_manifest$task)
}

#' @rdname sessions
#' @method sessions bids_h5_study_dataset
#' @export
sessions.bids_h5_study_dataset <- function(x, ...) {
  sess <- unique(x$scan_manifest$session)
  # Return NULL when all sessions are NA or empty string
  sess <- sess[!is.na(sess) & nzchar(sess)]
  if (length(sess) == 0L) NULL else sess
}

#' @rdname scan_manifest
#' @method scan_manifest bids_h5_study_dataset
#' @export
scan_manifest.bids_h5_study_dataset <- function(x, ...) {
  x$scan_manifest
}

#' @rdname parcellation_info
#' @method parcellation_info bids_h5_study_dataset
#' @export
parcellation_info.bids_h5_study_dataset <- function(x, ...) {
  if (identical(x$compression_mode, "latent")) {
    return(NULL)
  }

  h5 <- x$h5_connection$handle

  if (!h5$is_valid) {
    stop("H5 file handle is no longer valid.", call. = FALSE)
  }

  cluster_ids <- h5[["parcellation/cluster_ids"]]$read()
  n_parcels   <- length(cluster_ids)

  cluster_map <- if (h5$exists("parcellation/cluster_map")) {
    h5[["parcellation/cluster_map"]]$read()
  } else {
    NULL
  }

  labels <- tryCatch({
    if (h5$exists("parcellation") &&
        h5[["parcellation"]]$exists("cluster_meta") &&
        h5[["parcellation/cluster_meta"]]$exists("labels")) {
      h5[["parcellation/cluster_meta/labels"]]$read()
    } else {
      NULL
    }
  }, error = function(e) NULL)

  list(
    cluster_ids = cluster_ids,
    cluster_map = cluster_map,
    labels      = labels,
    n_parcels   = n_parcels
  )
}

#' @rdname get_loadings
#' @method get_loadings bids_h5_study_dataset
#' @export
get_loadings.bids_h5_study_dataset <- function(x, scan_name = NULL, ...) {
  if (!identical(x$compression_mode, "latent")) {
    stop("get_loadings() is only available for latent-mode archives.", call. = FALSE)
  }

  h5 <- x$h5_connection$handle

  if (!h5$is_valid) {
    stop("H5 file handle is no longer valid.", call. = FALSE)
  }

  # Check for shared template
  has_template <- .has_shared_template(h5)

  all_scans <- x$scan_manifest$scan_name

  .get_one_loadings <- function(sn) {
    sgp <- paste0("/scans/", sn)
    per_scan <- .read_scan_loadings(h5, sgp)
    if (!is.null(per_scan)) return(per_scan)
    # Fall back to shared template loadings
    if (has_template) return(.read_template_loadings(h5))
    stop(sprintf("No loadings found for scan '%s' and no shared template.", sn),
         call. = FALSE)
  }

  if (!is.null(scan_name)) {
    if (!scan_name %in% all_scans) {
      stop(sprintf("scan_name '%s' not found in this dataset.", scan_name), call. = FALSE)
    }
    return(.get_one_loadings(scan_name))
  }

  result <- lapply(all_scans, .get_one_loadings)
  names(result) <- all_scans
  result
}

#' @rdname reconstruct_voxels
#' @method reconstruct_voxels bids_h5_study_dataset
#' @export
reconstruct_voxels.bids_h5_study_dataset <- function(x, scan_name, rows = NULL,
                                                       voxels = NULL, ...) {
  if (!identical(x$compression_mode, "latent")) {
    stop("reconstruct_voxels() is only available for latent-mode archives.", call. = FALSE)
  }

  h5 <- x$h5_connection$handle

  if (!h5$is_valid) {
    stop("H5 file handle is no longer valid.", call. = FALSE)
  }

  all_scans <- x$scan_manifest$scan_name
  if (!scan_name %in% all_scans) {
    stop(sprintf("scan_name '%s' not found in this dataset.", scan_name), call. = FALSE)
  }

  backend <- x$.scan_backends[[scan_name]]
  if (is.null(backend)) {
    stop(sprintf("No backend found for scan '%s'.", scan_name), call. = FALSE)
  }

  sgp <- paste0("/scans/", scan_name)

  # Read basis [T, K]
  basis    <- backend_get_data(backend)
  # Read loadings [V, K] — per-scan or shared template
  loadings <- .read_scan_loadings(h5, sgp)
  if (is.null(loadings) && .has_shared_template(h5)) {
    loadings <- .read_template_loadings(h5)
  }
  if (is.null(loadings)) {
    stop(sprintf("No loadings found for scan '%s' and no shared template.", scan_name),
         call. = FALSE)
  }
  # Read offset [V]
  offset   <- .read_scan_offset(h5, sgp)

  # Reconstruct: [T, K] %*% t([V, K]) = [T, V]
  recon <- basis %*% t(loadings)

  if (length(offset) > 0L) {
    # offset is [V]; add row-wise
    recon <- sweep(recon, 2L, offset, `+`)
  }

  if (!is.null(rows)) {
    recon <- recon[rows, , drop = FALSE]
  }
  if (!is.null(voxels)) {
    recon <- recon[, voxels, drop = FALSE]
  }

  recon
}

#' @rdname encoding_info
#' @method encoding_info bids_h5_study_dataset
#' @export
encoding_info.bids_h5_study_dataset <- function(x, ...) {
  if (!identical(x$compression_mode, "latent")) {
    return(NULL)
  }

  h5 <- x$h5_connection$handle

  if (!h5$is_valid) {
    stop("H5 file handle is no longer valid.", call. = FALSE)
  }

  encoding_family <- if (h5$exists("latent_meta/encoding_family")) {
    h5[["latent_meta/encoding_family"]]$read()
  } else {
    NA_character_
  }

  encoding_params <- if (h5$exists("latent_meta/encoding_params")) {
    tryCatch(
      jsonlite::fromJSON(h5[["latent_meta/encoding_params"]]$read()),
      error = function(e) NULL
    )
  } else {
    NULL
  }

  n_components <- as.integer(h5[["latent_meta/n_components"]]$read())

  has_template <- .has_shared_template(h5)
  template_meta <- if (has_template) {
    tryCatch({
      meta_json <- h5[["latent_meta/template/meta"]]$read()
      jsonlite::fromJSON(meta_json)
    }, error = function(e) list())
  } else {
    NULL
  }

  list(
    encoding_family      = encoding_family,
    encoding_params      = encoding_params,
    n_components         = n_components,
    has_shared_template  = has_template,
    template_meta        = template_meta
  )
}

#' @rdname get_confounds
#' @method get_confounds bids_h5_study_dataset
#' @param scan_name Character. Scan name key (exact match), or \code{NULL}.
#' @param subject Character. Subject ID filter, or \code{NULL}.
#' @param task Character. Task filter, or \code{NULL}.
#' @export
get_confounds.bids_h5_study_dataset <- function(x,
                                                 scan_name = NULL,
                                                 subject   = NULL,
                                                 task      = NULL,
                                                 ...) {
  manifest <- x$scan_manifest
  keep      <- manifest$has_confounds

  if (!is.null(scan_name)) keep <- keep & (manifest$scan_name %in% scan_name)
  if (!is.null(subject))   keep <- keep & (manifest$subject   %in% subject)
  if (!is.null(task))      keep <- keep & (manifest$task      %in% task)

  matching <- manifest[keep, ]

  if (nrow(matching) == 0L) {
    return(NULL)
  }

  h5 <- x$h5_connection$handle
  result <- lapply(matching$scan_name, function(sn) {
    grp <- h5[[paste0("scans/", sn)]]
    h5_read_confounds(grp)
  })
  names(result) <- matching$scan_name

  if (length(result) == 1L) result[[1]] else result
}


# ============================================================
# study_to_group helper
# ============================================================

#' Convert a BIDS H5 Study Dataset to an fmri_group
#'
#' @description
#' Converts a \code{bids_h5_study_dataset} (or any \code{fmri_study_dataset})
#' to an \code{fmri_group} object with one row per subject. Use this when you
#' need per-subject group operations via \code{group_map()}.
#'
#' @param x A \code{bids_h5_study_dataset} or \code{fmri_study_dataset}.
#' @param ... Currently unused.
#'
#' @return An \code{fmri_group} with columns \code{subject_id} and \code{dataset}.
#'
#' @export
study_to_group <- function(x, ...) {
  UseMethod("study_to_group")
}

#' @rdname study_to_group
#' @method study_to_group bids_h5_study_dataset
#' @export
study_to_group.bids_h5_study_dataset <- function(x, ...) {
  .study_to_group_impl(x)
}

#' @rdname study_to_group
#' @method study_to_group fmri_study_dataset
#' @export
study_to_group.fmri_study_dataset <- function(x, ...) {
  .study_to_group_impl(x)
}

#' @keywords internal
.study_to_group_impl <- function(x) {
  sids <- x$subject_ids

  # Reconstruct per-subject fmri_dataset objects from the study_backend
  sb       <- x$backend
  backends <- sb$backends

  datasets <- lapply(seq_along(sids), function(i) {
    sid         <- sids[[i]]
    subj_backend <- backends[[i]]

    # Find matching rows in the combined event_table
    et <- x$event_table
    if (!is.null(et) && nrow(et) > 0 && "subject_id" %in% names(et)) {
      subj_et <- et[et$subject_id == sid, ]
    } else {
      subj_et <- if (!is.null(et)) et else data.frame()
    }

    # Run lengths for this subject from the backend time dims
    dims <- backend_get_dims(subj_backend)
    run_lengths <- if (inherits(subj_backend, "study_backend")) {
      subj_backend$time_dims
    } else {
      dims$time
    }

    tr <- get_TR(x)

    ds <- list(
      backend        = subj_backend,
      nruns          = length(run_lengths),
      event_table    = tibble::as_tibble(subj_et),
      sampling_frame = fmrihrf::sampling_frame(blocklens = run_lengths, TR = tr),
      censor         = rep(0L, dims$time)
    )
    class(ds) <- c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
    ds
  })

  # fmri_group validates that each element of the dataset list-column has length 1.
  # We must store each dataset wrapped in list() so the column is a list of
  # single-element lists (i.e., subjects_df$dataset[[i]] has length 1).
  subjects_df <- tibble::tibble(
    subject_id = sids,
    dataset    = lapply(datasets, list)
  )

  fmri_group(
    subjects    = subjects_df,
    id          = "subject_id",
    dataset_col = "dataset"
  )
}


# ============================================================
# Print method
# ============================================================

#' @export
print.bids_h5_study_dataset <- function(x, ...) {
  m          <- x$scan_manifest
  n_subjects <- length(unique(m$subject))
  n_tasks    <- length(unique(m$task))
  sess_vals  <- unique(m$session)
  sess_vals  <- sess_vals[!is.na(sess_vals) & nzchar(sess_vals)]
  n_sessions <- length(sess_vals)
  n_scans    <- nrow(m)
  n_features <- if (!is.null(x$.scan_backends) && length(x$.scan_backends) > 0) {
    x$.scan_backends[[1]]$n_features
  } else {
    NA_integer_
  }
  tr_val     <- get_TR(x)
  total_tp   <- sum(m$n_time)

  cat("<bids_h5_study_dataset>\n")
  cat("  format        : BIDS H5 Study Archive\n")

  if (identical(x$compression_mode, "latent")) {
    cat("  mode          : latent (", n_features, "components)\n", sep = "")
  } else {
    cat("  mode          : parcellated (", n_features, "parcels)\n", sep = "")
  }

  cat("  subjects      :", n_subjects, "\n")
  cat("  tasks         :", n_tasks,
      "(", paste(unique(m$task), collapse = ", "), ")\n")
  if (n_sessions > 0) {
    cat("  sessions      :", n_sessions,
        "(", paste(sess_vals, collapse = ", "), ")\n")
  }
  cat("  scans         :", n_scans, "\n")
  cat("  TR            :", tr_val, "s\n")
  cat("  total time    :", total_tp, "volumes\n")
  if (!is.null(x$.bids_metadata$name)) {
    cat("  study name    :", x$.bids_metadata$name, "\n")
  }
  if (!is.null(x$.bids_metadata$space)) {
    cat("  space         :", x$.bids_metadata$space, "\n")
  }

  invisible(x)
}
