#' Compress a BIDS Study into a Single HDF5 Archive
#'
#' @description
#' Converts a BIDS directory (or \code{bidser::bids_project}) into a single
#' compressed HDF5 file containing compressed fMRI data, events, confounds, and
#' study metadata. The output file can be opened with \code{\link{bids_h5_dataset}}.
#'
#' @details
#' The writer streams scans one at a time — only one NIfTI image is held in
#' memory at a time. For each scan it:
#' \enumerate{
#'   \item Reads the NIfTI via \code{neuroim2::read_vec()}.
#'   \item For \strong{parcellated} mode: computes parcel averages via
#'         \code{fmristore::summarize_by_clusters()} and writes \code{[T, K]}
#'         to \code{/scans/<name>/data/summary_data}.
#'   \item For \strong{latent} mode: encodes via \code{fmrilatent::encode()}
#'         and writes basis \code{[T, K]}, loadings \code{[V, K]}, and
#'         (optionally) offset \code{[V]} to \code{/scans/<name>/data/}.
#'   \item Writes events, confounds, censor, and metadata sub-groups.
#'   \item Releases the NIfTI from memory.
#' }
#'
#' After all scans are written the \code{/scan_index/} lookup table is
#' populated and the function returns a \code{bids_h5_dataset} reader object.
#'
#' @section HDF5 schema:
#' See \code{bids_plan.md} in the package source for the full v1.0 schema.
#' The root \code{compression_mode} attribute reflects the chosen \code{mode}.
#'
#' @param x A \code{bidser::bids_project} object **or** a character path to a
#'   BIDS directory (automatically opened with \code{bidser::bids_project()}).
#' @param file Character. Path for the output \code{.h5} file. Parent directory
#'   must exist. Existing files are overwritten.
#' @param mode Character. Compression strategy: \code{"parcellated"} (default)
#'   or \code{"latent"}.
#' @param clusters A \code{neuroim2::ClusteredNeuroVol} defining the parcellation
#'   atlas in study space. Required for \code{mode = "parcellated"};
#'   ignored for \code{"latent"}.
#' @param summary_fun Function applied to voxel time-series within each parcel
#'   to produce a scalar summary (default: \code{mean}). Only used for
#'   \code{mode = "parcellated"}.
#' @param encoding A \code{fmrilatent} encoding specification object (e.g.
#'   \code{fmrilatent::spec_time_dct(k = 15)}). Required for
#'   \code{mode = "latent"} unless \code{n_components} is provided.
#' @param n_components Integer. Shorthand for latent PCA with K components.
#'   If \code{encoding} is \code{NULL} and \code{n_components} is provided,
#'   \code{fmrilatent::spec_space_pca(k = n_components)} is used.
#'   Only used for \code{mode = "latent"}.
#' @param template Optional \code{fmrilatent} template object (e.g. from
#'   \code{fmrilatent::parcel_basis_template()} or
#'   \code{fmrilatent::build_hierarchical_template()}). When provided, the
#'   template's spatial loadings are stored once in \code{/latent_meta/template/}
#'   and per-scan data is reduced to \code{[T, K]} projection coefficients
#'   (no per-scan loadings). This significantly reduces file size for
#'   multi-subject studies. Only used for \code{mode = "latent"}.
#' @param mask A \code{neuroim2::LogicalNeuroVol} brain mask. For
#'   \code{mode = "parcellated"}, derived from \code{clusters} when \code{NULL}.
#'   For \code{mode = "latent"}, \code{mask} is required (cannot be derived
#'   without clusters).
#' @param space Character. Template space name stored as metadata
#'   (default: \code{"MNI152NLin2009cAsym"}).
#' @param tasks Character vector. Task filter; \code{NULL} means all tasks.
#' @param subjects Character vector. Subject filter; \code{NULL} means all
#'   subjects.
#' @param sessions Character vector. Session filter; \code{NULL} means all
#'   sessions (including session-less datasets).
#' @param confounds A confound specification passed to
#'   \code{bidser::read_confounds()}, e.g. a character vector of column names,
#'   a \code{bidser::confound_set()}, or \code{NULL} to skip confound writing.
#' @param compression Integer 0–9. HDF5 gzip compression level (default 4).
#' @param verbose Logical. If \code{TRUE} (default) print progress messages.
#'
#' @return A \code{bids_h5_dataset} object (reader for the newly created file).
#'   If the reader is not yet available the file path is returned invisibly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(bidser)
#' library(neuroim2)
#' library(fmristore)
#'
#' bids_dir  <- system.file("extdata", "ds001", package = "bidser")
#' atlas     <- fmristore::get_schaefer_atlas(100)   # example atlas
#'
#' # Parcellated mode
#' study <- compress_bids_study(
#'   x          = bids_dir,
#'   file       = tempfile(fileext = ".h5"),
#'   clusters   = atlas,
#'   tasks      = "nback",
#'   verbose    = TRUE
#' )
#'
#' # Latent mode (PCA with 50 components)
#' study_lat <- compress_bids_study(
#'   x            = bids_dir,
#'   file         = tempfile(fileext = ".h5"),
#'   mode         = "latent",
#'   n_components = 50L,
#'   mask         = brain_mask,
#'   tasks        = "nback",
#'   verbose      = TRUE
#' )
#' }
compress_bids_study <- function(
  x,
  file,
  mode         = c("parcellated", "latent"),
  clusters     = NULL,
  summary_fun  = mean,
  encoding     = NULL,
  n_components = NULL,
  template = NULL,
  mask         = NULL,
  space        = "MNI152NLin2009cAsym",
  tasks        = NULL,
  subjects     = NULL,
  sessions     = NULL,
  confounds    = NULL,
  compression  = 4L,
  verbose      = TRUE
) {
  # ---------------------------------------------------------------
  # 1. Dependency checks
  # ---------------------------------------------------------------
  if (!requireNamespace("bidser",  quietly = TRUE)) {
    stop("Package 'bidser' is required for compress_bids_study() but is not installed.\n",
         "Install it with: install.packages('bidser') or remotes::install_github('bbuchsbaum/bidser')",
         call. = FALSE)
  }
  if (!requireNamespace("hdf5r",   quietly = TRUE)) {
    stop("Package 'hdf5r' is required for compress_bids_study() but is not installed.\n",
         "Install it with: install.packages('hdf5r')",
         call. = FALSE)
  }

  mode <- match.arg(mode)

  if (mode == "parcellated" && !requireNamespace("fmristore", quietly = TRUE)) {
    stop("Package 'fmristore' is required for mode='parcellated' but is not installed.\n",
         "Install it with: remotes::install_github('bbuchsbaum/fmristore')",
         call. = FALSE)
  }
  if (mode == "latent" && !requireNamespace("fmrilatent", quietly = TRUE)) {
    stop("Package 'fmrilatent' is required for mode='latent' but is not installed.\n",
         "Install it with: remotes::install_github('bbuchsbaum/fmrilatent')",
         call. = FALSE)
  }

  compression <- as.integer(compression)

  # ---------------------------------------------------------------
  # 2. Resolve BIDS project
  # ---------------------------------------------------------------
  if (is.character(x)) {
    if (!dir.exists(x)) {
      stop("BIDS directory not found: ", x, call. = FALSE)
    }
    if (verbose) message("Opening BIDS project: ", x)
    x <- bidser::bids_project(x)
  }
  if (!inherits(x, "bids_project")) {
    stop("'x' must be a bidser::bids_project object or a path to a BIDS directory.",
         call. = FALSE)
  }

  # ---------------------------------------------------------------
  # 3. Query scan manifest
  # ---------------------------------------------------------------
  if (verbose) message("Querying scan manifest ...")
  scans_df <- bidser::preproc_scans(x)

  if (is.null(scans_df) || nrow(scans_df) == 0L) {
    stop("No preprocessed scans found in BIDS project. ",
         "Ensure fMRIPrep outputs are present.", call. = FALSE)
  }

  # Apply filters
  if (!is.null(subjects)) {
    scans_df <- scans_df[scans_df$subid %in% subjects, , drop = FALSE]
  }
  if (!is.null(tasks)) {
    scans_df <- scans_df[scans_df$task %in% tasks, , drop = FALSE]
  }
  if (!is.null(sessions) && "session" %in% names(scans_df)) {
    scans_df <- scans_df[scans_df$session %in% sessions, , drop = FALSE]
  }

  if (nrow(scans_df) == 0L) {
    stop("No scans remain after filtering by tasks/subjects/sessions.", call. = FALSE)
  }

  if (verbose) message(sprintf("Found %d scans to process.", nrow(scans_df)))

  # ---------------------------------------------------------------
  # 4. Validate TR consistency
  # ---------------------------------------------------------------
  trs <- .extract_trs(scans_df)

  if (length(unique(round(trs, 6))) > 1L) {
    stop(sprintf(
      "TR values are not consistent across scans:\n%s\n",
      paste(sprintf("  %s: TR=%.4f", scans_df$scan_name, trs), collapse = "\n")
    ), call. = FALSE)
  }
  TR_value <- trs[[1]]

  # ---------------------------------------------------------------
  # 5. Validate mode-specific arguments and derive mask/encoding
  # ---------------------------------------------------------------
  if (mode == "parcellated") {
    if (is.null(clusters) || !inherits(clusters, "ClusteredNeuroVol")) {
      stop("'clusters' must be a neuroim2::ClusteredNeuroVol object for mode='parcellated'.",
           call. = FALSE)
    }

    if (is.null(mask)) {
      # Derive mask: all voxels belonging to a non-zero cluster
      mask <- neuroim2::as.LogicalNeuroVol(clusters > 0)
    }

    if (!inherits(mask, "LogicalNeuroVol")) {
      stop("'mask' must be a neuroim2::LogicalNeuroVol object.", call. = FALSE)
    }

    # Parcel IDs and count
    cluster_ids <- sort(unique(as.integer(clusters[clusters > 0])))
    K <- length(cluster_ids)

    if (K == 0L) {
      stop("No parcels found in 'clusters'. Check that the ClusteredNeuroVol is valid.",
           call. = FALSE)
    }
  } else {
    # latent mode
    if (is.null(mask)) {
      stop("'mask' is required for mode='latent' (no clusters to derive mask from).",
           call. = FALSE)
    }
    if (!inherits(mask, "LogicalNeuroVol")) {
      stop("'mask' must be a neuroim2::LogicalNeuroVol object.", call. = FALSE)
    }

    if (!is.null(template)) {
      # Template mode: shared loadings, per-scan coefficients only
      if (!inherits(template, "parcel_basis_template") &&
          !inherits(template, "HierarchicalBasisTemplate") &&
          !is.list(template)) {
        stop("'template' must be a fmrilatent template object (parcel_basis_template or HierarchicalBasisTemplate).",
             call. = FALSE)
      }
      # Extract template loadings
      template_loadings <- tryCatch(
        as.matrix(fmrilatent::template_loadings(template)),
        error = function(e) {
          stop(sprintf("Failed to extract loadings from template: %s", e$message),
               call. = FALSE)
        }
      )
      K <- ncol(template_loadings)
      if (is.null(encoding)) {
        # Template provides the spatial basis; no explicit encoding needed
        encoding <- NULL
      }
    } else {
      template_loadings <- NULL
      if (is.null(encoding)) {
        if (is.null(n_components)) {
          stop("For mode='latent', either 'encoding', 'n_components', or 'template' must be provided.",
               call. = FALSE)
        }
        encoding <- fmrilatent::spec_space_pca(k = as.integer(n_components))
      }
      # K will be determined after encoding the first scan; set a placeholder
      K <- NULL
    }
    cluster_ids <- NULL
  }

  # ---------------------------------------------------------------
  # 6. Build scan names for manifest
  # ---------------------------------------------------------------
  scans_df$scan_name <- .make_scan_name(scans_df)

  # ---------------------------------------------------------------
  # 7. Create output HDF5 file
  # ---------------------------------------------------------------
  parent_dir <- dirname(file)
  if (!dir.exists(parent_dir)) {
    stop("Output directory does not exist: ", parent_dir, call. = FALSE)
  }

  if (verbose) message("Creating HDF5 file: ", file)
  h5 <- hdf5r::H5File$new(file, mode = "w")
  on.exit({
    if (h5$is_valid) {
      tryCatch(h5$close_all(), error = function(e) invisible(NULL))
    }
  }, add = TRUE)

  # ---------------------------------------------------------------
  # 8. Root attributes
  # ---------------------------------------------------------------
  h5$create_attr("format",           "bids_h5_study")
  h5$create_attr("version",          "1.0")
  h5$create_attr("compression_mode", mode)
  h5$create_attr("writer_version",   as.character(utils::packageVersion("fmridataset")))

  # ---------------------------------------------------------------
  # 9. Write /bids/ group
  # ---------------------------------------------------------------
  if (verbose) message("Writing /bids/ metadata ...")
  .write_bids_group(h5, x, scans_df, space, compression)

  # ---------------------------------------------------------------
  # 10. Write /spatial/ group
  # ---------------------------------------------------------------
  if (verbose) message("Writing /spatial/ metadata ...")
  .write_spatial_group(h5, mask, compression)

  # ---------------------------------------------------------------
  # 11. Write mode-specific metadata group
  # ---------------------------------------------------------------
  if (mode == "parcellated") {
    if (verbose) message("Writing /parcellation/ metadata ...")
    .write_parcellation_group(h5, clusters, cluster_ids, compression)
  }
  # latent_meta is written after scan loop once K is known

  # ---------------------------------------------------------------
  # 12. Write /scans/ — streaming one scan at a time
  # ---------------------------------------------------------------
  h5$create_group("scans")
  scans_grp <- h5[["scans"]]

  n_scans  <- nrow(scans_df)
  n_time_vec        <- integer(n_scans)
  n_features_vec    <- integer(n_scans)
  has_events_vec    <- logical(n_scans)
  has_confounds_vec <- logical(n_scans)

  for (i in seq_len(n_scans)) {
    scan_row  <- scans_df[i, , drop = FALSE]
    scan_name <- scan_row$scan_name
    scan_path <- .get_scan_path(scan_row)

    if (verbose) {
      message(sprintf("  [%d/%d] %s", i, n_scans, scan_name))
    }

    # -- Read NIfTI
    nvec <- tryCatch(
      neuroim2::read_vec(scan_path, mask = mask),
      error = function(e) {
        stop(sprintf("Failed to read scan '%s': %s", scan_path, e$message),
             call. = FALSE)
      }
    )

    n_time <- dim(nvec)[4]
    n_time_vec[[i]] <- n_time

    # -- Create scan HDF5 group
    scans_grp$create_group(scan_name)
    sg <- scans_grp[[scan_name]]
    sg$create_group("data")
    dg <- sg[["data"]]

    if (mode == "parcellated") {
      # -- Compute parcel averages: [T, K] matrix
      parcel_mat <- .compute_parcel_matrix(nvec, clusters, cluster_ids, summary_fun)

      chunk_t <- min(n_time, 128L)
      chunk_k <- min(K, 256L)

      dg$create_dataset(
        "summary_data",
        robj       = parcel_mat,
        chunk_dims = c(chunk_t, chunk_k),
        gzip_level = compression
      )

      rm(parcel_mat)
    } else {
      # latent mode
      mask_indices <- which(as.logical(mask))
      mat <- neuroim2::series(nvec, mask_indices)  # [T, V]

      if (!is.null(template_loadings)) {
        # -- Template mode: project data onto shared template loadings
        # Center data per-voxel, store offset for reconstruction fidelity
        col_means <- colMeans(mat)
        mat_centered <- sweep(mat, 2L, col_means, `-`)

        # Project: coefficients = centered_mat %*% loadings %*% solve(t(L) %*% L)
        basis_mat <- tryCatch({
          proj <- fmrilatent::template_project(template, mat_centered)
          as.matrix(proj)
        }, error = function(e) {
          # Manual fallback: OLS projection
          as.matrix(mat_centered %*% template_loadings %*%
                      solve(crossprod(template_loadings)))
        })

        # Store per-scan offset (voxel means) for round-trip fidelity
        loadings_mat <- NULL
        offset_vec   <- col_means
        rm(mat_centered, col_means)
      } else {
        # -- Independent encoding mode
        lvec <- fmrilatent::encode(mat, encoding, mask = mask)
        basis_mat    <- as.matrix(fmrilatent::basis(lvec))     # [T, K]
        loadings_mat <- as.matrix(fmrilatent::loadings(lvec))  # [V, K]
        offset_vec   <- fmrilatent::offset(lvec)               # [V] or numeric(0)
      }

      # Determine K from first scan
      if (is.null(K)) {
        K <- ncol(basis_mat)
      }

      chunk_t <- min(n_time, 128L)
      chunk_k <- min(K, 256L)

      dg$create_dataset(
        "basis",
        robj       = basis_mat,
        chunk_dims = c(chunk_t, chunk_k),
        gzip_level = compression
      )

      # Per-scan loadings only in non-template mode
      if (!is.null(loadings_mat)) {
        chunk_v <- min(nrow(loadings_mat), 4096L)
        dg$create_dataset(
          "loadings",
          robj       = loadings_mat,
          chunk_dims = c(chunk_v, chunk_k),
          gzip_level = compression
        )
      }
      if (length(offset_vec) > 0L) {
        dg$create_dataset(
          "offset",
          robj       = offset_vec,
          chunk_dims = min(length(offset_vec), 4096L),
          gzip_level = compression
        )
      }
      hdf5r::h5attr(dg, "k") <- ncol(basis_mat)

      rm(mat, basis_mat)
      if (!is.null(loadings_mat)) rm(loadings_mat)
      rm(offset_vec)
      if (exists("lvec", inherits = FALSE)) rm(lvec)
    }

    n_features_vec[[i]] <- K

    # -- Read and write events
    events_df <- tryCatch(
      .read_scan_events(x, scan_row),
      error = function(e) {
        if (verbose) message("    (no events: ", e$message, ")")
        NULL
      }
    )
    if (!is.null(events_df) && nrow(events_df) > 0L) {
      h5_write_events(sg, events_df, compression = compression)
      has_events_vec[[i]] <- TRUE
    }

    # -- Read and write confounds
    if (!is.null(confounds)) {
      confounds_df <- tryCatch(
        .read_scan_confounds(x, scan_row, confounds),
        error = function(e) {
          if (verbose) message("    (no confounds: ", e$message, ")")
          NULL
        }
      )
      if (!is.null(confounds_df) && nrow(confounds_df) > 0L) {
        h5_write_confounds(sg, confounds_df, compression = compression)
        has_confounds_vec[[i]] <- TRUE
      }
    }

    # -- Write all-zeros censor (default: no timepoints censored)
    censor_vec <- rep(0L, n_time)
    h5_write_censor(sg, censor_vec, compression = compression)

    # -- Write scan metadata
    has_session <- "session" %in% names(scan_row) && !is.na(scan_row$session) &&
                   nchar(as.character(scan_row$session)) > 0L
    meta <- list(
      subject = as.character(scan_row$subid),
      task    = as.character(scan_row$task),
      run     = as.character(scan_row$run),
      tr      = TR_value
    )
    if (has_session) {
      meta$session <- as.character(scan_row$session)
    }
    h5_write_scan_metadata(sg, meta)

    # -- Release NIfTI from memory
    rm(nvec)
    gc(verbose = FALSE)
  }

  # ---------------------------------------------------------------
  # 13. Write /latent_meta/ (latent mode only)
  # ---------------------------------------------------------------
  if (mode == "latent") {
    if (is.null(K)) {
      stop("No scans were processed; cannot determine K for latent_meta.", call. = FALSE)
    }
    if (verbose) message("Writing /latent_meta/ ...")
    h5$create_group("latent_meta")
    lm_grp <- h5[["latent_meta"]]

    # Template takes priority: when template is used, encoding is irrelevant
    # (data is projected onto template, not independently encoded)
    encoding_family <- if (!is.null(template_loadings)) {
      "shared_template"
    } else if (!is.null(encoding)) {
      class(encoding)[[1]]
    } else {
      "unknown"
    }
    encoding_params <- if (!is.null(template_loadings)) {
      "{}"
    } else if (!is.null(encoding)) {
      tryCatch(
        jsonlite::toJSON(as.list(encoding), auto_unbox = TRUE),
        error = function(e) "{}"
      )
    } else {
      "{}"
    }

    lm_grp$create_dataset("encoding_family",  robj = encoding_family)
    lm_grp$create_dataset("encoding_params",  robj = as.character(encoding_params))
    lm_grp$create_dataset("n_components",     robj = as.integer(K))

    # Write shared template if provided
    has_template <- !is.null(template_loadings)
    lm_grp$create_dataset("has_shared_template", robj = has_template)

    if (has_template) {
      if (verbose) message("Writing /latent_meta/template/ ...")
      lm_grp$create_group("template")
      tpl_grp <- lm_grp[["template"]]

      chunk_v <- min(nrow(template_loadings), 4096L)
      chunk_k <- min(ncol(template_loadings), 256L)
      tpl_grp$create_dataset(
        "loadings",
        robj       = template_loadings,
        chunk_dims = c(chunk_v, chunk_k),
        gzip_level = compression
      )

      # Store template metadata
      tpl_meta <- tryCatch(
        as.list(fmrilatent::template_meta(template)),
        error = function(e) list()
      )
      tpl_meta_json <- tryCatch(
        jsonlite::toJSON(tpl_meta, auto_unbox = TRUE),
        error = function(e) "{}"
      )
      tpl_grp$create_dataset("meta", robj = as.character(tpl_meta_json))
    }
  }

  # ---------------------------------------------------------------
  # 14. Write /scan_index/
  # ---------------------------------------------------------------
  if (verbose) message("Writing /scan_index/ ...")
  time_offset <- c(0L, cumsum(n_time_vec[-n_scans]))

  h5$create_group("scan_index")
  si <- h5[["scan_index"]]

  has_session_col <- "session" %in% names(scans_df)

  si$create_dataset("scan_name",      robj = scans_df$scan_name,           gzip_level = compression)
  si$create_dataset("subject",        robj = as.character(scans_df$subid), gzip_level = compression)
  si$create_dataset("task",           robj = as.character(scans_df$task),  gzip_level = compression)
  si$create_dataset("run",            robj = as.character(scans_df$run),   gzip_level = compression)
  si$create_dataset("n_time",         robj = n_time_vec,                   gzip_level = compression)
  si$create_dataset("n_features",     robj = n_features_vec,               gzip_level = compression)
  si$create_dataset("time_offset",    robj = time_offset,                  gzip_level = compression)
  si$create_dataset("has_events",     robj = as.integer(has_events_vec),   gzip_level = compression)
  si$create_dataset("has_confounds",  robj = as.integer(has_confounds_vec),gzip_level = compression)

  if (has_session_col) {
    session_vals <- ifelse(is.na(scans_df$session), "", as.character(scans_df$session))
    si$create_dataset("session", robj = session_vals, gzip_level = compression)
  } else {
    si$create_dataset("session", robj = rep("", n_scans), gzip_level = compression)
  }

  # ---------------------------------------------------------------
  # 14. Close file cleanly
  # ---------------------------------------------------------------
  h5$close_all()

  if (verbose) message("Done. Archive written to: ", file)

  # ---------------------------------------------------------------
  # 15. Return reader
  # ---------------------------------------------------------------
  # TODO: replace with bids_h5_dataset(file) once Task #3 is complete
  if (exists("bids_h5_dataset", mode = "function")) {
    return(bids_h5_dataset(file))
  }
  invisible(file)
}


# ================================================================
# Internal helpers
# ================================================================

#' Build scan name string from scan manifest row
#' @keywords internal
.make_scan_name <- function(scans_df) {
  has_session <- "session" %in% names(scans_df) &&
                 !all(is.na(scans_df$session)) &&
                 !all(nchar(as.character(scans_df$session)) == 0L)

  if (has_session) {
    sprintf(
      "sub-%s_ses-%s_task-%s_run-%s",
      scans_df$subid,
      scans_df$session,
      scans_df$task,
      scans_df$run
    )
  } else {
    sprintf(
      "sub-%s_task-%s_run-%s",
      scans_df$subid,
      scans_df$task,
      scans_df$run
    )
  }
}


#' Extract TR values from scan manifest
#'
#' Tries the \code{tr} column first, then the BIDS JSON sidecar.
#' @keywords internal
.extract_trs <- function(scans_df) {
  if ("tr" %in% names(scans_df) && !all(is.na(scans_df$tr))) {
    return(as.numeric(scans_df$tr))
  }

  # Fall back: try to read from the scan path sidecar
  trs <- vapply(seq_len(nrow(scans_df)), function(i) {
    scan_path <- .get_scan_path(scans_df[i, , drop = FALSE])
    json_path <- sub("\\.nii(\\.gz)?$", ".json", scan_path)
    if (file.exists(json_path) && requireNamespace("jsonlite", quietly = TRUE)) {
      meta <- tryCatch(
        jsonlite::read_json(json_path, simplifyVector = TRUE),
        error = function(e) list()
      )
      if (!is.null(meta$RepetitionTime)) {
        return(as.numeric(meta$RepetitionTime))
      }
    }
    NA_real_
  }, FUN.VALUE = numeric(1))

  if (anyNA(trs)) {
    warning("Could not determine TR for some scans; assuming TRs are consistent.")
    trs[is.na(trs)] <- trs[!is.na(trs)][[1]]
  }

  trs
}


#' Get scan file path from manifest row
#' @keywords internal
.get_scan_path <- function(scan_row) {
  # bidser preproc_scans() typically has a 'path' or 'full_path' column
  path_col <- intersect(c("path", "full_path", "filename", "file"), names(scan_row))
  if (length(path_col) == 0L) {
    stop("Cannot determine scan file path from manifest row. ",
         "Expected column 'path', 'full_path', 'filename', or 'file'.",
         call. = FALSE)
  }
  as.character(scan_row[[path_col[[1]]]])
}


#' Compute parcel-averaged time series matrix
#'
#' Returns a \code{[T, K]} numeric matrix by averaging voxel time series
#' within each parcel.
#'
#' @keywords internal
.compute_parcel_matrix <- function(nvec, clusters, cluster_ids, summary_fun) {
  # Try fmristore first
  if (requireNamespace("fmristore", quietly = TRUE)) {
    result <- tryCatch(
      fmristore::summarize_by_clusters(nvec, clusters, FUN = summary_fun),
      error = function(e) NULL
    )
    if (!is.null(result)) {
      return(as.matrix(result))
    }
  }

  # Manual fallback: extract masked matrix [T, N_voxels] then aggregate by cluster
  # neuroim2::series() returns [N_voxels, T]; we need [T, K]
  mask_logical <- as.logical(clusters > 0)
  vox_mat <- neuroim2::series(nvec, which(mask_logical))  # [T, N_voxels]

  cluster_vec <- as.integer(clusters)[mask_logical]

  K <- length(cluster_ids)
  T <- nrow(vox_mat)
  parcel_mat <- matrix(NA_real_, nrow = T, ncol = K)

  for (j in seq_len(K)) {
    idx <- which(cluster_vec == cluster_ids[[j]])
    if (length(idx) == 1L) {
      parcel_mat[, j] <- vox_mat[, idx]
    } else {
      parcel_mat[, j] <- apply(vox_mat[, idx, drop = FALSE], 1, summary_fun)
    }
  }

  parcel_mat
}


#' Write /bids/ group to HDF5 file
#' @keywords internal
.write_bids_group <- function(h5, bids_proj, scans_df, space, compression) {
  h5$create_group("bids")
  bg <- h5[["bids"]]

  # Study name: use root directory basename
  bids_root <- tryCatch(bids_proj$path, error = function(e) "unknown")
  bg$create_dataset("name",  robj = basename(bids_root))
  bg$create_dataset("space", robj = space)

  # Pipeline: try to detect from derivative path
  pipeline <- tryCatch({
    deriv_path <- bids_proj$derivative_path
    if (!is.null(deriv_path)) basename(deriv_path) else "unknown"
  }, error = function(e) "unknown")
  bg$create_dataset("pipeline", robj = pipeline)

  # Tasks and sessions
  unique_tasks    <- sort(unique(as.character(scans_df$task)))
  bg$create_dataset("tasks", robj = unique_tasks, gzip_level = compression)

  if ("session" %in% names(scans_df)) {
    unique_sessions <- sort(unique(as.character(scans_df$session[!is.na(scans_df$session)])))
    if (length(unique_sessions) > 0L) {
      bg$create_dataset("sessions", robj = unique_sessions, gzip_level = compression)
    }
  }

  # Dataset description JSON
  desc_path <- file.path(tryCatch(bids_proj$path, error = function(e) ""),
                         "dataset_description.json")
  if (file.exists(desc_path) && requireNamespace("jsonlite", quietly = TRUE)) {
    desc_json <- tryCatch(
      paste(readLines(desc_path, warn = FALSE), collapse = "\n"),
      error = function(e) "{}"
    )
    bg$create_dataset("dataset_description", robj = desc_json)
  }

  # Participants sub-group — filter to subjects actually included in scans_df
  parts <- tryCatch(bidser::participants(bids_proj), error = function(e) NULL)
  if (!is.null(parts) && is.data.frame(parts) && nrow(parts) > 0L) {
    # Identify the participant ID column and filter to selected subjects
    pid_col <- intersect(c("participant_id", "subid", "sub"), names(parts))
    selected_subs <- unique(as.character(scans_df$subid))
    if (length(pid_col) > 0L) {
      pid_vals <- as.character(parts[[pid_col[1]]])
      # Match with or without "sub-" prefix
      keep <- pid_vals %in% selected_subs |
              pid_vals %in% paste0("sub-", selected_subs) |
              sub("^sub-", "", pid_vals) %in% selected_subs
      parts <- parts[keep, , drop = FALSE]
    }
  }
  if (!is.null(parts) && is.data.frame(parts) && nrow(parts) > 0L) {
    bg$create_group("participants")
    pg <- bg[["participants"]]
    for (col in names(parts)) {
      col_vals <- parts[[col]]
      if (is.factor(col_vals)) col_vals <- as.character(col_vals)
      tryCatch(
        pg$create_dataset(col, robj = col_vals, gzip_level = compression),
        error = function(e) invisible(NULL)
      )
    }
  }

  invisible(NULL)
}


#' Write /spatial/ group to HDF5 file
#' @keywords internal
.write_spatial_group <- function(h5, mask, compression) {
  h5$create_group("spatial")
  spg <- h5[["spatial"]]
  spg$create_group("header")
  hg <- spg[["header"]]

  # Extract space info from mask
  sp <- neuroim2::space(mask)
  dims_3d <- dim(mask)

  hg$create_dataset("dim",    robj = as.integer(dims_3d))
  hg$create_dataset("pixdim", robj = as.numeric(neuroim2::spacing(sp)))

  # qform/affine
  aff <- tryCatch(neuroim2::trans(sp), error = function(e) diag(4))
  hg$create_dataset("qform", robj = aff)

  # Brain mask as 3D uint8
  mask_arr <- array(as.integer(as.array(mask)), dim = dims_3d)
  spg$create_dataset(
    "mask",
    robj       = mask_arr,
    chunk_dims = c(min(dims_3d[[1]], 64L), min(dims_3d[[2]], 64L), min(dims_3d[[3]], 64L)),
    gzip_level = compression
  )

  # Voxel coordinates: [N_voxels, 3] int32
  mask_idx  <- which(as.logical(mask))
  vox_coords <- tryCatch({
    neuroim2::index_to_grid(sp, mask_idx)
  }, error = function(e) {
    # Fallback: convert linear indices to 3D coords manually
    d <- dims_3d
    k <- ((mask_idx - 1L) %/% (d[[1]] * d[[2]])) + 1L
    j <- (((mask_idx - 1L) %% (d[[1]] * d[[2]])) %/% d[[1]]) + 1L
    i <- ((mask_idx - 1L) %% d[[1]]) + 1L
    cbind(i, j, k)
  })

  spg$create_dataset(
    "voxel_coords",
    robj       = matrix(as.integer(vox_coords), ncol = 3L),
    chunk_dims = c(min(nrow(vox_coords), 4096L), 3L),
    gzip_level = compression
  )

  invisible(NULL)
}


#' Write /parcellation/ group to HDF5 file
#' @keywords internal
.write_parcellation_group <- function(h5, clusters, cluster_ids, compression) {
  h5$create_group("parcellation")
  pg <- h5[["parcellation"]]

  # Full-volume cluster map as flattened int32
  cluster_map_arr <- as.integer(as.array(clusters))
  pg$create_dataset(
    "cluster_map",
    robj       = cluster_map_arr,
    chunk_dims = min(length(cluster_map_arr), 65536L),
    gzip_level = compression
  )

  pg$create_dataset(
    "cluster_ids",
    robj       = as.integer(cluster_ids),
    gzip_level = compression
  )

  # Optional cluster metadata (labels etc.)
  has_labels <- tryCatch({
    labs <- neuroim2::labels(clusters)
    !is.null(labs) && length(labs) > 0L
  }, error = function(e) FALSE)

  if (has_labels) {
    pg$create_group("cluster_meta")
    cmg <- pg[["cluster_meta"]]
    labs <- as.character(neuroim2::labels(clusters))
    cmg$create_dataset("labels", robj = labs, gzip_level = compression)
  }

  invisible(NULL)
}


#' Read events for a single scan from a BIDS project
#' @keywords internal
.read_scan_events <- function(bids_proj, scan_row) {
  # bidser::read_events expects subject + task (+ optionally session/run)
  subid <- as.character(scan_row$subid)
  task  <- as.character(scan_row$task)

  args <- list(x = bids_proj, subid = subid, task = task)

  if ("run" %in% names(scan_row) && !is.na(scan_row$run)) {
    args$run <- as.character(scan_row$run)
  }
  if ("session" %in% names(scan_row) && !is.na(scan_row$session) &&
      nchar(as.character(scan_row$session)) > 0L) {
    args$session <- as.character(scan_row$session)
  }

  result <- tryCatch(
    do.call(bidser::read_events, args),
    error = function(e) NULL
  )

  if (is.null(result)) {
    return(NULL)
  }

  # bidser returns a nested tibble with a `data` list-column by default.
  if (is.data.frame(result) && "data" %in% names(result) && is.list(result$data)) {
    result <- .bind_data_frames(lapply(result$data, as.data.frame))
  } else if (is.list(result) && !is.data.frame(result)) {
    result <- .bind_data_frames(lapply(result, as.data.frame))
  }

  result
}


#' Read confounds for a single scan from a BIDS project
#' @keywords internal
.read_scan_confounds <- function(bids_proj, scan_row, confounds_spec) {
  subid <- as.character(scan_row$subid)
  task  <- as.character(scan_row$task)

  args <- list(x = bids_proj, subid = subid, task = task)

  if ("run" %in% names(scan_row) && !is.na(scan_row$run)) {
    args$run <- as.character(scan_row$run)
  }
  if ("session" %in% names(scan_row) && !is.na(scan_row$session) &&
      nchar(as.character(scan_row$session)) > 0L) {
    args$session <- as.character(scan_row$session)
  }
  if (!is.null(confounds_spec)) {
    args$cvars <- confounds_spec
  }
  args$nest <- FALSE

  result <- tryCatch(
    do.call(bidser::read_confounds, args),
    error = function(e) NULL
  )

  if (is.null(result)) {
    return(NULL)
  }

  # Normalize to data.frame
  if (is.matrix(result)) {
    result <- as.data.frame(result)
  }
  if (is.data.frame(result) && "data" %in% names(result) && is.list(result$data)) {
    result <- .bind_data_frames(lapply(result$data, as.data.frame))
  } else if (is.list(result) && !is.data.frame(result)) {
    result <- .bind_data_frames(lapply(result, as.data.frame))
  }

  meta_cols <- intersect(
    c("participant_id", "subject", "subid", ".subid",
      "task", ".task", "run", ".run", "session", ".session"),
    names(result)
  )
  if (length(meta_cols) > 0L) {
    result <- result[setdiff(names(result), meta_cols)]
  }

  result
}
