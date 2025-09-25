#' Print Methods for fmridataset Objects
#'
#' Display formatted summaries of fmridataset objects including datasets,
#' chunk iterators, and data chunks.
#'
#' @param x An object to print (fmri_dataset, latent_dataset, chunkiter, or data_chunk)
#' @param object An object to summarize (for summary methods)
#' @param full Logical; if TRUE, print additional details for datasets (default: FALSE)
#' @param ... Additional arguments passed to print methods
#'
#' @return The object invisibly
#'
#' @examples
#' \donttest{
#' # Print dataset summary
#' # dataset <- fmri_dataset(...)
#' # print(dataset)
#' # print(dataset, full = TRUE)  # More details
#' }
#'
#' @name print
#' @aliases print.fmri_dataset print.chunkiter print.data_chunk
#' @importFrom utils head tail
NULL

#' @export
#' @rdname print
print.fmri_dataset <- function(x, full = FALSE, ...) {
  # Header
  cat("\n=== fMRI Dataset ===\n")

  # Basic dimensions
  cat("\n** Dimensions:\n")
  cat("  - Timepoints:", sum(x$sampling_frame$blocklens), "\n")
  cat("  - Runs:", x$nruns, if (x$nruns > 10) " runs" else "", "\n")

  # Data source info
  print_data_source_info(x, full = full)

  if (full) {
    mask <- get_mask(x)
    cat("  - Voxels in mask:", sum(mask > 0), "\n")
    cat("  - Mask dimensions:", paste(dim(mask), collapse = " x "), "\n")
  } else {
    cat("  - Voxels in mask: (lazy)\n")
  }

  # Sampling frame info
  cat("\n** Temporal Structure:\n")
  # Handle TR being a vector - use first value
  tr_value <- if (length(x$sampling_frame$TR) > 1) x$sampling_frame$TR[1] else x$sampling_frame$TR
  cat("  - TR: ", tr_value, " seconds\n", sep = "")
  # Handle long run lengths
  run_lens <- x$sampling_frame$blocklens
  if (length(run_lens) > 10) {
    run_str <- paste0(
      paste(head(run_lens, 5), collapse = ", "),
      ", ... (", length(run_lens), " runs total)"
    )
  } else {
    run_str <- paste(run_lens, collapse = ", ")
  }
  cat("  - Run lengths:", run_str, "\n")

  # Event table summary
  cat("\n** Event Table:\n")
  if (!is.null(x$event_table) && !is.null(nrow(x$event_table)) && nrow(x$event_table) > 0) {
    cat("  - Rows:", nrow(x$event_table), "\n")
    cat("  - Variables:", paste(names(x$event_table), collapse = ", "), "\n")

    # Show first few events if they exist
    if (nrow(x$event_table) > 0) {
      cat("  - First few events:\n")
      print(head(x$event_table, 3))
    }
  } else {
    cat("  - Empty event table\n")
  }

  cat("\n")
  invisible(x)
}

#' @export
#' @method summary fmri_dataset
#' @rdname print
summary.fmri_dataset <- function(object, ...) {
  # Header
  cat("\n=== fMRI Dataset Summary ===\n")

  # Basic dimensions
  cat("\n** Dimensions:\n")
  cat("  - Timepoints:", sum(object$sampling_frame$blocklens), "\n")
  cat("  - Runs:", object$nruns, "\n")

  # Data source info
  print_data_source_info(object, full = FALSE)

  cat("  - Voxels in mask: (lazy)\n")

  # Sampling frame info
  cat("\n** Temporal Structure:\n")
  tr_values <- object$sampling_frame$TR
  tr_value <- if (length(tr_values) > 1) tr_values[1] else tr_values
  cat("  - TR: ", tr_value, " seconds\n", sep = "")
  cat("  - Run lengths:", paste(object$sampling_frame$blocklens, collapse = ", "), "\n")

  # Event table summary
  cat("\n** Event Summary:\n")
  if (!is.null(object$event_table) && !is.null(nrow(object$event_table)) && nrow(object$event_table) > 0) {
    cat("  - Total events:", nrow(object$event_table), "\n")
    cat("  - Variables:", paste(names(object$event_table), collapse = ", "), "\n")

    # Summary by trial type if available
    if ("trial_type" %in% names(object$event_table)) {
      tt_summary <- table(object$event_table$trial_type)
      cat("  - Trial types:\n")
      for (i in seq_along(tt_summary)) {
        cat("    -", names(tt_summary)[i], ":", tt_summary[i], "events\n")
      }
    }
  } else {
    cat("  - No events\n")
  }

  cat("\n")
  invisible(object)
}


#' Pretty Print a Chunk Iterator
#'
#' This function prints a summary of a chunk iterator.
#'
#' @param x A chunkiter object.
#' @param ... Additional arguments (ignored).
#' @export
#' @rdname print
print.chunkiter <- function(x, ...) {
  cat("Chunk Iterator\n")
  cat("  nchunks: ", x$nchunks, "\n", sep = "")
  invisible(x)
}

#' Pretty Print a Data Chunk Object
#'
#' This function prints a summary of a data chunk.
#'
#' @param x A data_chunk object.
#' @param ... Additional arguments (ignored).
#' @export
#' @rdname print
print.data_chunk <- function(x, ...) {
  cat("Data Chunk Object\n")

  # Handle both possible field names for chunk id
  chunk_id <- if (!is.null(x$chunkid)) x$chunkid else x$chunk_num
  total_chunks <- if (!is.null(x$nchunks)) x$nchunks else 1

  cat("  chunk ", chunk_id, " of ", total_chunks, "\n", sep = "")

  # Handle different possible field names
  if (!is.null(x$voxel_ind)) {
    cat("  Number of voxels:", length(x$voxel_ind), "\n")
  }
  if (!is.null(x$row_ind)) {
    cat("  Number of rows:", length(x$row_ind), "\n")
  }

  if (!is.null(x$data)) {
    if (!is.null(dim(x$data))) {
      cat("  Data dimensions:", paste(dim(x$data), collapse = " x "), "\n")
    } else {
      cat("  Data length:", length(x$data), "\n")
    }
  }

  invisible(x)
}

#' Helper function to print data source information
#' @keywords internal
#' @noRd
print_data_source_info <- function(x, full = FALSE) {
  if (inherits(x, "matrix_dataset")) {
    cat("  - Matrix:", nrow(x$datamat), "x", ncol(x$datamat), "(timepoints x voxels)\n")
  } else if (inherits(x, "fmri_mem_dataset")) {
    n_objects <- length(x$scans)
    cat("  - Objects:", n_objects, "pre-loaded NeuroVec object(s)\n")
  } else if (inherits(x, "fmri_file_dataset")) {
    if (!is.null(x$backend)) {
      # New backend-based dataset
      cat("  - Backend:", class(x$backend)[1], "\n")
      dims <- backend_get_dims(x$backend)
      if (full) {
        vox <- sum(backend_get_mask(x$backend))
      } else {
        vox <- "?"
      }
      cat(
        "  - Data dimensions:", dims$time, "x", vox,
        "(timepoints x voxels)\n"
      )
    } else {
      # Legacy file-based dataset
      n_files <- length(x$scans)
      cat("  - Files:", n_files, "NIfTI file(s)\n")
      if (n_files <= 3) {
        file_names <- basename(x$scans)
        cat("    ", paste(file_names, collapse = ", "), "\n")
      } else {
        file_names <- basename(x$scans)
        cat(
          "    ", paste(head(file_names, 2), collapse = ", "),
          ", ..., ", tail(file_names, 1), "\n"
        )
      }
    }
  }
}

#' @export
#' @rdname print
print.matrix_dataset <- function(x, ...) {
  # Use the generic fmri_dataset print method
  print.fmri_dataset(x, ...)
  invisible(x)
}
