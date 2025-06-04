#' @importFrom utils head tail
#' @export
#' @rdname print
print.fmri_dataset <- function(x, ...) {
  # Header
  cat("\n=== fMRI Dataset ===\n")
  
  # Basic dimensions
  cat("\n** Dimensions:\n")
  cat("  - Timepoints:", sum(x$sampling_frame$run_length), "\n")
  cat("  - Runs:", x$nruns, "\n")
  
  # Data source info
  print_data_source_info(x)
  
  # Mask info
  mask <- get_mask(x)
  cat("  - Voxels in mask:", sum(mask > 0), "\n")
  cat("  - Mask dimensions:", paste(dim(mask), collapse=" x "), "\n")
  
  # Sampling frame info
  cat("\n** Temporal Structure:\n")
  cat("  - TR:", x$sampling_frame$TR, "seconds\n")
  cat("  - Run lengths:", paste(x$sampling_frame$run_length, collapse=", "), "\n")
  
  # Event table summary
  cat("\n** Event Table:\n")
  if (nrow(x$event_table) > 0) {
    cat("  - Rows:", nrow(x$event_table), "\n")
    cat("  - Variables:", paste(names(x$event_table), collapse=", "), "\n")
    
    # Show first few events if they exist
    if (nrow(x$event_table) > 0) {
      cat("  - First few events:\n")
      print(head(x$event_table, 3))
    }
  } else {
    cat("  - Empty event table\n")
  }
  
  cat("\n")
}

#' @export
#' @rdname print
print.latent_dataset <- function(x, ...) {
  # Header
  cat("\n=== Latent Dataset ===\n")
  
  # Basic dimensions
  cat("\n** Dimensions:\n")
  cat("  - Timepoints:", nrow(x$datamat), "\n")
  cat("  - Latent components:", ncol(x$datamat), "\n")
  cat("  - Runs:", x$nruns, "\n")
  
  # Original space info if available
  if (!is.null(x$original_space)) {
    cat("  - Original space:", paste(x$original_space, collapse=" x "), "\n")
  }
  
  # Sampling frame info
  cat("\n** Temporal Structure:\n")
  cat("  - TR:", x$sampling_frame$TR, "seconds\n")
  cat("  - Run lengths:", paste(x$sampling_frame$run_length, collapse=", "), "\n")
  
  # Event table summary
  cat("\n** Event Table:\n")
  if (nrow(x$event_table) > 0) {
    cat("  - Rows:", nrow(x$event_table), "\n")
    cat("  - Variables:", paste(names(x$event_table), collapse=", "), "\n")
    
    # Show first few events if they exist
    if (nrow(x$event_table) > 0) {
      cat("  - First few events:\n")
      print(head(x$event_table, 3))
    }
  } else {
    cat("  - Empty event table\n")
  }
  
  # Data summary
  cat("\n** Latent Data Summary:\n")
  data_summary <- summary(as.vector(x$datamat[1:min(1000, length(x$datamat))]))[c(1,3,4,6)]
  cat("  - Values (sample):", paste(names(data_summary), data_summary, sep=":", collapse=", "), "\n")
  
  cat("\n")
}

#' Pretty Print a Chunk Iterator
#'
#' This function prints a summary of a chunk iterator using colored output.
#'
#' @param x A chunkiter object.
#' @param ... Additional arguments (ignored).
#' @export
#' @rdname print
print.chunkiter <- function(x, ...) {
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("Please install the crayon package to use this function.")
  }
  cat(crayon::blue("Chunk Iterator:\n"))
  cat(crayon::magenta("  Total number of chunks: "), x$nchunks, "\n")
  invisible(x)
}

#' Pretty Print a Data Chunk Object
#'
#' This function prints a summary of a data chunk using crayon for colored output.
#'
#' @param x A data_chunk object.
#' @param ... Additional arguments (ignored).
#' @export
#' @rdname print
print.data_chunk <- function(x, ...) {
  if (!requireNamespace("crayon", quietly = TRUE)) {
    stop("Please install the crayon package to use this function.")
  }
  cat(crayon::blue("Data Chunk Object\n"))
  cat(crayon::magenta("  Chunk number: "), x$chunk_num, "\n")
  cat(crayon::magenta("  Number of voxels: "), length(x$voxel_ind), "\n")
  cat(crayon::magenta("  Number of rows: "), length(x$row_ind), "\n")
  if (!is.null(dim(x$data))) {
    cat(crayon::magenta("  Data dimensions: "), paste(dim(x$data), collapse = " x "), "\n")
  } else {
    cat(crayon::magenta("  Data: "), paste(head(x$data, 10), collapse = ", "), "\n")
  }
  invisible(x)
}

#' Helper function to print data source information
#' @keywords internal
#' @noRd
print_data_source_info <- function(x) {
  if (inherits(x, "matrix_dataset")) {
    cat("  - Matrix:", nrow(x$datamat), "x", ncol(x$datamat), "(timepoints x voxels)\n")
  } else if (inherits(x, "fmri_mem_dataset")) {
    n_objects <- length(x$scans)
    cat("  - Objects:", n_objects, "pre-loaded NeuroVec object(s)\n")
  } else if (inherits(x, "fmri_file_dataset")) {
    n_files <- length(x$scans)
    cat("  - Files:", n_files, "NIfTI file(s)\n")
    if (n_files <= 3) {
      file_names <- basename(x$scans)
      cat("    ", paste(file_names, collapse = ", "), "\n")
    } else {
      file_names <- basename(x$scans)
      cat("    ", paste(head(file_names, 2), collapse = ", "), 
          ", ..., ", tail(file_names, 1), "\n")
    }
  }
}

#' @export
#' @rdname print
print.matrix_dataset <- function(x, ...) {
  # Use the generic fmri_dataset print method
  print.fmri_dataset(x, ...)
} 