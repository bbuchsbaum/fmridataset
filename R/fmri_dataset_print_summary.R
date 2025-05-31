#' Print and Summary Methods for fmri_dataset Objects
#'
#' This file implements user-friendly print and summary methods for `fmri_dataset`
#' objects. These methods provide clear, informative displays of dataset structure,
#' metadata, and key statistics.
#'
#' @name fmri_dataset_print_summary
NULL

# ============================================================================
# Ticket #20: Print Method for fmri_dataset
# ============================================================================

#' Print Method for fmri_dataset Objects
#'
#' **Ticket #20**: User-friendly print method that displays a concise summary
#' of an `fmri_dataset` object, including data source, dimensions, temporal
#' structure, and key metadata.
#'
#' @param x An `fmri_dataset` object
#' @param ... Additional arguments (ignored)
#' @return Invisibly returns the input object
#' 
#' @details
#' The print method displays:
#' \itemize{
#'   \item Dataset type and data source information
#'   \item Temporal structure (TR, runs, timepoints)
#'   \item Spatial information (voxels, mask status)
#'   \item Event table summary
#'   \item Preprocessing and loading options
#' }
#' 
#' @examples
#' \dontrun{
#' dataset <- fmri_dataset_create(matrix(rnorm(1000), 100, 10), TR = 2, run_lengths = 100)
#' print(dataset)
#' }
#' 
#' @export
#' @family fmri_dataset
print.fmri_dataset <- function(x, ...) {
  
  # Header with dataset type
  cat("\n═══ fMRI Dataset ═══\n")
  cat("📊 Type:", get_dataset_type(x), "\n")
  
  # Data source information
  cat("\n💾 Data Source:\n")
  print_data_source_info(x)
  
  # Temporal structure
  cat("\n⏱️  Temporal Structure:\n")
  print_temporal_info(x)
  
  # Spatial information
  cat("\n🧠 Spatial Information:\n")
  print_spatial_info(x)
  
  # Event information
  cat("\n📋 Events & Design:\n")
  print_event_info(x)
  
  # Additional options and metadata
  cat("\n⚙️  Options & Metadata:\n")
  print_options_info(x)
  
  cat("\n")
  invisible(x)
}

# ============================================================================
# Ticket #21: Summary Method for fmri_dataset  
# ============================================================================

#' Summary Method for fmri_dataset Objects
#'
#' **Ticket #21**: Detailed summary method that provides comprehensive statistics
#' and information about an `fmri_dataset` object, including data characteristics,
#' validation status, and memory usage.
#'
#' @param object An `fmri_dataset` object
#' @param include_data_stats Logical indicating whether to load data and compute
#'   statistics (default: FALSE, as this can be slow for large datasets)
#' @param validate Logical indicating whether to run validation checks (default: TRUE)
#' @param ... Additional arguments (ignored)
#' @return Invisibly returns the input object
#' 
#' @details
#' The summary method provides:
#' \itemize{
#'   \item Complete dataset overview with all metadata
#'   \item Detailed temporal and spatial statistics
#'   \item Event table analysis (if present)
#'   \item Validation report (if requested)
#'   \item Data statistics (if `include_data_stats = TRUE`)
#'   \item Memory usage and caching information
#' }
#' 
#' @examples
#' \dontrun{
#' dataset <- fmri_dataset_create(matrix(rnorm(1000), 100, 10), TR = 2, run_lengths = 100)
#' summary(dataset)
#' summary(dataset, include_data_stats = TRUE, validate = FALSE)
#' }
#' 
#' @export
#' @family fmri_dataset
summary.fmri_dataset <- function(object, include_data_stats = FALSE, validate = TRUE, ...) {
  
  cat("\n════════════════════════════════════════════════════════════════════════════════\n")
  cat("                              fMRI Dataset Summary                              \n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  # === BASIC INFORMATION ===
  cat("\n📊 BASIC INFORMATION\n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  summary_basic_info(object)
  
  # === TEMPORAL STRUCTURE ===
  cat("\n⏱️  TEMPORAL STRUCTURE\n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  summary_temporal_structure(object)
  
  # === SPATIAL INFORMATION ===
  cat("\n🧠 SPATIAL INFORMATION\n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  summary_spatial_info(object)
  
  # === EVENT TABLE ANALYSIS ===
  if (!is.null(object$event_table) && nrow(object$event_table) > 0) {
    cat("\n📋 EVENT TABLE ANALYSIS\n")
    cat("════════════════════════════════════════════════════════════════════════════════\n")
    
    summary_event_analysis(object)
  }
  
  # === VALIDATION STATUS ===
  if (validate) {
    cat("\n✅ VALIDATION REPORT\n")
    cat("════════════════════════════════════════════════════════════════════════════════\n")
    
    summary_validation_report(object)
  }
  
  # === DATA STATISTICS ===
  if (include_data_stats) {
    cat("\n📈 DATA STATISTICS\n")
    cat("════════════════════════════════════════════════════════════════════════════════\n")
    
    summary_data_statistics(object)
  }
  
  # === MEMORY & CACHING ===
  cat("\n🔧 MEMORY & CACHING\n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  summary_memory_info(object)
  
  # === METADATA DETAILS ===
  cat("\n📝 METADATA DETAILS\n")
  cat("════════════════════════════════════════════════════════════════════════════════\n")
  
  summary_metadata_details(object)
  
  cat("\n")
  invisible(object)
}

# ============================================================================
# Internal Helper Functions for Print Method
# ============================================================================

#' Print Data Source Information
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
print_data_source_info <- function(x) {
  dataset_type <- get_dataset_type(x)
  
  if (dataset_type %in% c("file_vec", "bids_file")) {
    n_files <- length(x$image_paths)
    cat("  • Files:", n_files, "NIfTI file(s)\n")
    
    if (n_files <= 3) {
      file_names <- basename(x$image_paths)
      cat("    ", paste(file_names, collapse = ", "), "\n")
    } else {
      file_names <- basename(x$image_paths)
      cat("    ", paste(head(file_names, 2), collapse = ", "), ", ..., ", tail(file_names, 1), "\n")
    }
    
  } else if (dataset_type %in% c("memory_vec", "bids_mem")) {
    n_objects <- length(x$image_objects)
    cat("  • Objects:", n_objects, "pre-loaded NeuroVec object(s)\n")
    
  } else if (dataset_type == "matrix") {
    cat("  • Matrix:", nrow(x$image_matrix), "×", ncol(x$image_matrix), "(timepoints × voxels)\n")
  }
  
  # Mask information
  if (!is.null(x$mask_path)) {
    cat("  • Mask: NIfTI file (", basename(x$mask_path), ")\n")
  } else if (!is.null(x$mask_object)) {
    cat("  • Mask: Pre-loaded NeuroVol object\n")
  } else if (!is.null(x$mask_vector)) {
    cat("  • Mask: Logical vector (", sum(x$mask_vector), "/", length(x$mask_vector), " voxels)\n")
  } else {
    cat("  • Mask: None (all voxels included)\n")
  }
}

#' Print Temporal Information
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
print_temporal_info <- function(x) {
  sf <- get_sampling_frame(x)
  
  cat("  • TR:", get_TR(sf), "seconds\n")
  cat("  • Runs:", n_runs(sf), "\n")
  cat("  • Total timepoints:", n_timepoints(sf), "\n")
  
  run_lengths <- get_run_lengths(sf)
  if (length(run_lengths) == 1) {
    cat("  • Run length:", run_lengths, "timepoints\n")
  } else if (length(run_lengths) <= 5) {
    cat("  • Run lengths:", paste(run_lengths, collapse = ", "), "timepoints\n")
  } else {
    cat("  • Run lengths:", paste(head(run_lengths, 3), collapse = ", "), "...", tail(run_lengths, 1), "timepoints\n")
  }
  
  total_duration <- get_total_duration(sf)
  cat("  • Total duration:", round(total_duration / 60, 1), "minutes\n")
}

#' Print Spatial Information
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
print_spatial_info <- function(x) {
  
  tryCatch({
    n_voxels <- get_num_voxels(x)
    if (is.na(n_voxels)) {
      cat("  • Voxels: Unknown (requires data loading)\n")
    } else {
      cat("  • Voxels:", n_voxels, "(after masking)\n")
    }
  }, error = function(e) {
    cat("  • Voxels: Unknown (error:", e$message, ")\n")
  })
  
  # Additional spatial info for matrix datasets
  if (get_dataset_type(x) == "matrix") {
    cat("  • Original dimensions:", ncol(x$image_matrix), "voxels\n")
  }
}

#' Print Event Information
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
print_event_info <- function(x) {
  event_table <- get_event_table(x)
  
  if (is.null(event_table) || nrow(event_table) == 0) {
    cat("  • Event table: None\n")
  } else {
    cat("  • Event table:", nrow(event_table), "events,", ncol(event_table), "variables\n")
    
    # Show variable names
    var_names <- names(event_table)
    if (length(var_names) <= 5) {
      cat("    Variables:", paste(var_names, collapse = ", "), "\n")
    } else {
      cat("    Variables:", paste(head(var_names, 4), collapse = ", "), "...\n")
    }
  }
  
  # Censoring information
  censor_vector <- get_censor_vector(x)
  if (is.null(censor_vector)) {
    cat("  • Censoring: None\n")
  } else {
    n_censored <- sum(!censor_vector)
    n_total <- length(censor_vector)
    pct_censored <- round(100 * n_censored / n_total, 1)
    cat("  • Censoring:", n_censored, "/", n_total, "timepoints (", pct_censored, "%)\n")
  }
}

#' Print Options Information
#' @param x fmri_dataset object
#' @keywords internal
#' @noRd
print_options_info <- function(x) {
  metadata <- get_metadata(x)
  
  # File options
  if (!is.null(metadata$file_options)) {
    fo <- metadata$file_options
    cat("  • File mode:", fo$mode %||% "normal", "\n")
    cat("  • Preloaded:", fo$preload %||% FALSE, "\n")
  }
  
  # Matrix options
  if (!is.null(metadata$matrix_options)) {
    mo <- metadata$matrix_options
    preprocessing <- c()
    if (isTRUE(mo$temporal_zscore)) preprocessing <- c(preprocessing, "temporal z-score")
    if (isTRUE(mo$voxelwise_detrend)) preprocessing <- c(preprocessing, "voxelwise detrend")
    
    if (length(preprocessing) > 0) {
      cat("  • Preprocessing:", paste(preprocessing, collapse = ", "), "\n")
    } else {
      cat("  • Preprocessing: None\n")
    }
  }
  
  # BIDS information
  if (!is.null(metadata$bids_info) && !is.null(metadata$bids_info$subject_id)) {
    bi <- metadata$bids_info
    cat("  • BIDS subject:", bi$subject_id %||% "unknown", "\n")
    if (!is.null(bi$task_id)) cat("  • BIDS task:", bi$task_id, "\n")
  }
}

# ============================================================================
# Internal Helper Functions for Summary Method
# ============================================================================

#' Summary Basic Information
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_basic_info <- function(object) {
  cat("Dataset Type        :", get_dataset_type(object), "\n")
  cat("Creation Time       :", Sys.time(), "\n")
  cat("R Session           :", R.version.string, "\n")
  
  # Source description if available
  if (!is.null(object$metadata$source_description)) {
    cat("Source Description  :", object$metadata$source_description, "\n")
  }
  
  # Base path for file-based datasets
  if (!is.null(object$metadata$base_path)) {
    cat("Base Path           :", object$metadata$base_path, "\n")
  }
}

#' Summary Temporal Structure
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_temporal_structure <- function(object) {
  sf <- get_sampling_frame(object)
  
  cat("Repetition Time (TR)    :", get_TR(sf), "seconds\n")
  cat("Number of Runs          :", n_runs(sf), "\n")
  cat("Total Timepoints        :", n_timepoints(sf), "\n")
  cat("Total Duration          :", round(get_total_duration(sf), 2), "seconds (", 
      round(get_total_duration(sf) / 60, 2), "minutes)\n")
  
  run_lengths <- get_run_lengths(sf)
  cat("\nRun Length Statistics:\n")
  cat("  Mean                  :", round(mean(run_lengths), 2), "timepoints\n")
  cat("  Standard Deviation    :", round(sd(run_lengths), 2), "timepoints\n")
  cat("  Min                   :", min(run_lengths), "timepoints\n")
  cat("  Max                   :", max(run_lengths), "timepoints\n")
  cat("  Individual Lengths    :", paste(run_lengths, collapse = ", "), "\n")
  
  # Sampling frame details
  cat("\nSampling Frame Details:\n")
  cat("  Start Time            :", paste(unique(sf$start_time), collapse = ", "), "seconds\n")
  cat("  Precision             :", sf$precision, "seconds\n")
}

#' Summary Spatial Information
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_spatial_info <- function(object) {
  
  tryCatch({
    n_voxels <- get_num_voxels(object)
    cat("Voxels (after masking)  :", n_voxels, "\n")
  }, error = function(e) {
    cat("Voxels (after masking)  : Unknown (", e$message, ")\n")
  })
  
  # Dataset-specific spatial info
  dataset_type <- get_dataset_type(object)
  
  if (dataset_type == "matrix") {
    cat("Original Matrix Dims    :", nrow(object$image_matrix), "×", ncol(object$image_matrix), "(time × voxels)\n")
    
  } else if (dataset_type %in% c("file_vec", "bids_file")) {
    cat("Number of Image Files   :", length(object$image_paths), "\n")
    
  } else if (dataset_type %in% c("memory_vec", "bids_mem")) {
    cat("Number of Image Objects :", length(object$image_objects), "\n")
  }
  
  # Mask information
  cat("\nMask Information:\n")
  if (!is.null(object$mask_path)) {
    cat("  Type                  : NIfTI file\n")
    cat("  Path                  :", object$mask_path, "\n")
  } else if (!is.null(object$mask_object)) {
    cat("  Type                  : Pre-loaded NeuroVol object\n")
  } else if (!is.null(object$mask_vector)) {
    mask_stats <- summary(as.numeric(object$mask_vector))
    cat("  Type                  : Logical vector\n")
    cat("  Length                :", length(object$mask_vector), "\n")
    cat("  TRUE voxels           :", sum(object$mask_vector), "\n")
    cat("  FALSE voxels          :", sum(!object$mask_vector), "\n")
  } else {
    cat("  Type                  : None (all voxels included)\n")
  }
}

#' Summary Event Analysis
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_event_analysis <- function(object) {
  event_table <- get_event_table(object)
  
  cat("Number of Events        :", nrow(event_table), "\n")
  cat("Number of Variables     :", ncol(event_table), "\n")
  cat("Variable Names          :", paste(names(event_table), collapse = ", "), "\n")
  
  # Onset analysis
  if ("onset" %in% names(event_table)) {
    onsets <- event_table$onset
    cat("\nOnset Statistics:\n")
    cat("  Min                   :", round(min(onsets, na.rm = TRUE), 2), "seconds\n")
    cat("  Max                   :", round(max(onsets, na.rm = TRUE), 2), "seconds\n")
    cat("  Mean                  :", round(mean(onsets, na.rm = TRUE), 2), "seconds\n")
    cat("  Median                :", round(median(onsets, na.rm = TRUE), 2), "seconds\n")
  }
  
  # Duration analysis
  if ("duration" %in% names(event_table)) {
    durations <- event_table$duration
    cat("\nDuration Statistics:\n")
    cat("  Min                   :", round(min(durations, na.rm = TRUE), 2), "seconds\n")
    cat("  Max                   :", round(max(durations, na.rm = TRUE), 2), "seconds\n")
    cat("  Mean                  :", round(mean(durations, na.rm = TRUE), 2), "seconds\n")
    cat("  Total                 :", round(sum(durations, na.rm = TRUE), 2), "seconds\n")
  }
  
  # Trial type analysis
  if ("trial_type" %in% names(event_table)) {
    trial_types <- table(event_table$trial_type)
    cat("\nTrial Type Counts:\n")
    for (i in seq_along(trial_types)) {
      cat("  ", names(trial_types)[i], ":", trial_types[i], "\n")
    }
  }
}

#' Summary Validation Report
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_validation_report <- function(object) {
  
  cat("Running validation checks...\n\n")
  
  tryCatch({
    validate_fmri_dataset(object, check_data_load = FALSE, verbose = FALSE)
    cat("✅ Basic validation     : PASSED\n")
    
    # Try more comprehensive validation
    tryCatch({
      validate_fmri_dataset(object, check_data_load = TRUE, verbose = FALSE)
      cat("✅ Full validation      : PASSED\n")
    }, error = function(e) {
      cat("⚠️  Full validation      : FAILED (", e$message, ")\n")
    })
    
  }, error = function(e) {
    cat("❌ Basic validation     : FAILED\n")
    cat("   Error                :", e$message, "\n")
  })
}

#' Summary Data Statistics
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_data_statistics <- function(object) {
  
  cat("Loading data for statistical analysis...\n")
  
  tryCatch({
    # Load a sample of data for statistics
    data_matrix <- get_data_matrix(object, apply_transformations = FALSE)
    
    # Basic statistics
    cat("\nData Matrix Statistics:\n")
    cat("  Dimensions            :", nrow(data_matrix), "×", ncol(data_matrix), "(time × voxels)\n")
    
    # Sample statistics
    sample_size <- min(10000, length(data_matrix))
    data_sample <- sample(as.vector(data_matrix), sample_size)
    stats <- summary(data_sample)
    
    cat("  Data Range            :", round(stats[1], 3), "to", round(stats[6], 3), "\n")
    cat("  Mean                  :", round(stats[4], 3), "\n")
    cat("  Median                :", round(stats[3], 3), "\n")
    cat("  Standard Deviation    :", round(sd(data_sample, na.rm = TRUE), 3), "\n")
    
    # Missing values
    n_missing <- sum(is.na(data_matrix))
    pct_missing <- round(100 * n_missing / length(data_matrix), 2)
    cat("  Missing Values        :", n_missing, "(", pct_missing, "%)\n")
    
    # Temporal statistics
    cat("\nTemporal Statistics:\n")
    temporal_means <- rowMeans(data_matrix, na.rm = TRUE)
    cat("  Temporal Mean Range   :", round(min(temporal_means, na.rm = TRUE), 3), "to", 
        round(max(temporal_means, na.rm = TRUE), 3), "\n")
    cat("  Temporal SD           :", round(sd(temporal_means, na.rm = TRUE), 3), "\n")
    
    # Spatial statistics
    cat("\nSpatial Statistics:\n")
    spatial_means <- colMeans(data_matrix, na.rm = TRUE)
    cat("  Spatial Mean Range    :", round(min(spatial_means, na.rm = TRUE), 3), "to", 
        round(max(spatial_means, na.rm = TRUE), 3), "\n")
    cat("  Spatial SD            :", round(sd(spatial_means, na.rm = TRUE), 3), "\n")
    
  }, error = function(e) {
    cat("❌ Error loading data for statistics:", e$message, "\n")
  })
}

#' Summary Memory Information
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_memory_info <- function(object) {
  
  # Object size
  obj_size <- object.size(object)
  cat("Object Size             :", format(obj_size, units = "auto"), "\n")
  
  # Cache information
  cache_names <- ls(object$data_cache)
  cat("Cached Items            :", length(cache_names), "\n")
  
  if (length(cache_names) > 0) {
    cat("Cache Contents          :", paste(cache_names, collapse = ", "), "\n")
    
    # Estimate cache size
    cache_sizes <- sapply(cache_names, function(name) {
      obj <- get(name, envir = object$data_cache)
      object.size(obj)
    })
    total_cache_size <- sum(cache_sizes)
    cat("Total Cache Size        :", format(total_cache_size, units = "auto"), "\n")
  }
  
  # Memory recommendations
  if (get_dataset_type(object) %in% c("file_vec", "bids_file")) {
    cat("\nMemory Notes:\n")
    cat("  • Data loaded on-demand (lazy loading)\n")
    cat("  • Use preload_data() for faster repeated access\n")
    cat("  • Cache persists for session lifetime\n")
  }
}

#' Summary Metadata Details
#' @param object fmri_dataset object
#' @keywords internal
#' @noRd
summary_metadata_details <- function(object) {
  metadata <- get_metadata(object)
  
  # Print structured metadata
  for (section in names(metadata)) {
    if (section %in% c("dataset_type", "TR")) next  # Already printed
    
    section_data <- metadata[[section]]
    if (is.null(section_data)) next
    
    cat(sprintf("%-20s: ", tools::toTitleCase(section)))
    
    if (is.list(section_data) && length(section_data) > 0) {
      cat("\n")
      for (item in names(section_data)) {
        if (!is.null(section_data[[item]])) {
          cat(sprintf("  %-18s: %s\n", item, format(section_data[[item]])))
        }
      }
    } else if (!is.null(section_data)) {
      cat(format(section_data), "\n")
    }
  }
}

# ============================================================================
# Utility Functions
# ============================================================================

#' Null-coalescing operator
#' @param x First value
#' @param y Default value if x is NULL
#' @return x if not NULL, otherwise y
#' @keywords internal
#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
} 