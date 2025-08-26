#' Helper Functions for Vignettes
#' 
#' @description
#' Internal functions to support vignette examples with synthetic data
#' and consistent demonstrations.
#' 
#' @keywords internal
#' @noRd

#' Generate Synthetic fMRI Data for Examples
#' 
#' @param n_timepoints Number of time points
#' @param n_voxels Number of voxels
#' @param n_active Number of active voxels with signal
#' @param activation_periods Time indices where activation occurs
#' @param signal_strength Strength of activation signal
#' @param seed Random seed for reproducibility
#' @return Matrix of synthetic fMRI data
#' @keywords internal
#' @export
#' @examples
#' \dontrun{
#' data <- generate_example_fmri_data(200, 1000)
#' }
generate_example_fmri_data <- function(n_timepoints = 200, 
                                       n_voxels = 1000,
                                       n_active = 100,
                                       activation_periods = NULL,
                                       signal_strength = 0.5,
                                       seed = 123) {
  set.seed(seed)
  
  # Create base noise
  data_matrix <- matrix(rnorm(n_timepoints * n_voxels), 
                        nrow = n_timepoints, 
                        ncol = n_voxels)
  
  # Add activation if specified
  if (!is.null(activation_periods) && n_active > 0) {
    active_voxels <- seq_len(min(n_active, n_voxels))
    data_matrix[activation_periods, active_voxels] <- 
      data_matrix[activation_periods, active_voxels] + signal_strength
  }
  
  # Add some temporal autocorrelation for realism
  for (v in seq_len(n_voxels)) {
    data_matrix[, v] <- stats::filter(data_matrix[, v], 
                                      filter = c(0.3, 0.4, 0.3), 
                                      sides = 1,
                                      circular = TRUE)
  }
  
  data_matrix
}

#' Generate Example Event Table
#' 
#' @param n_runs Number of runs
#' @param events_per_run Events per run
#' @param TR Repetition time
#' @param run_length Length of each run in TRs
#' @return Data frame with event information
#' @keywords internal
#' @export
generate_example_events <- function(n_runs = 2, 
                                   events_per_run = 4,
                                   TR = 2.0,
                                   run_length = 100) {
  events <- data.frame()
  
  for (run in seq_len(n_runs)) {
    run_start <- (run - 1) * run_length * TR
    
    # Create evenly spaced events within run
    event_spacing <- (run_length * TR) / (events_per_run + 1)
    onsets <- run_start + seq(event_spacing, 
                              by = event_spacing, 
                              length.out = events_per_run)
    
    run_events <- data.frame(
      onset = onsets,
      duration = rep(10, events_per_run),  # 10 second events
      trial_type = rep(c("task_A", "task_B"), length.out = events_per_run),
      run = run
    )
    
    events <- rbind(events, run_events)
  }
  
  events
}

#' Create Example File Paths for Vignettes
#' 
#' @param n_runs Number of runs
#' @param subject_id Subject identifier
#' @param base_path Base directory path
#' @return Character vector of file paths
#' @keywords internal
#' @export
generate_example_paths <- function(n_runs = 2, 
                                  subject_id = "sub-001",
                                  base_path = tempdir()) {
  file.path(base_path, 
           sprintf("%s_run-%02d_bold.nii.gz", subject_id, seq_len(n_runs)))
}

#' Create Example Mask
#' 
#' @param n_voxels Total number of voxels
#' @param fraction_valid Fraction of voxels to include in mask
#' @return Logical vector mask
#' @keywords internal
#' @export
generate_example_mask <- function(n_voxels = 1000, fraction_valid = 0.8) {
  n_valid <- round(n_voxels * fraction_valid)
  mask <- logical(n_voxels)
  mask[seq_len(n_valid)] <- TRUE
  mask
}

#' Generate Performance Benchmark Data
#' 
#' @param dataset_sizes Vector of dataset sizes to test
#' @param operations Operations to benchmark
#' @return Data frame with benchmark results
#' @keywords internal
#' @export
generate_benchmark_data <- function(dataset_sizes = c(100, 500, 1000, 5000),
                                   operations = c("load", "chunk", "process")) {
  results <- expand.grid(
    size = dataset_sizes,
    operation = operations,
    stringsAsFactors = FALSE
  )
  
  # Simulate benchmark times (proportional to size)
  results$time_ms <- results$size * switch(
    as.character(results$operation[1]),
    load = 0.5,
    chunk = 0.1,
    process = 1.0
  ) + rnorm(nrow(results), 0, 5)
  
  results$time_ms <- pmax(results$time_ms, 1)  # Ensure positive times
  results
}

#' Print Formatted Dataset Information
#' 
#' @param dataset Dataset object
#' @param title Optional title
#' @keywords internal
#' @export
print_dataset_info <- function(dataset, title = NULL) {
  if (!is.null(title)) {
    cat("\n", title, "\n", strrep("-", nchar(title)), "\n", sep = "")
  }
  
  cat("Dataset class:", paste(class(dataset), collapse = ", "), "\n")
  
  if (!is.null(dataset$sampling_frame)) {
    cat("TR:", get_TR(dataset), "seconds\n")
    cat("Number of runs:", n_runs(dataset), "\n")
    cat("Run lengths:", get_run_lengths(dataset$sampling_frame), "\n")
    cat("Total timepoints:", n_timepoints(dataset), "\n")
  }
  
  if (!is.null(dataset$event_table) && nrow(dataset$event_table) > 0) {
    cat("Events:", nrow(dataset$event_table), "events across", 
        length(unique(dataset$event_table$run)), "runs\n")
  }
  
  invisible(dataset)
}

#' Simulate Analysis Function for Examples
#' 
#' @param data Data matrix
#' @param method Analysis method
#' @return Numeric vector of results
#' @keywords internal
#' @export
analyze_run <- function(data, method = "mean") {
  switch(method,
    mean = colMeans(data),
    var = apply(data, 2, var),
    max = apply(data, 2, max),
    colMeans(data)  # default
  )
}