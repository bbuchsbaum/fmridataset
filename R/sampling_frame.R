#' Sampling Frame S3 Class for fMRI Data
#'
#' The `sampling_frame` class represents the temporal structure of fMRI data,
#' including repetition time (TR), run lengths, and derived temporal properties.
#' This is a core component of the `fmri_dataset` class.
#'
#' @section Structure:
#' A `sampling_frame` object contains:
#' \describe{
#'   \item{blocklens}{Numeric vector: Length of each block/run in timepoints (fmrireg compatibility)}
#'   \item{run_lengths}{Numeric vector: Alias for blocklens (fmridataset convention)}
#'   \item{TR}{Numeric: Repetition Time in seconds}
#'   \item{start_time}{Numeric: Offset of first scan of each block (default TR/2)}
#'   \item{precision}{Numeric: Discrete sampling interval for HRF convolution (default 0.1)}
#'   \item{total_timepoints}{Numeric: Total timepoints across all runs}
#'   \item{n_runs}{Integer: Number of runs}
#' }
#'
#' @name sampling_frame-class
#' @family sampling_frame
NULL

# Internal constructor - never exported
#' @noRd
#' @keywords internal
new_sampling_frame <- function(blocklens, TR, start_time, precision) {
  structure(
    list(
      blocklens = blocklens,           # fmrireg compatibility
      run_lengths = blocklens,         # fmridataset alias
      TR = TR,
      start_time = start_time,
      precision = precision,
      total_timepoints = sum(blocklens),
      n_runs = length(blocklens)
    ),
    class = "sampling_frame"
  )
}

#' Create a Sampling Frame Object
#'
#' Creates a `sampling_frame` object that encapsulates the temporal structure
#' of an fMRI experiment, including run lengths and repetition time.
#' Compatible with both fmrireg (blocklens) and fmridataset (run_lengths) conventions.
#'
#' @param blocklens A numeric vector representing the number of scans in each block/run.
#'   Can also be passed as `run_lengths` for fmridataset compatibility.
#' @param TR A numeric value or vector representing the repetition time in seconds.
#' @param start_time A numeric value or vector representing the offset of the first scan 
#'   of each block (default is TR/2).
#' @param precision A numeric value representing the discrete sampling interval used for 
#'   convolution with the hemodynamic response function (default is 0.1).
#' @param run_lengths Alternative parameter name for `blocklens` (fmridataset compatibility).
#'   If provided, takes precedence over `blocklens`.
#'
#' @return A `sampling_frame` object containing the temporal structure information.
#'
#' @examples
#' # fmrireg style
#' frame1 <- sampling_frame(blocklens = c(100, 100, 100), TR = 2, precision = 0.5)
#' 
#' # fmridataset style
#' frame2 <- sampling_frame(run_lengths = c(180, 200, 190), TR = 2.5)
#' 
#' # Single run with 200 timepoints
#' sf1 <- sampling_frame(200, TR = 2.0)
#' 
#' # Access properties
#' n_timepoints(frame1)
#' n_runs(frame2)
#' get_TR(sf1)
#'
#' @export
sampling_frame <- function(blocklens, TR, start_time = TR / 2, precision = 0.1, run_lengths = NULL) {
  
  # Handle fmridataset-style calling with run_lengths
  if (!missing(run_lengths) || (!missing(blocklens) && missing(TR) && is.list(blocklens))) {
    # If run_lengths is explicitly provided, use it
    if (!is.null(run_lengths)) {
      blocklens <- run_lengths
    }
    # Handle case where first argument might be run_lengths and second is TR
    # This supports: sampling_frame(c(100, 200), TR = 2)
  }
  
  # Input validation
  if (!is.numeric(blocklens) || length(blocklens) == 0) {
    stop("run_lengths cannot be empty")
  }
  if (any(blocklens <= 0)) {
    stop("run_lengths must be positive")
  }
  
  # Validate TR and run_lengths length compatibility
  if (length(TR) > 1 && length(TR) != length(blocklens)) {
    stop("Length of TR (", length(TR), ") must match length of run_lengths (", length(blocklens), ")")
  }
  
  # --- recycle & validate (fmrireg compatibility) ------------------------------------------------
  # Ensure all vectors have the same length
  max_len <- max(length(blocklens), length(TR), length(start_time))
  blocklens <- rep_len(blocklens, max_len)
  TR <- rep_len(TR, max_len)
  start_time <- rep_len(start_time, max_len)
  
  # Validate inputs with proper error messages
  if (!all(TR > 0)) {
    stop("TR values must be positive")
  }
  if (!all(start_time >= 0)) {
    stop("Start times must be non-negative")
  }
  if (precision <= 0) {
    stop("Precision must be positive")
  }
  if (precision >= min(TR)) {
    stop("Precision must be positive and less than the minimum TR")
  }
  
  # Convert to integers for blocklens (timepoints should be whole numbers)
  blocklens <- as.integer(round(blocklens))
  
  new_sampling_frame(blocklens, TR, start_time, precision)
}

#' Check if Object is a sampling_frame
#'
#' @param x Object to test
#' @return Logical indicating whether `x` is a `sampling_frame`
#' @export
is.sampling_frame <- function(x) {
  inherits(x, "sampling_frame")
}

#' Get samples from sampling frame
#'
#' @param x A `sampling_frame` object
#' @param blockids Optional vector of block IDs to include
#' @param global Logical indicating whether to return global times
#' @param ... Additional arguments
#' @return Numeric vector of sample times
#' @export
samples <- function(x, ...) {
  UseMethod("samples")
}

#' @export
samples.sampling_frame <- function(x, blockids = NULL, global = FALSE, ...) {
  if (is.null(blockids)) blockids <- seq_along(x$blocklens)

  # number of scans per selected block
  lens <- x$blocklens[blockids]

  # Return sequential timepoint indices starting from 1
  # For fmrireg compatibility, this should return 1:total_timepoints
  total_timepoints <- sum(lens)
  seq_len(total_timepoints)
}

#' Get global onsets
#'
#' @param x A `sampling_frame` object
#' @param onsets Vector of onset times
#' @param blockids Vector of block IDs
#' @param ... Additional arguments
#' @return Vector of global onset times
#' @export
global_onsets <- function(x, ...) {
  UseMethod("global_onsets")
}

#' @export
global_onsets.sampling_frame <- function(x, onsets = NULL, blockids = NULL, ...) {
  if (is.null(onsets) && is.null(blockids)) {
    # Return all timepoint onset times when called without arguments
    # This generates the time of each timepoint across all runs
    all_times <- numeric(0)
    cumulative_time <- 0
    
    for (i in seq_along(x$blocklens)) {
      # Times for this run
      run_times <- cumulative_time + x$start_time[i] + (0:(x$blocklens[i] - 1)) * x$TR[i]
      all_times <- c(all_times, run_times)
      
      # Update cumulative time for next run
      cumulative_time <- cumulative_time + x$blocklens[i] * x$TR[i]
    }
    
    return(all_times)
  }
  
  # Original functionality with arguments
  if (is.null(onsets) || is.null(blockids)) {
    stop("When providing arguments, both 'onsets' and 'blockids' must be specified")
  }
  
  # Calculate cumulative time offsets for each block
  block_durations <- x$blocklens * x$TR
  cumulative_time <- c(0, cumsum(block_durations))
  
  blockids <- as.integer(blockids)
  stopifnot(length(onsets) == length(blockids),
            all(blockids >= 1L), all(blockids <= length(x$blocklens)))

  onsets + cumulative_time[blockids]
}

#' Get block IDs
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments
#' @return Integer vector of block IDs
#' @export
blockids <- function(x, ...) {
  UseMethod("blockids")
}

#' @export
blockids.sampling_frame <- function(x, ...) {
  rep(seq_along(x$blocklens), times = x$blocklens)
}

#' Get block lengths from a sampling frame
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments
#' @return Numeric vector giving the number of scans in each block
#' @export
blocklens <- function(x, ...) {
  UseMethod("blocklens")
}

#' @export
blocklens.sampling_frame <- function(x, ...) {
  x$blocklens
}

#' Get Number of Timepoints
#'
#' Returns the total number of timepoints or timepoints for specific runs.
#'
#' @param x A `sampling_frame` object
#' @param run_id Optional integer vector specifying which runs to include.
#'   If NULL (default), returns total across all runs.
#' @param ... Additional arguments (not used)
#'
#' @return Integer: number of timepoints
#'
#' @examples
#' sf <- sampling_frame(c(100, 120, 110), TR = 2)
#' n_timepoints(sf)           # Total: 330
#' n_timepoints(sf, run_id = 1)     # Run 1: 100
#' n_timepoints(sf, run_id = c(1,3)) # Runs 1&3: 210
#'
#' @export
n_timepoints <- function(x, ...) {
  UseMethod("n_timepoints")
}

#' @export
n_timepoints.sampling_frame <- function(x, run_id = NULL, ...) {
  if (is.null(run_id)) {
    return(x$total_timepoints)
  } else {
    if (!all(run_id %in% 1:x$n_runs)) {
      stop("run_id values must be between 1 and ", x$n_runs)
    }
    return(sum(x$blocklens[run_id]))
  }
}

#' Get Number of Runs
#'
#' Returns the number of runs in the sampling frame.
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments (not used)
#'
#' @return Integer: number of runs
#'
#' @examples
#' sf <- sampling_frame(c(100, 120, 110), TR = 2)
#' n_runs(sf)  # Returns 3
#'
#' @export
n_runs <- function(x, ...) {
  UseMethod("n_runs")
}

#' @export
n_runs.sampling_frame <- function(x, ...) {
  x$n_runs
}

#' Get Repetition Time (TR)
#'
#' Returns the repetition time from a sampling frame.
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments (not used)
#'
#' @return Numeric: repetition time in seconds
#'
#' @examples
#' sf <- sampling_frame(200, TR = 2.5)
#' get_TR(sf)  # Returns 2.5
#'
#' @export
get_TR <- function(x, ...) {
  UseMethod("get_TR")
}

#' @export
get_TR.sampling_frame <- function(x, ...) {
  # Return TR for each run (replicated to match the number of runs)
  # This maintains fmrireg compatibility where TR can vary by run
  if (length(x$TR) == 1) {
    # Single TR value - replicate for each run
    rep(x$TR, x$n_runs)
  } else {
    # Already a vector - return as is
    x$TR
  }
}

#' Get Run Lengths
#'
#' Returns the vector of run lengths from a sampling frame.
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments (not used)
#'
#' @return Integer vector: length of each run in timepoints
#'
#' @examples
#' sf <- sampling_frame(c(100, 120, 110), TR = 2)
#' get_run_lengths(sf)  # Returns c(100, 120, 110)
#'
#' @export
get_run_lengths <- function(x, ...) {
  UseMethod("get_run_lengths")
}

#' @export
get_run_lengths.sampling_frame <- function(x, ...) {
  x$blocklens  # Correct field name in sampling_frame object
}

#' Get Total Duration
#'
#' Returns the total duration of the experiment in seconds.
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments (not used)
#'
#' @return Numeric: total duration in seconds
#'
#' @examples
#' sf <- sampling_frame(c(100, 120), TR = 2.0)
#' get_total_duration(sf)  # Returns 440 (220 timepoints * 2 seconds)
#'
#' @export
get_total_duration <- function(x, ...) {
  UseMethod("get_total_duration")
}

#' @export
get_total_duration.sampling_frame <- function(x, ...) {
  sum(x$blocklens * x$TR)
}

#' Get Run Duration
#'
#' Returns the duration of specific runs in seconds.
#'
#' @param x A `sampling_frame` object
#' @param run_id Optional integer vector specifying which runs. 
#'   If NULL, returns durations for all runs.
#' @param ... Additional arguments (not used)
#'
#' @return Numeric vector: duration of specified runs in seconds
#'
#' @examples
#' sf <- sampling_frame(c(100, 120, 110), TR = 2.0)
#' get_run_duration(sf)           # All runs: c(200, 240, 220)
#' get_run_duration(sf, run_id = 2)     # Run 2: 240
#'
#' @export
get_run_duration <- function(x, ...) {
  UseMethod("get_run_duration")
}

#' @export
get_run_duration.sampling_frame <- function(x, run_id = NULL, ...) {
  if (is.null(run_id)) {
    return(x$blocklens * x$TR)
  } else {
    if (!all(run_id %in% 1:x$n_runs)) {
      stop("run_id values must be between 1 and ", x$n_runs)
    }
    return(x$blocklens[run_id] * x$TR[run_id])
  }
}

#' Print Method for sampling_frame (fmrireg compatible)
#'
#' @param x A `sampling_frame` object
#' @param ... Additional arguments passed to print
#'
#' @export
print.sampling_frame <- function(x, ...) {
  n_blk <- length(x$blocklens)
  total_scans <- sum(x$blocklens)
  
  cat("sampling_frame object\n")
  cat("====================\n\n")
  
  cat("Structure:\n")
  cat(sprintf("  Runs: %d\n", n_blk))
  cat(sprintf("  Total timepoints: %d\n\n", total_scans))
  
  cat("Timing:\n")
  cat(sprintf("  TR: %s s\n", paste(unique(x$TR), collapse = ", ")))
  cat(sprintf("  Precision: %.3g s\n\n", x$precision))
  
  cat("Duration:\n")
  total_time <- sum(x$blocklens * x$TR)
  cat(sprintf("  Total time: %.1f s\n", total_time))
  
  invisible(x)
}

#' Summary Method for sampling_frame
#'
#' @param object A `sampling_frame` object
#' @param ... Additional arguments passed to summary
#'
#' @export
summary.sampling_frame <- function(object, ...) {
  cat("fMRI Sampling Frame Summary\n")
  cat("==========================\n")
  cat("Number of runs:", object$n_runs, "\n")
  cat("Total timepoints:", object$total_timepoints, "\n")
  cat("Repetition time (TR):", paste(unique(object$TR), collapse = ", "), "seconds\n")
  cat("Total duration:", get_total_duration(object), "seconds\n")
  cat("Precision:", object$precision, "seconds\n")
  
  if (object$n_runs > 1) {
    cat("\nRun length statistics:\n")
    cat("  Mean:", round(mean(object$blocklens), 1), "timepoints\n")
    cat("  Min:", min(object$blocklens), "timepoints\n")
    cat("  Max:", max(object$blocklens), "timepoints\n")
    cat("  SD:", round(sd(object$blocklens), 1), "timepoints\n")
  }
  
  invisible(object)
} 