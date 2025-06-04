#' Sampling Frame for fMRI Temporal Structure
#'
#' Creates and manipulates sampling frame objects that represent the temporal
#' structure of fMRI datasets. A sampling frame encapsulates run lengths,
#' repetition time (TR), and provides derived temporal properties.
#'
#' @param run_length A numeric vector of run lengths (number of timepoints per run)
#' @param TR Repetition time in seconds
#' @return A sampling_frame object
#' @export
sampling_frame <- function(run_length, TR) {
  assertthat::assert_that(is.numeric(run_length), all(run_length > 0))
  assertthat::assert_that(is.numeric(TR), length(TR) == 1, TR > 0)
  
  structure(
    list(
      run_length = run_length,
      TR = TR
    ),
    class = "sampling_frame"
  )
}

#' Test if Object is a Sampling Frame
#'
#' This function tests whether an object is of class 'sampling_frame'.
#'
#' @param x An object to test
#' @return TRUE if x is a sampling_frame object, FALSE otherwise
#' @export
is.sampling_frame <- function(x) {
  inherits(x, "sampling_frame")
}

#' @export
get_TR.sampling_frame <- function(x, ...) {
  x$TR
}

#' @export
get_run_lengths.sampling_frame <- function(x, ...) {
  x$run_length
}

#' @export
n_runs.sampling_frame <- function(x, ...) {
  length(x$run_length)
}

#' @export
n_timepoints.sampling_frame <- function(x, ...) {
  sum(x$run_length)
}

#' @export
blocklens.sampling_frame <- function(x, ...) {
  x$run_length
}

#' @export
blockids.sampling_frame <- function(x, ...) {
  rep(1:length(x$run_length), times = x$run_length)
}

#' @export
samples.sampling_frame <- function(x, ...) {
  1:sum(x$run_length)
}

#' @export
global_onsets.sampling_frame <- function(x, ...) {
  if (length(x$run_length) == 1) {
    return(1)
  }
  c(1, cumsum(x$run_length[-length(x$run_length)]) + 1)
}

#' @export
get_total_duration.sampling_frame <- function(x, ...) {
  sum(x$run_length) * x$TR
}

#' @export
get_run_duration.sampling_frame <- function(x, ...) {
  x$run_length * x$TR
}

#' @export
print.sampling_frame <- function(x, ...) {
  cat("Sampling Frame:\n")
  cat("  TR:", x$TR, "seconds\n")
  cat("  Runs:", length(x$run_length), "\n")
  cat("  Run lengths:", paste(x$run_length, collapse = ", "), "\n")
  cat("  Total timepoints:", sum(x$run_length), "\n")
  cat("  Total duration:", get_total_duration(x), "seconds\n")
  invisible(x)
} 