#' Adapter Methods for fmrihrf sampling_frame
#'
#' These methods provide compatibility between the local sampling_frame
#' implementation and the fmrihrf sampling_frame implementation.
#'
#' @name sampling_frame_adapters
#' @keywords internal
#' @importFrom fmrihrf sampling_frame
#' @importFrom fmrihrf blockids
#' @importFrom fmrihrf blocklens
#' @importFrom fmrihrf samples
#' @importFrom fmrihrf global_onsets
NULL

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
get_run_lengths.sampling_frame <- function(x, ...) {
  x$blocklens
}

#' @export
get_total_duration.sampling_frame <- function(x, ...) {
  sum(x$blocklens * x$TR)
}

#' @export
get_run_duration.sampling_frame <- function(x, ...) {
  x$blocklens * x$TR
}

#' @export
n_runs.sampling_frame <- function(x, ...) {
  length(x$blocklens)
}

#' @export
n_timepoints.sampling_frame <- function(x, ...) {
  sum(x$blocklens)
}

#' @export
get_TR.sampling_frame <- function(x, ...) {
  # Always return the first TR value for compatibility
  # (fmrihrf supports per-block TR but our code expects single TR)
  x$TR[1]
}

# Explicit method definitions that delegate to fmrihrf
#' @export
blocklens.sampling_frame <- function(x, ...) {
  x$blocklens
}

#' @export
blockids.sampling_frame <- function(x, ...) {
  rep(seq_along(x$blocklens), times = x$blocklens)
}

#' @export
samples.sampling_frame <- function(x, ...) {
  # Implement samples method directly since fmrihrf method is not exported
  1:sum(x$blocklens)
} 