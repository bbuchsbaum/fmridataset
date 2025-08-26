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
#' @importFrom fmrihrf acquisition_onsets
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

#' @rdname get_run_lengths
#' @method get_run_lengths sampling_frame
#' @export
get_run_lengths.sampling_frame <- function(x, ...) {
  x$blocklens
}

#' @rdname get_total_duration
#' @method get_total_duration sampling_frame
#' @export
get_total_duration.sampling_frame <- function(x, ...) {
  sum(x$blocklens * x$TR)
}

#' @rdname get_run_duration
#' @method get_run_duration sampling_frame
#' @export
get_run_duration.sampling_frame <- function(x, ...) {
  x$blocklens * x$TR
}

#' @rdname n_runs
#' @method n_runs sampling_frame
#' @export
n_runs.sampling_frame <- function(x, ...) {
  length(x$blocklens)
}

#' @rdname n_timepoints
#' @method n_timepoints sampling_frame
#' @export
n_timepoints.sampling_frame <- function(x, ...) {
  sum(x$blocklens)
}

#' @rdname get_TR
#' @method get_TR sampling_frame
#' @export
get_TR.sampling_frame <- function(x, ...) {
  # Always return the first TR value for compatibility
  # (fmrihrf supports per-block TR but our code expects single TR)
  x$TR[1]
}

# Explicit method definitions that delegate to fmrihrf
#' @rdname blocklens
#' @method blocklens sampling_frame
#' @export
blocklens.sampling_frame <- function(x, ...) {
  x$blocklens
}

#' @rdname blockids
#' @method blockids sampling_frame
#' @export
blockids.sampling_frame <- function(x, ...) {
  rep(seq_along(x$blocklens), times = x$blocklens)
}

#' @rdname samples
#' @method samples sampling_frame
#' @export
samples.sampling_frame <- function(x, ...) {
  # Implement samples method directly since fmrihrf method is not exported
  1:sum(x$blocklens)
}
