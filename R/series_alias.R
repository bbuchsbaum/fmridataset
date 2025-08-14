#' Deprecated alias for `fmri_series`
#'
#' `series()` forwards to [fmri_series()] for backward compatibility.
#' A deprecation warning is emitted once per session.
#'
#' @inheritParams fmri_series
#' @return See [fmri_series()]
#' @export
series <- function(dataset, selector = NULL, timepoints = NULL,
                   output = c("fmri_series", "DelayedMatrix"),
                   event_window = NULL, ...) {
  # Force immediate warning for testing
  warning("series() was deprecated in fmridataset 0.3.0.\nPlease use fmri_series() instead.", 
          call. = FALSE, immediate. = TRUE)
  fmri_series(dataset, selector = selector, timepoints = timepoints,
              output = output, event_window = event_window, ...)
}
