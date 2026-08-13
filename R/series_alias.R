#' Deprecated alias for `fmri_series`
#'
#' `series()` forwards to [fmri_series()] for backward compatibility.
#' A deprecation warning is emitted via the lifecycle package.
#'
#' @inheritParams fmri_series
#' @return See [fmri_series()]
#' @export
series <- function(dataset, selector = NULL, timepoints = NULL,
                   output = c("fmri_series"),
                   event_window = NULL, ...) {
  lifecycle::deprecate_warn("0.3.0", "series()", "fmri_series()")
  fmri_series(dataset,
    selector = selector, timepoints = timepoints,
    output = output, event_window = event_window, ...
  )
}
