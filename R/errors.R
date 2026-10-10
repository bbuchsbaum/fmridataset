#' Custom Error Classes for fmridataset
#'
#' @description
#' Errors raised by fmridataset are conditions of class `fmridataset_error`
#' with a more specific subclass such as `fmridataset_error_alignment`,
#' `fmridataset_error_space_mismatch`, or `fmridataset_error_backend_io`.
#' Structured fields on the condition describe what was rejected.
#'
#' @name fmridataset-errors
#' @keywords internal
NULL

# Construct a structured fmridataset condition. `class` names the specific
# subclass; `...` supplies structured fields carried on the condition.
fmridataset_error <- function(message, class = character(), ...) {
  structure(
    list(message = message, ...),
    class = c(class, "fmridataset_error", "error", "condition")
  )
}
