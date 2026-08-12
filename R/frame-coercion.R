#' Coerce an object to the canonical frame protocol
#'
#' Companion packages implement methods for legacy or domain-specific objects.
#' The generic is owned by `fmridataset` so conversions have one canonical
#' destination and do not introduce competing frame classes.
#'
#' @param x An object convertible to an `fmri_frame`.
#' @param ... Method-specific arguments.
#' @return An `fmri_frame`.
#' @export
as_fmri_frame <- function(x, ...) UseMethod("as_fmri_frame")

#' @export
as_fmri_frame.fmri_frame <- function(x, ...) x

#' @export
as_fmri_frame.default <- function(x, ...) {
  stop(
    "No as_fmri_frame method is registered for class '",
    class(x)[[1L]], "'.",
    call. = FALSE
  )
}
