#' Convert a serializable array source to a delarr plan
#'
#' The 1.0 core delegates lazy numerical execution to `delarr`. Backends with
#' open handles belonged to the 0.x architecture; canonical callers pass an
#' `ArraySource` descriptor, normally obtained from `assay(frame)$source`.
#'
#' @param backend An `array_source` descriptor.
#' @param ... Method-specific arguments.
#' @return A reconstructible `delarr` plan.
#' @export
as_delarr <- function(backend, ...) UseMethod("as_delarr")

.ensure_delarr <- function() {
  if (!requireNamespace("delarr", quietly = TRUE)) {
    stop("The delarr package is required for lazy matrix operations.", call. = FALSE)
  }
}

#' @rdname as_delarr
#' @export
as_delarr.default <- function(backend, ...) {
  stop(
    "No as_delarr method is registered for class '", class(backend)[1L],
    "'. Canonical execution requires an array_source descriptor.",
    call. = FALSE
  )
}
