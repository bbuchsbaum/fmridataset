#' Convert an array source to a lazy delarr array
#'
#' `as_delarr()` wraps a serializable [array source][array-source] as a
#' `delarr` provider so that bounded, chunk-aware execution can be delegated
#' to `delarr` without materializing the assay. The realization budget is
#' enforced before any provider is created.
#'
#' @param x An array source, or another object with an `as_delarr()` method.
#' @param memory_budget Maximum realized bytes permitted for a single pull.
#'   `Inf` disables the check.
#' @param ... Additional arguments passed to methods.
#' @return A `delarr` lazy array whose pulls route through [source_read()].
#' @seealso [array-source] for the source protocol.
#' @examples
#' src <- memory_source(matrix(seq_len(6), nrow = 2))
#' lazy <- as_delarr(src)
#' dim(lazy)
#' @export
as_delarr <- function(x, memory_budget = Inf, ...) {
  UseMethod("as_delarr")
}

.ensure_delarr <- function() {
  if (!requireNamespace("delarr", quietly = TRUE)) {
    stop(
      "Package 'delarr' is required for as_delarr(). ",
      "Install it from https://bbuchsbaum.r-universe.dev.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' @rdname as_delarr
#' @export
as_delarr.default <- function(x, memory_budget = Inf, ...) {
  .frame_abort(
    sprintf(
      "No as_delarr method is registered for class '%s'.",
      class(x)[[1L]]
    ),
    "fmridataset_error_config",
    parameter = "x",
    value = class(x)
  )
}
