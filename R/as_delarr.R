#' Convert an array source to a lazy delarr array
#'
#' `as_delarr()` wraps a serializable [array source][array-source] as a
#' `delarr` provider so that bounded, chunk-aware execution can be delegated
#' to `delarr` without materializing the assay. Wrapping reads nothing and
#' is never refused on the size of the whole array: the realization budget
#' is attached to the provider and enforced on every pull, so a source far
#' larger than `memory_budget` can be wrapped and consumed in bounded blocks,
#' while any single pull whose estimated peak exceeds the budget raises
#' `fmridataset_error_budget` before it reads.
#'
#' @param x An array source, or another object with an `as_delarr()` method.
#' @param memory_budget Maximum estimated peak bytes permitted for a single
#'   pull, as estimated by [source_realization_cost()] for the pulled
#'   selection. `Inf` disables the check. The whole array is not budgeted.
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
