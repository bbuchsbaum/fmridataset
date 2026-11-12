#' Convert an array source to a lazy delarr array
#'
#' `as_delarr()` wraps a serializable [array source][array-source] as a
#' `delarr` provider so that bounded, chunk-aware execution can be delegated
#' to `delarr` without materializing the assay. The realization budget bounds
#' each provider pull, not the whole source, so a lazy array may describe more
#' data than the budget allows; every pull is checked before it reads.
#'
#' @param x An array source, or another object with an `as_delarr()` method.
#' @param memory_budget Maximum realized bytes permitted for a single pull.
#'   `Inf` disables the check.
#' @param ... Additional arguments passed to methods.
#' @details
#' `delarr` is a suggested package. `as_delarr()` needs version 0.2.0 or newer
#' and fails with an error of class `fmridataset_error_backend_io` without it.
#' @return A `delarr` lazy array whose pulls route through [source_read()].
#' @seealso [array-source] for the source protocol.
#' @examples
#' if (requireNamespace("delarr", quietly = TRUE) &&
#'   utils::packageVersion("delarr") >= "0.2.0") {
#'   src <- memory_source(matrix(seq_len(6), nrow = 2))
#'   lazy <- as_delarr(src)
#'   dim(lazy)
#' }
#' @export
as_delarr <- function(x, memory_budget = Inf, ...) {
  UseMethod("as_delarr")
}

.delarr_min_version <- "0.2.0"

.delarr_available <- function() {
  requireNamespace("delarr", quietly = TRUE) &&
    utils::packageVersion("delarr") >= .delarr_min_version
}

.ensure_delarr <- function() {
  if (!.delarr_available()) {
    .frame_abort(
      paste0(
        "as_delarr() requires the delarr package, version ",
        .delarr_min_version, " or newer; install it from ",
        "https://bbuchsbaum.r-universe.dev."
      ),
      "fmridataset_error_backend_io",
      operation = "as_delarr"
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
