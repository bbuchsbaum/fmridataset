#' Check that an array source obeys the source protocol
#'
#' `check_array_source()` runs the behavioural laws every array source must
#' satisfy against a source and the matrix it is supposed to hold. Storage
#' packages call it in their tests for every source type they ship; the
#' package's own sources pass the same checks.
#'
#' The laws are:
#' \describe{
#'   \item{descriptor}{[validate_array_source()] accepts the source; its shape
#'     matches `reference`; the dtype is supported; the chunk grid is bounded
#'     by the shape; the capabilities include `"block_slice"` and
#'     `"serializable"`; the fingerprint is non-empty; and the descriptor holds
#'     no runtime state.}
#'   \item{serialization}{A `serialize()`/`unserialize()` round trip keeps the
#'     fingerprint, shape, dtype, and chunks.}
#'   \item{lifecycle}{[source_open()] returns an `array_source_handle`,
#'     [source_close()] on it is silent, and neither changes the fingerprint.}
#'   \item{reads}{Reads match `reference` for corner, reversed, permuted, and
#'     full selections, through the descriptor and through an open handle, and
#'     repeating a read gives the same values.}
#'   \item{selectors}{Logical masks and negative positions read like the
#'     equivalent positions; empty selections keep two dimensions; repeated
#'     and out-of-range positions are refused.}
#'   \item{views}{[source_view()] selections, nested in any order, read
#'     exactly what the equivalent flat positions read.}
#'   \item{content}{[content_hash()] equals the hash of an in-memory copy of
#'     `reference` and leaves the fingerprint unchanged.}
#' }
#'
#' Values are compared with [all.equal()], ignoring dimnames, so a float32
#' source is compared with `reference` at float32 precision only if
#' `reference` already holds float32-representable values.
#'
#' @param source An array source.
#' @param reference A matrix holding the values `source` describes.
#' @param error If `TRUE` (the default), raise an error of class
#'   `fmridataset_error_source_contract` naming every failed law. If `FALSE`,
#'   return the report without raising.
#' @return A `source_conformance` data frame with one row per law and columns
#'   `law`, `passed`, and `detail`, invisibly when every law passes. It prints
#'   as a checklist.
#' @examples
#' reference <- matrix(as.double(1:12), nrow = 3)
#' check_array_source(memory_source(reference), reference)
#' @export
check_array_source <- function(source, reference, error = TRUE) {
  reference <- unname(as.matrix(reference))
  laws <- list(
    descriptor = function() .conformance_descriptor(source, reference),
    serialization = function() .conformance_serialization(source),
    lifecycle = function() .conformance_lifecycle(source),
    reads = function() .conformance_reads(source, reference),
    selectors = function() .conformance_selectors(source, reference),
    views = function() .conformance_views(source, reference),
    content = function() .conformance_content(source, reference)
  )
  results <- lapply(laws, function(law) {
    tryCatch(law(), error = function(e) paste("error:", conditionMessage(e)))
  })
  report <- data.frame(
    law = names(laws),
    passed = vapply(results, isTRUE, logical(1)),
    detail = vapply(results, function(r) if (isTRUE(r)) "" else as.character(r)[[1L]], ""),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
  class(report) <- c("source_conformance", class(report))
  if (isTRUE(error) && !all(report$passed)) {
    failed <- report[!report$passed, , drop = FALSE]
    .frame_abort(
      paste0(
        "Array source of class '", class(source)[[1L]], "' fails ",
        nrow(failed), " conformance law", if (nrow(failed) > 1L) "s", ":\n",
        paste0("  ", failed$law, ": ", failed$detail, collapse = "\n")
      ),
      "fmridataset_error_source_contract",
      field = "conformance",
      report = report
    )
  }
  if (all(report$passed)) invisible(report) else report
}

#' @export
print.source_conformance <- function(x, ...) {
  .line(
    "<source_conformance>", sum(x$passed), "of", nrow(x),
    "laws pass"
  )
  width <- max(nchar(x$law))
  for (i in seq_len(nrow(x))) {
    .line(if (x$passed[[i]]) {
      paste("  ok  ", x$law[[i]])
    } else {
      sprintf("  FAIL %-*s  %s", width, x$law[[i]], x$detail[[i]])
    })
  }
  invisible(x)
}

# Each law returns TRUE, or a one-line description of the first violation.

.conformance_same <- function(actual, expected, what) {
  if (isTRUE(all.equal(actual, expected, check.attributes = FALSE))) {
    return(TRUE)
  }
  sprintf("%s does not match the reference.", what)
}

.conformance_first_failure <- function(...) {
  for (result in list(...)) {
    if (!isTRUE(result)) {
      return(result)
    }
  }
  TRUE
}

.conformance_descriptor <- function(source, reference) {
  if (!inherits(source, "array_source")) {
    return("the object does not inherit from 'array_source'.")
  }
  validate_array_source(source)
  shape <- source_shape(source)
  chunks <- source_chunks(source)
  capabilities <- source_capabilities(source)
  .conformance_first_failure(
    if (identical(shape, as.integer(dim(reference)))) TRUE else
      sprintf("shape %s does not match the reference's %s.",
        paste(shape, collapse = " x "), paste(dim(reference), collapse = " x ")),
    if (source_dtype(source) %in% .supported_source_dtypes) TRUE else
      sprintf("dtype '%s' is not supported.", source_dtype(source)),
    if (length(chunks) == 2L && all(chunks > 0L) && all(chunks <= pmax(1L, shape))) TRUE else
      "chunks are not two positive integers bounded by the shape.",
    if (all(c("block_slice", "serializable") %in% capabilities)) TRUE else
      "capabilities lack 'block_slice' or 'serializable'.",
    if (!anyDuplicated(capabilities)) TRUE else "capabilities repeat an entry.",
    if (nzchar(source_fingerprint(source))) TRUE else "the fingerprint is empty.",
    if (!.source_contains_runtime_state(source)) TRUE else
      "the descriptor holds runtime state (a function, environment, or pointer)."
  )
}

.conformance_serialization <- function(source) {
  restored <- unserialize(serialize(source, NULL))
  .conformance_first_failure(
    if (identical(source_fingerprint(restored), source_fingerprint(source))) TRUE else
      "the fingerprint changes across serialize()/unserialize().",
    if (identical(source_shape(restored), source_shape(source))) TRUE else
      "the shape changes across serialize()/unserialize().",
    if (identical(source_dtype(restored), source_dtype(source))) TRUE else
      "the dtype changes across serialize()/unserialize().",
    if (identical(source_chunks(restored), source_chunks(source))) TRUE else
      "the chunks change across serialize()/unserialize()."
  )
}

.conformance_lifecycle <- function(source) {
  fingerprint <- source_fingerprint(source)
  handle <- source_open(source)
  if (!inherits(handle, "array_source_handle")) {
    return("source_open() does not return an 'array_source_handle'.")
  }
  noise <- utils::capture.output(
    warnings <- tryCatch(
      {
        source_close(handle)
        character()
      },
      warning = function(w) conditionMessage(w)
    )
  )
  .conformance_first_failure(
    if (!length(warnings) && !length(noise)) TRUE else
      "source_close() on a handle is not silent.",
    if (identical(source_fingerprint(source), fingerprint)) TRUE else
      "opening and closing changes the fingerprint."
  )
}

.conformance_reads <- function(source, reference) {
  n_row <- nrow(reference)
  n_col <- ncol(reference)
  if (!n_row || !n_col) {
    return(.conformance_same(dim(source_read(source)), dim(reference), "An empty read's shape"))
  }
  rows <- unique(c(n_row, 1L))
  cols <- unique(c(n_col, 1L))
  checks <- list(
    .conformance_same(source_read(source, rows, cols), reference[rows, cols, drop = FALSE], "A corner read"),
    .conformance_same(
      source_read(source, rev(rows), rev(cols)),
      reference[rev(rows), rev(cols), drop = FALSE], "A reversed read"
    ),
    .conformance_same(source_read(source), reference, "A full read"),
    if (identical(source_read(source, rows, cols), source_read(source, rows, cols))) TRUE else
      "two identical reads return different values."
  )
  if (n_row > 2L && n_col > 2L) {
    perm_rows <- c(2L, n_row, 1L)
    perm_cols <- c(n_col, 2L, 1L)
    checks <- c(checks, list(.conformance_same(
      source_read(source, perm_rows, perm_cols),
      reference[perm_rows, perm_cols, drop = FALSE], "A permuted read"
    )))
  }
  handle <- source_open(source)
  on.exit(source_close(handle), add = TRUE)
  checks <- c(checks, list(.conformance_same(
    source_read(handle, rev(rows), cols),
    reference[rev(rows), cols, drop = FALSE], "A read through an open handle"
  )))
  do.call(.conformance_first_failure, checks)
}

.conformance_refuses <- function(expr, what) {
  refused <- tryCatch(
    {
      force(expr)
      FALSE
    },
    error = function(e) TRUE
  )
  if (refused) TRUE else sprintf("%s is accepted instead of refused.", what)
}

.conformance_selectors <- function(source, reference) {
  n_row <- nrow(reference)
  n_col <- ncol(reference)
  checks <- list(
    .conformance_same(dim(source_read(source, integer(), NULL)), c(0L, n_col), "An empty observation selection's shape"),
    .conformance_same(dim(source_read(source, NULL, integer())), c(n_row, 0L), "An empty feature selection's shape"),
    .conformance_same(dim(source_read(source, integer(), integer())), c(0L, 0L), "An empty selection's shape"),
    .conformance_refuses(source_read(source, n_row + 1L, NULL), "An out-of-range observation"),
    .conformance_refuses(source_read(source, NULL, n_col + 1L), "An out-of-range feature")
  )
  if (n_row && n_col) {
    mask <- rep_len(c(TRUE, FALSE), n_row)
    checks <- c(checks, list(
      .conformance_same(source_read(source, mask, NULL), reference[mask, , drop = FALSE], "A logical-mask read"),
      .conformance_same(source_read(source, NULL, -1L), reference[, -1L, drop = FALSE], "A negative-position read"),
      .conformance_refuses(source_read(source, c(1L, 1L), NULL), "A repeated observation")
    ))
  }
  do.call(.conformance_first_failure, checks)
}

.conformance_views <- function(source, reference) {
  n_row <- nrow(reference)
  n_col <- ncol(reference)
  if (n_row < 3L || n_col < 3L) {
    return(TRUE)
  }
  inner_rows <- 2:n_row
  inner_cols <- seq_len(n_col - 1L)
  ranged <- source_view(source, observations = inner_rows, features = inner_cols)
  picked <- source_view(ranged, observations = c(n_row - 1L, 1L))
  picked_rows <- inner_rows[c(n_row - 1L, 1L)]
  deeper <- source_view(picked, observations = 2:1, features = c(n_col - 1L, 1L))
  .conformance_first_failure(
    .conformance_same(source_read(ranged), reference[inner_rows, inner_cols, drop = FALSE], "A ranged view"),
    .conformance_same(source_read(picked), reference[picked_rows, inner_cols, drop = FALSE], "A view of a view"),
    .conformance_same(
      source_read(deeper),
      reference[picked_rows[2:1], c(n_col - 1L, 1L), drop = FALSE], "A three-level view"
    )
  )
}

.conformance_content <- function(source, reference) {
  fingerprint <- source_fingerprint(source)
  content <- content_hash(source)
  .conformance_first_failure(
    if (identical(content, content_hash(memory_source(reference)))) TRUE else
      "content_hash() differs from the hash of an in-memory copy of the reference.",
    if (identical(source_fingerprint(source), fingerprint)) TRUE else
      "content_hash() changes the fingerprint."
  )
}
