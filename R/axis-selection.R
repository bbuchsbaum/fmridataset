# The selection algebra.
#
# Every axis selection in the package -- a frame or view subset, a source view,
# a collection subset, an entity or axis frame subset, a study entity filter,
# and the selector arguments of source_read() -- is normalized through
# .normalize_selection() into one internal type, `axis_selection`, and is
# composed, stored, fingerprinted, and read from that type. The type has three
# canonical forms:
#
#   all        every element of an axis of length n, in order; stores no vector
#   range      one contiguous ascending run start..end; stores two integers
#   positions  an explicit unique, order-preserving integer vector, which is
#              also the form of an empty selection
#
# Construction canonicalizes: positions that spell out the whole axis become
# `all`, and positions that form one ascending run become `range`, so equal
# selections have equal descriptors regardless of how a caller expressed them.
# A selection therefore never scales with the axis length unless the caller
# actually enumerated an arbitrary subset, and nested selections compose
# without materializing the axis they pass through. The policy is recorded in
# inst/architecture/ADR-010-selection-algebra.md.

.selection_forms <- c("all", "range", "positions")

.new_axis_selection <- function(form, n, start = NULL, end = NULL,
                                positions = NULL) {
  out <- list(form = form, n = as.integer(n))
  if (identical(form, "range")) {
    out$start <- as.integer(start)
    out$end <- as.integer(end)
  } else if (identical(form, "positions")) {
    out$positions <- as.integer(positions)
  }
  structure(out, class = "axis_selection")
}

.selection_all <- function(n) .new_axis_selection("all", n)

.selection_range <- function(n, start, end) {
  n <- as.integer(n)
  start <- as.integer(start)
  end <- as.integer(end)
  if (end < start) {
    return(.new_axis_selection("positions", n, positions = integer()))
  }
  if (start == 1L && end == n) {
    return(.selection_all(n))
  }
  .new_axis_selection("range", n, start = start, end = end)
}

# Canonicalizing constructor for validated positions: unique, in bounds.
.selection_positions <- function(n, positions) {
  n <- as.integer(n)
  positions <- as.integer(positions)
  len <- length(positions)
  if (len && !is.unsorted(positions, strictly = TRUE) &&
    positions[[len]] - positions[[1L]] == len - 1L) {
    return(.selection_range(n, positions[[1L]], positions[[len]]))
  }
  .new_axis_selection("positions", n, positions = positions)
}

.is_axis_selection <- function(x) inherits(x, "axis_selection")

.selection_length <- function(x) {
  switch(x$form,
    all = x$n,
    range = x$end - x$start + 1L,
    positions = length(x$positions)
  )
}

.selection_is_all <- function(x) identical(x$form, "all")

# Explicit positions. `all` and `range` expand to compact base-R sequences, so
# this is cheap to call but is still the one place an axis is spelled out;
# prefer .selection_index() when handing a selection to a source.
.selection_expand <- function(x) {
  switch(x$form,
    all = seq_len(x$n),
    range = seq.int(x$start, x$end),
    positions = x$positions
  )
}

# The selector a child source receives: NULL means "everything" and is the
# select-all pushdown every source already understands.
.selection_index <- function(x) {
  if (identical(x$form, "all")) NULL else .selection_expand(x)
}

.selection_element <- function(x, i) {
  switch(x$form,
    all = as.integer(i),
    range = x$start + as.integer(i) - 1L,
    positions = x$positions[[i]]
  )
}

# Subset a vector or list by a selection without expanding `all`.
.selection_subset <- function(values, x) {
  if (identical(x$form, "all")) values else values[.selection_expand(x)]
}

# Resolve `outer`, a selection over the axis that `inner` presents, into one
# selection over the axis `inner` selects from. Neither `all` nor `range`
# is materialized on the way through.
.selection_compose <- function(outer, inner) {
  if (!identical(outer$n, .selection_length(inner))) {
    .frame_abort(
      "A nested selection must address the axis its parent presents.",
      "fmridataset_error_alignment",
      reason = "compose_length",
      expected = .selection_length(inner),
      actual = outer$n
    )
  }
  if (identical(outer$form, "all")) {
    return(inner)
  }
  if (identical(inner$form, "all")) {
    return(outer)
  }
  if (identical(inner$form, "range")) {
    if (identical(outer$form, "range")) {
      return(.selection_range(
        inner$n,
        inner$start + outer$start - 1L,
        inner$start + outer$end - 1L
      ))
    }
    return(.selection_positions(inner$n, inner$start + outer$positions - 1L))
  }
  .selection_positions(inner$n, inner$positions[.selection_expand(outer)])
}

# The serializable, canonical form: what fingerprints and plans hash.
.selection_descriptor <- function(x) unclass(x)

# Selected positions in ascending order, and the contiguous runs they form.
# Backends that read rectangular chunks (Zarr) consume the runs directly; the
# `all` and `range` forms are one run without any sorting or diffing.
.selection_sorted <- function(x) {
  if (identical(x$form, "positions")) sort(x$positions) else .selection_expand(x)
}

.selection_runs <- function(x) {
  if (identical(x$form, "all")) {
    return(if (x$n) list(c(1L, x$n)) else list())
  }
  if (identical(x$form, "range")) {
    return(list(c(x$start, x$end)))
  }
  sorted <- sort(x$positions)
  if (!length(sorted)) {
    return(list())
  }
  breaks <- which(diff(sorted) != 1L)
  starts <- sorted[c(1L, breaks + 1L)]
  ends <- sorted[c(breaks, length(sorted))]
  Map(function(start, end) c(start, end), starts, ends)
}

# Bounded summary for explain(): the form and extent, never the vector.
.selection_summary <- function(x) {
  out <- list(form = x$form, count = .selection_length(x), axis_length = x$n)
  if (identical(x$form, "range")) {
    out$start <- x$start
    out$end <- x$end
  }
  out
}

# Selector pushdown declarations.
#
# A source declares which selector forms it consumes natively as capability
# strings "pushdown:<form>". Everything else it receives is emulated (Zarr, for
# example, decomposes arbitrary positions into range reads). The declaration is
# inspectable through source_capabilities(); absence means the source has not
# certified any form and the package never relies on it for correctness.
.pushdown_capabilities <- function(forms = .selection_forms) {
  forms <- match.arg(forms, .selection_forms, several.ok = TRUE)
  paste0("pushdown:", forms)
}

.source_pushdown_forms <- function(x) {
  capabilities <- source_capabilities(x)
  sub("^pushdown:", "", grep("^pushdown:", capabilities, value = TRUE))
}

# One normalization law -------------------------------------------------------
#
# `index`  the caller's selector: missing/NULL, an axis_selection, character
#          stable IDs, a logical mask, or numeric positions
# `n`      the axis length
# `ids`    the axis IDs, required only to resolve character selectors; a
#          positional axis (a raw source) passes NULL and refuses IDs
# `axis`   a label for messages ("observation", "feature", "frame", ...)
# `abort`  function(message, ...) raising the owner's structured error
#
# Character selectors are stable IDs that must exist, must be unique, and keep
# request order. Logical selectors must match the axis length with no NA.
# Numeric selectors must be whole numbers, may reorder, must not mix signs,
# drop zero, and must be in bounds; an element may appear once. An empty
# selection is legal on every axis.
.normalize_selection <- function(index, n, ids = NULL, axis = "observation",
                                 abort = NULL) {
  n <- as.integer(n)
  abort <- abort %||% function(message, ...) {
    .frame_abort(message, "fmridataset_error_alignment", ...)
  }
  if (missing(index) || is.null(index)) {
    return(.selection_all(n))
  }
  if (.is_axis_selection(index)) {
    if (!identical(index$n, n)) {
      abort(
        sprintf("%s selection addresses an axis of a different length.", axis),
        axis = axis, reason = "axis_length", expected = n, actual = index$n
      )
    }
    return(index)
  }
  if (is.character(index)) {
    if (is.null(ids)) {
      abort(
        sprintf("%s axis is positional and cannot be selected by ID.", axis),
        axis = axis, reason = "positional_axis"
      )
    }
    if (anyNA(index)) {
      abort(
        sprintf("%s ID selectors must be non-missing.", axis),
        axis = axis, reason = "missing"
      )
    }
    if (anyDuplicated(index)) {
      abort(
        sprintf("%s ID selectors must be unique.", axis),
        axis = axis, reason = "duplicate"
      )
    }
    positions <- match(index, ids)
    if (anyNA(positions)) {
      abort(
        sprintf("Unknown %s ID in selector.", axis),
        axis = axis, reason = "unknown_id",
        unknown = index[is.na(positions)]
      )
    }
    return(.selection_positions(n, positions))
  }
  if (is.logical(index)) {
    if (length(index) != n) {
      abort(
        sprintf("Logical %s selectors must match the axis length.", axis),
        axis = axis, reason = "length", expected = n, actual = length(index)
      )
    }
    if (anyNA(index)) {
      abort(
        sprintf("Logical %s selectors cannot contain NA.", axis),
        axis = axis, reason = "missing"
      )
    }
    return(.selection_positions(n, which(index)))
  }
  if (!is.numeric(index)) {
    abort(
      sprintf("Unsupported %s selector type '%s'.", axis, class(index)[[1L]]),
      axis = axis, reason = "unsupported_type", actual = class(index)
    )
  }
  if (anyNA(index)) {
    abort(
      sprintf("%s positions must be non-missing.", axis),
      axis = axis, reason = "missing"
    )
  }
  if (is.double(index) && any(index != trunc(index))) {
    abort(
      sprintf("%s positions must be whole numbers.", axis),
      axis = axis, reason = "non_integer"
    )
  }
  index <- as.integer(index)
  negative <- index < 0L
  if (any(negative)) {
    if (any(index > 0L)) {
      abort(
        sprintf("%s selectors cannot mix positive and negative positions.", axis),
        axis = axis, reason = "mixed_sign"
      )
    }
    dropped <- -index[negative]
    if (any(dropped > n)) {
      abort(
        sprintf("%s selector is out of bounds.", axis),
        axis = axis, reason = "out_of_bounds", axis_length = n,
        actual = -dropped[dropped > n]
      )
    }
    if (anyDuplicated(dropped)) {
      abort(
        sprintf("%s selector repeats an element.", axis),
        axis = axis, reason = "duplicate"
      )
    }
    return(.selection_positions(n, seq_len(n)[-dropped]))
  }
  index <- index[index != 0L]
  if (any(index > n)) {
    abort(
      sprintf("%s selector is out of bounds.", axis),
      axis = axis, reason = "out_of_bounds", axis_length = n,
      actual = index[index > n]
    )
  }
  if (anyDuplicated(index)) {
    abort(
      sprintf("%s selector repeats an element.", axis),
      axis = axis, reason = "duplicate"
    )
  }
  .selection_positions(n, index)
}

#' @export
print.axis_selection <- function(x, ...) {
  extent <- switch(x$form,
    all = "",
    range = sprintf(" %d:%d", x$start, x$end),
    positions = sprintf(" (%d positions)", length(x$positions))
  )
  cat(sprintf("<axis_selection> %s%s of %d\n", x$form, extent, x$n))
  invisible(x)
}
