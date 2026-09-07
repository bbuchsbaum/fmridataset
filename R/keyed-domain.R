# Shared validation for keyed domains.
#
# Observation axes, feature axes, entities, event tables, auxiliary tables,
# relation edge tables, and persisted manifests all describe the same kind of
# object: rows identified by stable keys, annotated by scalar columns, and
# optionally aligned with two-dimensional blocks. Those rules live here once.
# Every helper takes the calling domain's abort function so that each domain
# keeps its own error class, message wording, and structured fields; nothing
# in this file is a class or an ordinary-user entry point.
#
# Event-specific onset/duration rules and FeatureSpace identity remain explicit
# specializations in their own modules (see ADR-008 for the block rule).

.alignment_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_alignment", ...)
}

.call_abort <- function(abort, message, field = NULL, ...) {
  args <- c(list(message), if (!is.null(field)) list(field = field), list(...))
  do.call(abort, args)
}

# One non-empty string --------------------------------------------------------

.is_one_string <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
}

.assert_one_string <- function(x, field, abort, message = NULL, ...) {
  if (!.is_one_string(x)) {
    .call_abort(
      abort,
      message %||% sprintf("%s must be one non-empty string.", field),
      field = field, ...
    )
  }
  x
}

.assert_optional_string <- function(x, field, abort, message = NULL, ...) {
  if (is.null(x)) {
    return(NULL)
  }
  .assert_one_string(
    x, field, abort,
    message = message %||% sprintf("%s must be NULL or one non-empty string.", field),
    ...
  )
}

# Scalar columns --------------------------------------------------------------

# A scalar column holds exactly one atomic value per row. List columns, matrix
# columns, and columns whose length disagrees with the row count all smuggle
# aligned structure into a table that is only allowed to carry annotations;
# multivariate values belong in an axis_block.
.non_scalar_columns <- function(data) {
  n <- nrow(data)
  vapply(data, function(value) {
    is.list(value) || !is.null(dim(value)) || length(value) != n
  }, logical(1))
}

.assert_scalar_columns <- function(data, abort, message, field = "data", ...) {
  bad <- .non_scalar_columns(data)
  if (any(bad)) {
    .call_abort(abort, message, field = field, columns = names(data)[bad], ...)
  }
  invisible(data)
}

# Stable keys -----------------------------------------------------------------

# Stable keys are character, non-missing, non-empty, and unique. `message`
# collapses the three diagnostics into one domain-specific sentence; without
# it the reason is reported separately, labelled by `what`.
.assert_stable_keys <- function(ids, abort, what = "key", field = NULL,
                                message = NULL, ...) {
  fail <- function(default) {
    .call_abort(abort, message %||% default, field = field, ...)
  }
  if (!is.character(ids)) {
    fail(sprintf("%s IDs must be character values.", what))
  }
  if (anyNA(ids) || any(!nzchar(ids))) {
    fail(sprintf("%s IDs must be non-missing and non-empty.", what))
  }
  if (anyDuplicated(ids)) {
    fail(sprintf("%s IDs must be unique.", what))
  }
  ids
}

# Named registries ------------------------------------------------------------

.has_unique_names <- function(x) {
  if (!length(x)) {
    return(TRUE)
  }
  names_value <- names(x)
  !is.null(names_value) && !anyNA(names_value) && all(nzchar(names_value)) &&
    !anyDuplicated(names_value)
}

# An empty registry is valid; a non-empty one needs unique, non-empty names.
.assert_unique_names <- function(x, abort, message, field = "names", ...) {
  if (!.has_unique_names(x)) {
    .call_abort(abort, message, field = field, ...)
  }
  invisible(x)
}

# Runtime state ---------------------------------------------------------------

.assert_no_runtime_state <- function(x, abort, message,
                                     field = "runtime_state", ...) {
  if (.source_contains_runtime_state(x)) {
    .call_abort(abort, message, field = field, ...)
  }
  invisible(x)
}

# Blocks ----------------------------------------------------------------------

# Block data are two-dimensional: rows are the owning axis elements and
# columns are named components (ADR-008). Array sources are two-dimensional by
# the source contract; in-memory data must carry exactly two dimensions.
.block_shape <- function(data) {
  if (inherits(data, "array_source")) {
    return(as.integer(source_shape(data)))
  }
  d <- dim(data)
  if (is.null(d)) NULL else as.integer(d)
}

.assert_block_shape <- function(data, abort = .alignment_abort, block = NULL,
                                ...) {
  if (inherits(data, "array_source")) {
    return(invisible(data))
  }
  shape <- .block_shape(data)
  if (is.null(shape) || length(shape) != 2L) {
    label <- if (is.null(block)) {
      "Axis block data"
    } else {
      sprintf("Axis block '%s'", block)
    }
    shape_label <- if (is.null(shape)) {
      sprintf("a length-%d vector", length(data))
    } else {
      paste(shape, collapse = " x ")
    }
    .call_abort(
      abort,
      sprintf(
        "%s must be two-dimensional (axis elements by named components); got %s. Higher-order structure belongs in named components, separate blocks, or an assay.",
        label, shape_label
      ),
      field = "blocks",
      block = block,
      shape = shape,
      dims = length(shape),
      ...
    )
  }
  invisible(data)
}

# Validate a named registry of axis_block objects against an axis of length
# `n`. `what` labels the owner in messages ("Axis", "Entity 'subject'");
# `...` adds structured fields such as `entity`.
.assert_aligned_blocks <- function(blocks, n, abort = .alignment_abort,
                                   what = "Axis", ...) {
  if (!is.list(blocks)) {
    .call_abort(
      abort, sprintf("%s blocks must be a named list.", what),
      field = "blocks", ...
    )
  }
  if (length(blocks) && !.has_unique_names(blocks)) {
    .call_abort(
      abort,
      sprintf("%s blocks must be named with unique, non-empty values.", what),
      field = "blocks", ...
    )
  }
  for (name in names(blocks)) {
    block <- blocks[[name]]
    if (!inherits(block, "axis_block")) {
      .call_abort(
        abort, sprintf("%s block '%s' is not an axis_block.", what, name),
        field = "blocks", block = name, ...
      )
    }
    .assert_block_shape(axis_block_data(block), abort, block = name, ...)
    leading <- .block_shape(axis_block_data(block))[[1L]]
    if (leading != n) {
      .call_abort(
        abort,
        sprintf(
          "%s block '%s' is not aligned with its %d elements (block has %d rows).",
          what, name, n, leading
        ),
        field = "blocks", block = name, expected = n, actual = leading, ...
      )
    }
  }
  invisible(blocks)
}

.subset_axis_block <- function(x, index) {
  data <- axis_block_data(x)
  data <- if (inherits(data, "array_source")) {
    source_view(data, observations = index)
  } else {
    data[index, , drop = FALSE]
  }
  axis_block(
    data,
    components = x$components,
    role = x$role,
    units = x$units,
    metadata = x$metadata
  )
}

# Subset scalar rows and every aligned block by the same integer index so the
# two can never drift apart.
.subset_keyed_rows <- function(data, blocks, index) {
  index <- as.integer(index)
  list(
    data = data[index, , drop = FALSE],
    blocks = lapply(blocks, .subset_axis_block, index = index)
  )
}

# Row-bind block data whose component axes have already been aligned. Every
# value must be a two-dimensional matrix-like object or array source with the
# same component count; rbind() would otherwise flatten or recycle silently.
.bind_block_values <- function(values, block = NULL) {
  for (value in values) .assert_block_shape(value, block = block)
  widths <- vapply(values, function(value) .block_shape(value)[[2L]], integer(1))
  if (any(widths != widths[[1L]])) {
    .alignment_abort(
      sprintf(
        "Block %s has different component counts across bound frames (%s).",
        encodeString(block %||% "", quote = "\""),
        paste(widths, collapse = ", ")
      ),
      block = block, actual = widths
    )
  }
  if (any(vapply(values, inherits, logical(1), what = "array_source"))) {
    return(row_bound_source(lapply(values, as_array_source)))
  }
  do.call(rbind, values)
}
