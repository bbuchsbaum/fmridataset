.source_counter_registry <- new.env(parent = emptyenv())

#' Serializable numerical array sources
#'
#' @param x An array source or object coercible to one.
#' @param observations Optional observation positions.
#' @param features Optional feature positions.
#' @param ... Additional method arguments.
#' @name array-source
NULL

#' @rdname array-source
#' @export
as_array_source <- function(x, ...) UseMethod("as_array_source")

#' @export
as_array_source.array_source <- function(x, ...) x

#' @export
as_array_source.matrix <- function(x, ...) memory_source(x, ...)

#' @export
as_array_source.array <- function(x, ...) memory_source(x, ...)

#' @rdname array-source
#' @export
source_shape <- function(x, ...) UseMethod("source_shape")

#' @rdname array-source
#' @export
source_dtype <- function(x, ...) UseMethod("source_dtype")

#' @rdname array-source
#' @export
source_chunks <- function(x, ...) UseMethod("source_chunks")

#' @rdname array-source
#' @export
source_capabilities <- function(x, ...) UseMethod("source_capabilities")

#' @rdname array-source
#' @export
source_fingerprint <- function(x, ...) UseMethod("source_fingerprint")

#' @rdname array-source
#' @export
source_open <- function(x, ...) UseMethod("source_open")

#' @rdname array-source
#' @export
source_read <- function(x, observations = NULL, features = NULL, ...) {
  UseMethod("source_read")
}

#' @rdname array-source
#' @export
source_read_native <- function(x, observations = NULL, ...) {
  UseMethod("source_read_native")
}

#' @rdname array-source
#' @export
source_close <- function(x, ...) UseMethod("source_close")

.source_dtype_from_data <- function(x) {
  if (is.raw(x)) "uint8" else if (is.logical(x)) "logical" else if (is.numeric(x)) "float64" else typeof(x)
}

.supported_source_dtypes <- c(
  "logical", "uint8", "int8", "uint16", "int16", "float16", "bfloat16",
  "uint32", "int32", "float32", "uint64", "int64", "float64",
  "complex64", "complex128"
)

.dtype_bytes <- function(dtype) {
  sizes <- c(
    logical = 1, uint8 = 1, int8 = 1,
    uint16 = 2, int16 = 2, float16 = 2, bfloat16 = 2,
    uint32 = 4, int32 = 4, float32 = 4,
    uint64 = 8, int64 = 8, float64 = 8, complex64 = 8,
    complex128 = 16
  )
  if (!is.character(dtype) || length(dtype) != 1L || is.na(dtype) ||
    !dtype %in% names(sizes)) {
    .frame_abort(
      sprintf("Unsupported source dtype '%s'.", dtype),
      "fmridataset_error_source_contract",
      field = "dtype",
      actual = dtype,
      supported = names(sizes)
    )
  }
  unname(sizes[[dtype]])
}

# Width of one value once R holds it, as opposed to its width in storage.
#
# Budgets exist to bound what a read will occupy in memory, and every read
# realizes an R vector: a float32 assay becomes R doubles, a uint8 assay becomes
# R doubles. Budgeting against the storage width therefore under-counts by the
# dtype ratio -- 2x for float32, 8x for uint8 -- and lets a collection exceed
# the ceiling the caller asked for. This is the shared cost basis; storage width
# (.dtype_bytes) remains the right measure for I/O accounting.
.realized_dtype_bytes <- function(dtype) {
  .dtype_bytes(dtype) # validates the dtype and its error message
  switch(dtype,
    logical = 4,
    complex64 = ,
    complex128 = 16,
    8
  )
}

# The R storage mode a dtype is realized into. Shares its mapping with
# .realized_dtype_bytes() so that the accumulator a read allocates and the
# budget charged for it always describe the same value.
.realized_dtype_mode <- function(dtype) {
  .dtype_bytes(dtype)
  switch(dtype,
    logical = "logical",
    complex64 = ,
    complex128 = "complex",
    "double"
  )
}

.source_cost_traits <- function(x) {
  if (inherits(x, "memory_source")) {
    return(list(already_realized = TRUE, compressed = FALSE))
  }
  if (inherits(x, "nifti_array_source")) {
    return(list(
      already_realized = FALSE,
      compressed = any(grepl("\\.gz$", x$uri, ignore.case = TRUE))
    ))
  }
  if (inherits(x, "zarr_array_source")) {
    # Zarr chunks may be compressed even though compression metadata is kept by
    # the physical store rather than the logical ArraySource descriptor.
    return(list(already_realized = FALSE, compressed = TRUE))
  }
  if (inherits(x, c("source_view", "counting_source", "fault_source")) &&
    inherits(x$source, "array_source")) {
    return(.source_cost_traits(x$source))
  }
  list(already_realized = FALSE, compressed = FALSE)
}

.realization_cost_from_shape <- function(shape, dtype,
                                         already_realized = FALSE,
                                         compressed = FALSE) {
  shape <- as.integer(shape)
  values <- prod(as.double(shape))
  storage_width <- .dtype_bytes(dtype)
  realized_width <- .realized_dtype_bytes(dtype)
  storage_bytes <- values * storage_width
  output_bytes <- values * realized_width
  conversion_buffer_bytes <- if (isTRUE(already_realized) ||
    storage_width == realized_width) {
    0
  } else {
    output_bytes
  }
  # Non-memory readers commonly hold selected values while copying into the
  # final R matrix. Keep that intermediate distinct from dtype conversion so
  # equal-width and converted sources are both represented conservatively.
  selection_buffer_bytes <- if (isTRUE(already_realized)) 0 else output_bytes
  decompression_buffer_bytes <- if (isTRUE(compressed)) storage_bytes else 0
  temporary_bytes <- selection_buffer_bytes + conversion_buffer_bytes +
    decompression_buffer_bytes

  structure(
    list(
      shape = shape,
      values = values,
      storage_dtype = dtype,
      storage_bytes = storage_bytes,
      realized_dtype = .realized_dtype_mode(dtype),
      realized_dtype_bytes = realized_width,
      estimated_output_bytes = output_bytes,
      selection_buffer_bytes = selection_buffer_bytes,
      conversion_buffer_bytes = conversion_buffer_bytes,
      decompression_buffer_bytes = decompression_buffer_bytes,
      estimated_temporary_bytes = temporary_bytes,
      estimated_peak_bytes = output_bytes + temporary_bytes
    ),
    class = "source_realization_cost"
  )
}

#' Estimate the memory cost of realizing an array-source selection
#'
#' The estimate distinguishes source storage from the R object returned by a
#' read. Numeric source dtypes, including float16 and float32, are realized as
#' R doubles. The conservative peak estimate adds selection, conversion, and
#' compressed-input buffers to the retained output. It covers numerical
#' payloads; fixed R object headers and selector metadata are outside the
#' estimate.
#'
#' @param x An array source or object coercible to one.
#' @return A `source_realization_cost` list containing storage and realized
#'   dtypes, storage and output bytes, temporary buffer components, and the
#'   estimated peak bytes.
#' @rdname array-source
#' @export
source_realization_cost <- function(x, observations = NULL, features = NULL) {
  x <- as_array_source(x)
  shape <- source_shape(x)
  observations <- .normalize_source_index(observations, shape[[1L]])
  features <- .normalize_source_index(features, shape[[2L]])
  traits <- .source_cost_traits(x)
  .realization_cost_from_shape(
    c(length(observations), length(features)),
    source_dtype(x),
    already_realized = traits$already_realized,
    compressed = traits$compressed
  )
}

.assert_realization_budget <- function(cost, memory_budget, operation,
                                       force = FALSE) {
  if (isTRUE(force)) {
    return(invisible(cost))
  }
  if (!is.numeric(memory_budget) || length(memory_budget) != 1L ||
    is.na(memory_budget) || memory_budget <= 0) {
    .frame_abort(
      "memory_budget must be one positive number.",
      "fmridataset_error_budget",
      operation = operation,
      memory_budget = memory_budget
    )
  }
  if (cost$estimated_peak_bytes > memory_budget) {
    .frame_abort(
      sprintf(
        paste0(
          "%s is estimated to retain %s output bytes and peak at %s bytes ",
          "(%s source values realized as R %s), above memory_budget of %s bytes."
        ),
        operation,
        format(cost$estimated_output_bytes, scientific = FALSE),
        format(cost$estimated_peak_bytes, scientific = FALSE),
        cost$storage_dtype,
        cost$realized_dtype,
        format(memory_budget, scientific = FALSE)
      ),
      "fmridataset_error_budget",
      operation = operation,
      storage_dtype = cost$storage_dtype,
      realized_dtype = cost$realized_dtype,
      estimated_output_bytes = cost$estimated_output_bytes,
      estimated_temporary_bytes = cost$estimated_temporary_bytes,
      estimated_peak_bytes = cost$estimated_peak_bytes,
      required_bytes = cost$estimated_peak_bytes,
      memory_budget = memory_budget
    )
  }
  invisible(cost)
}

# Accumulator for a composed read, typed by the declared dtype.
#
# Composing sources used to preallocate matrix(NA_real_, ...) regardless of
# dtype, so reading a logical assay through row_bound_source() or
# row_sharded_source() returned doubles while source_dtype() still reported
# "logical" - the descriptor and the data disagreed.
.realized_na_matrix <- function(dtype, nrow, ncol) {
  fill <- switch(.realized_dtype_mode(dtype),
    logical = NA,
    complex = NA_complex_,
    NA_real_
  )
  matrix(fill, nrow = nrow, ncol = ncol)
}

# Zero-extent result of the same type.
.realized_empty_matrix <- function(dtype, nrow, ncol) {
  matrix(
    vector(mode = .realized_dtype_mode(dtype), length = 0L),
    nrow = nrow, ncol = ncol
  )
}

.source_contains_runtime_state <- function(x) {
  if (is.environment(x) || is.function(x) || typeof(x) == "externalptr") {
    return(TRUE)
  }
  if (is.pairlist(x)) {
    return(any(vapply(as.list(x), .source_contains_runtime_state, logical(1))))
  }
  if (is.list(x)) {
    return(any(vapply(unclass(x), .source_contains_runtime_state, logical(1))))
  }
  FALSE
}

#' Inspect and validate an array-source contract
#'
#' A valid canonical source is two-dimensional, has an explicit supported
#' dtype and chunk grid, advertises serializable block slicing, provides a
#' stable non-empty fingerprint, and contains no runtime handles or closures.
#'
#' @param x An `array_source` descriptor.
#' @return `source_descriptor()` returns a plain serializable contract list.
#'   `validate_array_source()` invisibly returns `x` or raises a structured
#'   source-contract error.
#' @export
source_descriptor <- function(x) {
  if (!inherits(x, "array_source")) {
    .frame_abort(
      "Object does not inherit from array_source.",
      "fmridataset_error_source_contract",
      field = "class"
    )
  }
  list(
    shape = source_shape(x),
    dtype = source_dtype(x),
    chunks = source_chunks(x),
    capabilities = source_capabilities(x),
    fingerprint = source_fingerprint(x)
  )
}

#' @rdname source_descriptor
#' @export
validate_array_source <- function(x) {
  descriptor <- source_descriptor(x)
  shape <- descriptor$shape
  if (!is.numeric(shape) || length(shape) != 2L || anyNA(shape) ||
    any(shape < 0) || any(shape != as.integer(shape))) {
    .frame_abort(
      "Source shape must contain two non-negative integers.",
      "fmridataset_error_source_contract",
      field = "shape",
      actual = shape
    )
  }
  dtype <- descriptor$dtype
  if (!is.character(dtype) || length(dtype) != 1L || is.na(dtype) ||
    !dtype %in% .supported_source_dtypes) {
    .frame_abort(
      "Source dtype is missing or unsupported.",
      "fmridataset_error_source_contract",
      field = "dtype",
      actual = dtype,
      supported = .supported_source_dtypes
    )
  }
  chunks <- descriptor$chunks
  if (!is.numeric(chunks) || length(chunks) != 2L || anyNA(chunks) ||
    any(chunks <= 0) || any(chunks != as.integer(chunks)) ||
    any(chunks > pmax(1L, as.integer(shape)))) {
    .frame_abort(
      "Source chunks must be two positive integers bounded by the source shape.",
      "fmridataset_error_source_contract",
      field = "chunks",
      actual = chunks,
      shape = shape
    )
  }
  capabilities <- descriptor$capabilities
  required <- c("block_slice", "serializable")
  if (!is.character(capabilities) || anyNA(capabilities) ||
    any(!nzchar(capabilities)) || anyDuplicated(capabilities) ||
    !all(required %in% capabilities)) {
    .frame_abort(
      "Source capabilities must be unique strings including block_slice and serializable.",
      "fmridataset_error_source_contract",
      field = "capabilities",
      actual = capabilities,
      required = required
    )
  }
  fingerprint <- descriptor$fingerprint
  if (!is.character(fingerprint) || length(fingerprint) != 1L ||
    is.na(fingerprint) || !nzchar(fingerprint)) {
    .frame_abort(
      "Source fingerprint must be one non-empty string.",
      "fmridataset_error_source_contract",
      field = "fingerprint",
      actual = fingerprint
    )
  }
  if (.source_contains_runtime_state(x)) {
    .frame_abort(
      "Canonical source descriptors cannot contain functions, environments, or external pointers.",
      "fmridataset_error_source_contract",
      field = "runtime_state"
    )
  }
  invisible(x)
}

.normalize_source_index <- function(index, n) {
  if (is.null(index)) {
    return(seq_len(n))
  }
  if (is.logical(index)) {
    if (length(index) != n || anyNA(index)) {
      .frame_abort("Logical source selectors must match the axis length and contain no NA.", "fmridataset_error_alignment")
    }
    return(which(index))
  }
  index <- as.integer(index)
  if (anyNA(index) || any(index < 1L | index > n)) {
    .frame_abort("Source selector is out of bounds.", "fmridataset_error_alignment")
  }
  index
}

#' Construct an in-memory array source
#'
#' @param data A two-dimensional matrix or array.
#' @param dtype Logical storage dtype. Numeric R matrices default to
#'   `"float64"`.
#' @param chunks Optional logical chunk shape.
#' @return A serializable `memory_source`.
#' @export
memory_source <- function(data, dtype = NULL, chunks = NULL) {
  d <- dim(data)
  if (is.null(d) || length(d) != 2L) {
    .frame_abort("A frame ArraySource must be two dimensional.", "fmridataset_error_alignment")
  }
  d <- as.integer(d)
  chunks <- as.integer(chunks %||% pmax(1L, d))
  if (length(chunks) != 2L || any(chunks <= 0L)) {
    .frame_abort("Source chunks must contain two positive integers.", "fmridataset_error_alignment")
  }
  dtype <- dtype %||% .source_dtype_from_data(data)
  .dtype_bytes(dtype)
  out <- structure(
    list(
      data = data,
      shape = d,
      dtype = dtype,
      chunks = pmin(chunks, pmax(1L, d)),
      capabilities = c("row_slice", "column_slice", "block_slice", "serializable"),
      schema_version = 1L
    ),
    class = c("memory_source", "array_source")
  )
  out$fingerprint <- .canonical_digest(list(
    type = "memory",
    schema_version = out$schema_version,
    shape = out$shape,
    dtype = out$dtype,
    chunks = out$chunks,
    data = out$data
  ))
  validate_array_source(out)
  out
}

#' @export
source_shape.memory_source <- function(x, ...) x$shape
#' @export
source_dtype.memory_source <- function(x, ...) x$dtype
#' @export
source_chunks.memory_source <- function(x, ...) x$chunks
#' @export
source_capabilities.memory_source <- function(x, ...) {
  x$capabilities
}
#' @export
source_fingerprint.memory_source <- function(x, ...) {
  x$fingerprint
}
#' @export
source_open.memory_source <- function(x, ...) {
  structure(list(source = x), class = c("memory_source_handle", "array_source_handle"))
}
#' @export
source_read.memory_source <- function(x, observations = NULL, features = NULL, ...) {
  observations <- .normalize_source_index(observations, x$shape[1L])
  features <- .normalize_source_index(features, x$shape[2L])
  x$data[observations, features, drop = FALSE]
}
#' @export
source_read_native.memory_source <- function(x, observations = NULL, ...) {
  .frame_abort(
    "memory_source has no native spatial read path.",
    "fmridataset_error_backend_io",
    operation = "native_read"
  )
}
#' @export
source_close.memory_source <- function(x, ...) invisible(TRUE)

#' @export
source_shape.array_source_handle <- function(x, ...) source_shape(x$source)
#' @export
source_dtype.array_source_handle <- function(x, ...) source_dtype(x$source)
#' @export
source_chunks.array_source_handle <- function(x, ...) source_chunks(x$source)
#' @export
source_capabilities.array_source_handle <- function(x, ...) source_capabilities(x$source)
#' @export
source_fingerprint.array_source_handle <- function(x, ...) source_fingerprint(x$source)
#' @export
source_read.array_source_handle <- function(x, observations = NULL, features = NULL, ...) {
  source_read(x$source, observations = observations, features = features, ...)
}
#' @export
source_read_native.array_source_handle <- function(x, observations = NULL, ...) {
  source_read_native(x$source, observations = observations, ...)
}
#' @export
source_close.array_source_handle <- function(x, ...) invisible(TRUE)

#' Construct a lazy view over an array source
#'
#' @param source An `array_source`.
#' @param observations Stored observation selector.
#' @param features Stored feature selector.
#' @return A serializable source view.
#' @export
source_view <- function(source, observations = NULL, features = NULL) {
  source <- as_array_source(source)
  shape <- source_shape(source)
  observations <- .normalize_source_index(observations, shape[1L])
  features <- .normalize_source_index(features, shape[2L])
  out <- structure(
    list(source = source, observations = observations, features = features),
    class = c("source_view", "array_source")
  )
  validate_array_source(out)
  out
}

#' @export
source_shape.source_view <- function(x, ...) c(length(x$observations), length(x$features))
#' @export
source_dtype.source_view <- function(x, ...) source_dtype(x$source)
#' @export
source_chunks.source_view <- function(x, ...) pmin(source_chunks(x$source), pmax(1L, source_shape(x)))
#' @export
source_capabilities.source_view <- function(x, ...) {
  capabilities <- source_capabilities(x$source)
  if (!identical(x$features, seq_len(source_shape(x$source)[2L]))) {
    capabilities <- setdiff(capabilities, "native_read")
  }
  capabilities
}
#' @export
source_fingerprint.source_view <- function(x, ...) {
  .canonical_digest(list(
    source = source_fingerprint(x$source),
    observations = x$observations,
    features = x$features
  ))
}
#' @export
source_open.source_view <- function(x, ...) {
  structure(list(source = x), class = c("source_view_handle", "array_source_handle"))
}
#' @export
source_read.source_view <- function(x, observations = NULL, features = NULL, ...) {
  observations <- .normalize_source_index(observations, length(x$observations))
  features <- .normalize_source_index(features, length(x$features))
  source_read(
    x$source,
    observations = x$observations[observations],
    features = x$features[features],
    ...
  )
}
#' @export
source_read_native.source_view <- function(x, observations = NULL, ...) {
  if (!"native_read" %in% source_capabilities(x)) {
    .frame_abort(
      "A feature-restricted source view has no valid native spatial read path.",
      "fmridataset_error_backend_io",
      operation = "native_read"
    )
  }
  observations <- .normalize_source_index(observations, length(x$observations))
  source_read_native(x$source, observations = x$observations[observations], ...)
}
#' @export
source_close.source_view <- function(x, ...) invisible(TRUE)

#' Instrument an array source
#'
#' `counting_source()` records numerical reads without placing a mutable
#' environment inside the source descriptor.
#'
#' @param source An array source.
#' @return A serializable instrumented source.
#' @export
counting_source <- function(source) {
  id <- uuid::UUIDgenerate()
  .source_counter_registry[[id]] <- list(
    reads = 0, values = 0, bytes = 0, storage_bytes = 0,
    output_bytes = 0, opens = 0, closes = 0
  )
  out <- structure(
    list(source = as_array_source(source), counter_id = id),
    class = c("counting_source", "array_source")
  )
  validate_array_source(out)
  out
}

.source_counter <- function(x) {
  value <- .source_counter_registry[[x$counter_id]]
  if (is.null(value)) {
    value <- list(reads = 0, values = 0, bytes = 0, opens = 0, closes = 0)
    .source_counter_registry[[x$counter_id]] <- value
  }
  value
}

.set_source_counter <- function(x, value) {
  .source_counter_registry[[x$counter_id]] <- value
  invisible(x)
}

#' @param x A counting source.
#' @rdname counting_source
#' @export
source_counts <- function(x) .source_counter(x)

#' @rdname counting_source
#' @export
reset_source_counts <- function(x) {
  .set_source_counter(x, list(
    reads = 0, values = 0, bytes = 0, storage_bytes = 0,
    output_bytes = 0, opens = 0, closes = 0
  ))
}

#' @export
source_shape.counting_source <- function(x, ...) source_shape(x$source)
#' @export
source_dtype.counting_source <- function(x, ...) source_dtype(x$source)
#' @export
source_chunks.counting_source <- function(x, ...) source_chunks(x$source)
#' @export
source_capabilities.counting_source <- function(x, ...) source_capabilities(x$source)
#' @export
source_fingerprint.counting_source <- function(x, ...) source_fingerprint(x$source)
#' @export
source_open.counting_source <- function(x, ...) {
  count <- .source_counter(x)
  count$opens <- count$opens + 1
  .set_source_counter(x, count)
  structure(list(source = x), class = c("counting_source_handle", "array_source_handle"))
}
#' @export
source_read.counting_source <- function(x, observations = NULL, features = NULL, ...) {
  shape <- source_shape(x)
  observations <- .normalize_source_index(observations, shape[1L])
  features <- .normalize_source_index(features, shape[2L])
  count <- .source_counter(x)
  n <- length(observations) * length(features)
  storage_bytes <- n * .dtype_bytes(source_dtype(x))
  output_bytes <- n * .realized_dtype_bytes(source_dtype(x))
  count$reads <- count$reads + 1
  count$values <- count$values + n
  count$bytes <- count$bytes + output_bytes
  count$storage_bytes <- count$storage_bytes + storage_bytes
  count$output_bytes <- count$output_bytes + output_bytes
  .set_source_counter(x, count)
  source_read(x$source, observations = observations, features = features, ...)
}
#' @export
source_read_native.counting_source <- function(x, observations = NULL, ...) {
  value <- source_read_native(x$source, observations = observations, ...)
  count <- .source_counter(x)
  storage_bytes <- length(value) * .dtype_bytes(source_dtype(x))
  output_bytes <- length(value) * .realized_dtype_bytes(source_dtype(x))
  count$reads <- count$reads + 1
  count$values <- count$values + length(value)
  count$bytes <- count$bytes + output_bytes
  count$storage_bytes <- count$storage_bytes + storage_bytes
  count$output_bytes <- count$output_bytes + output_bytes
  .set_source_counter(x, count)
  value
}
#' @export
source_close.counting_source <- function(x, ...) {
  count <- .source_counter(x)
  count$closes <- count$closes + 1
  .set_source_counter(x, count)
  invisible(TRUE)
}

#' Inject deterministic source failures
#'
#' @param source An array source.
#' @param stage One of `"open"`, `"read"`, `"native_read"`, or `"close"`.
#' @param message Failure message.
#' @return A serializable fault-injecting source.
#' @export
fault_source <- function(source, stage = c("read", "open", "native_read", "close"),
                         message = NULL) {
  stage <- match.arg(stage)
  out <- structure(
    list(source = as_array_source(source), stage = stage, message = message %||% paste("Injected", stage, "failure")),
    class = c("fault_source", "array_source")
  )
  validate_array_source(out)
  out
}

.fault_maybe <- function(x, stage) {
  if (identical(x$stage, stage)) {
    .frame_abort(
      x$message,
      "fmridataset_error_backend_io",
      operation = stage,
      injected = TRUE
    )
  }
}

#' @export
source_shape.fault_source <- function(x, ...) source_shape(x$source)
#' @export
source_dtype.fault_source <- function(x, ...) source_dtype(x$source)
#' @export
source_chunks.fault_source <- function(x, ...) source_chunks(x$source)
#' @export
source_capabilities.fault_source <- function(x, ...) source_capabilities(x$source)
#' @export
source_fingerprint.fault_source <- function(x, ...) {
  .canonical_digest(list(
    type = "fault_source",
    source = source_fingerprint(x$source),
    stage = x$stage,
    message = x$message
  ))
}
#' @export
source_open.fault_source <- function(x, ...) {
  .fault_maybe(x, "open")
  source_open(x$source, ...)
}
#' @export
source_read.fault_source <- function(x, observations = NULL, features = NULL, ...) {
  .fault_maybe(x, "read")
  source_read(x$source, observations = observations, features = features, ...)
}
#' @export
source_read_native.fault_source <- function(x, observations = NULL, ...) {
  .fault_maybe(x, "native_read")
  source_read_native(x$source, observations = observations, ...)
}
#' @export
source_close.fault_source <- function(x, ...) {
  .fault_maybe(x, "close")
  source_close(x$source, ...)
}

#' Construct a manifest-backed row-sharded source
#'
#' `row_sharded_source()` presents compatible child sources as one logical
#' observation-by-feature array. Stable shard IDs and explicit boundaries make
#' global-to-local row routing inspectable and serializable. Reads are grouped
#' by touched shard, so an arbitrary ordered selector is issued at most once to
#' each selected child.
#'
#' @param sources A non-empty list of compatible two-dimensional array sources.
#' @param shard_ids Stable, unique shard identifiers.
#' @param shard_data Optional scalar metadata with one row per shard. Names used
#'   by the shard manifest are reserved.
#' @return A serializable `row_sharded_source`.
#' @export
row_sharded_source <- function(sources, shard_ids = NULL, shard_data = NULL) {
  if (!is.list(sources) || !length(sources)) {
    .frame_abort("At least one source is required.", "fmridataset_error_alignment")
  }
  sources <- lapply(sources, as_array_source)
  invisible(lapply(sources, validate_array_source))
  shapes <- lapply(sources, source_shape)
  # Empty shards are permitted. They make two consecutive boundaries equal, and
  # findInterval() resolves a row to the last boundary at or below it, so an
  # empty shard is simply never routed to. Rejecting them made
  # bind_observations() refuse a zero-observation frame, which is legal
  # everywhere else in the model - such frames subset, collect, plan, and FDS
  # round-trip - and which binding should treat as an identity.
  rows <- vapply(shapes, `[[`, integer(1), 1L)
  n_feature <- vapply(shapes, `[[`, integer(1), 2L)
  if (length(unique(n_feature)) != 1L) {
    .frame_abort("Row-sharded sources must have the same feature count.", "fmridataset_error_alignment")
  }
  dtypes <- vapply(sources, source_dtype, character(1))
  if (length(unique(dtypes)) != 1L) {
    .frame_abort("Row-sharded sources must have the same dtype.", "fmridataset_error_alignment")
  }
  if (is.null(shard_ids)) {
    shard_ids <- sprintf("shard-%06d", seq_along(sources))
  }
  if (!is.character(shard_ids) || length(shard_ids) != length(sources) ||
    anyNA(shard_ids) || any(!nzchar(shard_ids)) || anyDuplicated(shard_ids)) {
    .frame_abort(
      "shard_ids must contain one unique, non-empty string per source.",
      "fmridataset_error_alignment"
    )
  }
  shard_data <- .normalize_shard_data(shard_data, length(sources))
  total_rows <- sum(as.double(rows))
  if (total_rows > .Machine$integer.max) {
    .frame_abort(
      "The logical observation axis exceeds R's integer indexing limit.",
      "fmridataset_error_alignment"
    )
  }
  boundaries <- as.integer(c(0, cumsum(rows)))
  out <- structure(
    list(
      sources = sources,
      shard_ids = unname(shard_ids),
      shard_data = shard_data,
      rows = rows,
      boundaries = boundaries,
      shape = c(as.integer(total_rows), n_feature[[1L]]),
      dtype = dtypes[[1L]],
      schema_version = 1L
    ),
    class = c("row_sharded_source", "row_bound_source", "array_source")
  )
  validate_array_source(out)
  out
}

.shard_manifest_reserved <- c(
  ".shard_id", ".start", ".end", ".n_observation", ".source_fingerprint"
)

.normalize_shard_data <- function(shard_data, n_shard) {
  if (is.null(shard_data)) {
    return(data.frame(row.names = seq_len(n_shard)))
  }
  if (!is.data.frame(shard_data) || nrow(shard_data) != n_shard) {
    .frame_abort(
      "shard_data must be a data frame with one row per shard.",
      "fmridataset_error_alignment"
    )
  }
  if (anyDuplicated(names(shard_data)) || any(names(shard_data) %in% .shard_manifest_reserved)) {
    .frame_abort(
      "shard_data names must be unique and cannot use reserved manifest names.",
      "fmridataset_error_alignment"
    )
  }
  row.names(shard_data) <- seq_len(n_shard)
  shard_data
}

#' Inspect a row-sharded source manifest
#'
#' @param x A `row_sharded_source`.
#' @return A data frame describing stable IDs, logical row ranges, source
#'   fingerprints, and user-supplied shard metadata.
#' @export
shard_manifest <- function(x) {
  if (!inherits(x, "row_sharded_source")) {
    .frame_abort("x must be a row_sharded_source.", "fmridataset_error_alignment")
  }
  core <- data.frame(
    .shard_id = x$shard_ids,
    .start = head(x$boundaries, -1L) + 1L,
    .end = tail(x$boundaries, -1L),
    .n_observation = x$rows,
    .source_fingerprint = vapply(x$sources, source_fingerprint, character(1)),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  cbind(core, x$shard_data, stringsAsFactors = FALSE)
}

#' Resolve logical observation rows to shards
#'
#' @param x A `row_sharded_source`.
#' @param observations Logical observation positions in requested order.
#' @return A data frame mapping each request position to a shard and local row.
#' @export
locate_source_rows <- function(x, observations = NULL) {
  if (!inherits(x, "row_sharded_source")) {
    .frame_abort("x must be a row_sharded_source.", "fmridataset_error_alignment")
  }
  observations <- .normalize_source_index(observations, x$shape[[1L]])
  shard <- if (length(observations)) {
    findInterval(observations - 1L, x$boundaries[-length(x$boundaries)])
  } else {
    integer()
  }
  data.frame(
    .request_position = seq_along(observations),
    .observation = observations,
    .shard_index = as.integer(shard),
    .shard_id = x$shard_ids[shard],
    .local_observation = observations - x$boundaries[shard],
    stringsAsFactors = FALSE
  )
}

#' Append immutable source shards
#'
#' @param x An existing `row_sharded_source`.
#' @param sources New compatible child sources.
#' @param shard_ids Stable IDs for the new shards.
#' @param shard_data Optional metadata for the new shards. Its columns must
#'   match existing shard metadata.
#' @return A new `row_sharded_source`; `x` and its child descriptors are not
#'   modified.
#' @export
append_source_shards <- function(x, sources, shard_ids = NULL, shard_data = NULL) {
  if (!inherits(x, "row_sharded_source")) {
    .frame_abort("x must be a row_sharded_source.", "fmridataset_error_alignment")
  }
  if (!is.list(sources) || !length(sources)) {
    .frame_abort("At least one new source is required.", "fmridataset_error_alignment")
  }
  n_new <- length(sources)
  if (is.null(shard_ids)) {
    shard_ids <- sprintf("shard-%06d", length(x$sources) + seq_len(n_new))
  }
  new_data <- .normalize_shard_data(shard_data, n_new)
  if (!identical(names(x$shard_data), names(new_data))) {
    .frame_abort(
      "Appended shard_data columns must match the existing manifest metadata.",
      "fmridataset_error_alignment"
    )
  }
  combined_data <- if (!ncol(x$shard_data)) {
    data.frame(row.names = seq_len(nrow(x$shard_data) + n_new))
  } else {
    value <- rbind(x$shard_data, new_data)
    row.names(value) <- seq_len(nrow(value))
    value
  }
  row_sharded_source(
    c(x$sources, sources),
    shard_ids = c(x$shard_ids, shard_ids),
    shard_data = combined_data
  )
}

#' @export
source_shape.row_sharded_source <- function(x, ...) x$shape
#' @export
source_dtype.row_sharded_source <- function(x, ...) x$dtype
#' @export
source_chunks.row_sharded_source <- function(x, ...) {
  chunks <- lapply(x$sources, source_chunks)
  c(
    max(1L, min(vapply(chunks, `[[`, integer(1), 1L))),
    max(1L, min(vapply(chunks, `[[`, integer(1), 2L)))
  )
}
#' @export
source_capabilities.row_sharded_source <- function(x, ...) {
  setdiff(Reduce(intersect, lapply(x$sources, source_capabilities)), "native_read")
}
#' @export
source_fingerprint.row_sharded_source <- function(x, ...) {
  .canonical_digest(list(
    type = "row_sharded_source",
    schema_version = x$schema_version,
    shape = x$shape,
    dtype = x$dtype,
    boundaries = x$boundaries,
    shard_ids = x$shard_ids,
    shard_data = x$shard_data,
    sources = lapply(x$sources, source_fingerprint)
  ))
}
#' @export
source_open.row_sharded_source <- function(x, ...) {
  structure(list(source = x), class = c("row_sharded_source_handle", "array_source_handle"))
}
#' @export
source_read.row_sharded_source <- function(x, observations = NULL, features = NULL, ...) {
  observations <- .normalize_source_index(observations, x$shape[[1L]])
  features <- .normalize_source_index(features, x$shape[[2L]])
  if (!length(observations) || !length(features)) {
    return(.realized_empty_matrix(x$dtype, length(observations), length(features)))
  }
  location <- locate_source_rows(x, observations)
  out <- .realized_na_matrix(x$dtype, length(observations), length(features))
  for (shard in unique(location$.shard_index)) {
    at <- which(location$.shard_index == shard)
    out[at, ] <- source_read(
      x$sources[[shard]],
      observations = location$.local_observation[at],
      features = features,
      ...
    )
  }
  out
}
#' @export
source_read_native.row_sharded_source <- function(x, observations = NULL, ...) {
  .frame_abort(
    "Native reads require an explicit per-shard dispatch plan.",
    "fmridataset_error_backend_io",
    operation = "native_read"
  )
}
#' @export
source_close.row_sharded_source <- function(x, ...) invisible(TRUE)

#' Bind compatible sources along observations
#'
#' @param sources A non-empty list of two-dimensional array sources.
#' @return A serializable `row_sharded_source`. This compatibility constructor
#'   assigns deterministic shard IDs.
#' @export
row_bound_source <- function(sources) {
  row_sharded_source(sources)
}

#' @export
source_shape.row_bound_source <- function(x, ...) x$shape
#' @export
source_dtype.row_bound_source <- function(x, ...) x$dtype
#' @export
source_chunks.row_bound_source <- function(x, ...) {
  chunks <- lapply(x$sources, source_chunks)
  c(max(1L, min(vapply(chunks, `[[`, integer(1), 1L))), max(1L, min(vapply(chunks, `[[`, integer(1), 2L))))
}
#' @export
source_capabilities.row_bound_source <- function(x, ...) {
  setdiff(Reduce(intersect, lapply(x$sources, source_capabilities)), "native_read")
}
#' @export
source_fingerprint.row_bound_source <- function(x, ...) {
  .canonical_digest(list(
    type = "row_bound_source",
    shape = x$shape,
    dtype = x$dtype,
    boundaries = x$boundaries,
    sources = lapply(x$sources, source_fingerprint)
  ))
}
#' @export
source_open.row_bound_source <- function(x, ...) {
  structure(list(source = x), class = c("row_bound_source_handle", "array_source_handle"))
}
#' @export
source_read.row_bound_source <- function(x, observations = NULL, features = NULL, ...) {
  observations <- .normalize_source_index(observations, x$shape[1L])
  features <- .normalize_source_index(features, x$shape[2L])
  if (!length(observations) || !length(features)) {
    return(.realized_empty_matrix(x$dtype, length(observations), length(features)))
  }
  subject <- findInterval(observations - 1L, x$boundaries[-length(x$boundaries)])
  out <- .realized_na_matrix(x$dtype, length(observations), length(features))
  for (s in unique(subject)) {
    at <- which(subject == s)
    local <- observations[at] - x$boundaries[s]
    out[at, ] <- source_read(x$sources[[s]], observations = local, features = features, ...)
  }
  out
}
#' @export
source_read_native.row_bound_source <- function(x, observations = NULL, ...) {
  .frame_abort("Native reads must target one child source.", "fmridataset_error_backend_io")
}
#' @export
source_close.row_bound_source <- function(x, ...) invisible(TRUE)

#' @importFrom delarr delarr_provider_pull
#' @export
delarr_provider_pull.array_source <- function(provider, indices, ...) {
  if (length(indices) != 2L) {
    .frame_abort(
      "An fmridataset ArraySource provider requires two selectors.",
      "fmridataset_error_backend_io",
      operation = "provider_read"
    )
  }
  memory_budget <- attr(provider, "fmridataset.memory_budget", exact = TRUE)
  if (!is.null(memory_budget)) {
    cost <- source_realization_cost(
      provider,
      observations = indices[[1L]],
      features = indices[[2L]]
    )
    .assert_realization_budget(cost, memory_budget, "delarr provider pull")
  }
  source_read(
    provider,
    observations = indices[[1L]],
    features = indices[[2L]],
    ...
  )
}

#' @export
as_delarr.array_source <- function(backend, memory_budget = Inf, ...) {
  .ensure_delarr()
  if (!"delarr_provider" %in% getNamespaceExports("delarr")) {
    .frame_abort(
      "The installed delarr lacks reconstructible provider seeds.",
      "fmridataset_error_backend_io",
      operation = "as_delarr"
    )
  }
  shape <- source_shape(backend)
  chunks <- source_chunks(backend)
  cost <- source_realization_cost(backend)
  .assert_realization_budget(cost, memory_budget, "delarr realization")
  attr(backend, "fmridataset.memory_budget") <- memory_budget
  delarr::delarr_provider(
    provider = backend,
    dims = shape,
    chunk_hint = list(
      axis1 = chunks[[1L]],
      axis2 = chunks[[2L]],
      rows = chunks[[1L]],
      cols = chunks[[2L]]
    )
  )
}
