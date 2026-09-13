.content_hash_contract <- list(
  id = "org.fmridataset.content-hash/v1",
  version = 1L,
  algorithm = "sha256",
  order = "row-major",
  leaf_values = 65536L,
  byte_order = "big-endian",
  nan = "canonical-payload",
  negative_zero = "preserved",
  portability = "R-only"
)

#' Content-hash contract
#'
#' Content hashing version 1 is a chained SHA-256 over fixed-size leaves of the
#' realized values in row-major order. The digest depends on the shape, the
#' realized R mode, and every value, and on nothing else: not on the storage
#' dtype, the chunk grid, the read block size, or the composition of sources
#' that produced the values.
#'
#' @return A serializable content-hash contract descriptor.
#' @examples
#' content_hash_contract()
#' @export
content_hash_contract <- function() .content_hash_contract

#' Compute a numerical content hash of an array source
#'
#' `content_hash()` is the one operation in the package that identifies an
#' array by its values. It is the content domain of the identity contract
#' (see [identity_descriptor()]) and it is always explicit: nothing in the
#' package computes it for you. Ordinary construction, inspection,
#' fingerprinting, planning, and persistence never call it.
#'
#' The digest is `O(n)` in the number of values. The default method streams:
#' it opens one handle, reads bounded row blocks through the source protocol
#' aligned to the source chunk grid, and hashes each block into a running
#' state, so no source is materialized whole. `block_bytes` bounds the
#' realized size of one read and affects only I/O; every block size produces
#' the same digest. Blocks cover whole rows when a row fits the bound and
#' consecutive feature ranges of one row otherwise.
#'
#' Two sources hash equal exactly when they have the same shape, realize to the
#' same R mode (double, logical, or complex), and hold equal values. Storage
#' dtype, chunking, wrappers, views, and shard composition do not enter the
#' digest, so an in-memory copy and a row-bound composition of the same values
#' agree. `NaN` payloads and `NA` are normalized to one representation each and
#' remain distinct; negative zero stays distinct from zero.
#'
#' The digest describes the values at the moment they were read. It is a
#' receipt, not a live property: a later in-memory mutation of a memory
#' source's payload is not detected by [source_fingerprint()] and shows only
#' when a fresh `content_hash()` is compared with the earlier one.
#'
#' @param x An array source, a handle, or a matrix coercible to a memory
#'   source.
#' @param ... Passed to methods.
#' @param block_bytes Upper bound on the realized bytes of one read. The
#'   default reuses the block-planning option
#'   `fmridataset.target_block_bytes`.
#' @return One lowercase SHA-256 hexadecimal string.
#' @seealso [source_fingerprint()] for the cheap revision identity of a
#'   descriptor, [identity_descriptor()] for typed identity domains, and
#'   `inst/architecture/ADR-009-source-fingerprints-and-content-hashes.md`
#'   for the policy.
#' @examples
#' a <- memory_source(matrix(1:6, 2, 3))
#' b <- memory_source(matrix(as.double(1:6), 2, 3), chunks = c(1, 1))
#' identical(source_fingerprint(a), source_fingerprint(b))
#' identical(content_hash(a), content_hash(b))
#' @export
content_hash <- function(x, ...) UseMethod("content_hash")

#' @rdname content_hash
#' @export
content_hash.default <- function(x, ...) {
  if (inherits(x, "array_source_handle")) {
    return(.content_hash_through(x, ...))
  }
  content_hash(as_array_source(x), ...)
}

#' @rdname content_hash
#' @export
content_hash.array_source <- function(
  x, ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  handle <- source_open(x)
  on.exit(source_close(handle), add = TRUE)
  .content_hash_through(handle, block_bytes = block_bytes)
}

#' @rdname content_hash
#' @export
content_hash.memory_source <- function(
  x, ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  # The values are already realized. Hashing still proceeds in row blocks so
  # that at most one bounded row-major copy exists at a time; a whole-array
  # transpose would double the resident payload.
  .content_hash_through(x, block_bytes = block_bytes)
}

#' @rdname content_hash
#' @export
content_hash.source_view <- function(
  x, ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  # A view hashes as the array it presents: its own row-major order, through
  # its own reads, so selector reordering is reflected.
  handle <- source_open(x)
  on.exit(source_close(handle), add = TRUE)
  .content_hash_through(handle, block_bytes = block_bytes)
}

#' @rdname content_hash
#' @export
content_hash.row_bound_source <- function(
  x, ...,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  # Shards are consecutive row ranges of the logical array, so streaming each
  # child in turn reproduces the row-major order of the whole. Each child is
  # opened once and no read crosses a shard boundary.
  state <- .content_hash_state(x)
  for (child in x$sources) {
    handle <- source_open(child)
    state <- tryCatch(
      .content_hash_feed(state, handle, block_bytes = block_bytes),
      finally = source_close(handle)
    )
  }
  .content_hash_finish(state)
}

# Streaming core ---------------------------------------------------------

.content_hash_through <- function(
  reader,
  block_bytes = getOption("fmridataset.target_block_bytes", 4 * 1024^2)
) {
  state <- .content_hash_state(reader)
  state <- .content_hash_feed(state, reader, block_bytes = block_bytes)
  .content_hash_finish(state)
}

.content_hash_state <- function(reader) {
  shape <- as.integer(source_shape(reader))
  dtype <- source_dtype(reader)
  mode <- .realized_dtype_mode(dtype)
  header <- c(
    charToRaw(paste0(.content_hash_contract$id, "\n")),
    .canonical_int32(shape),
    charToRaw(switch(mode,
      double = "d",
      logical = "l",
      complex = "z"
    ))
  )
  list(
    shape = shape,
    mode = mode,
    width = switch(mode,
      double = 8L,
      logical = 1L,
      complex = 16L
    ),
    digest = .content_hash_digest(header),
    carry = raw(),
    values = 0
  )
}

.content_hash_digest <- function(bytes) {
  digest::digest(
    bytes,
    algo = .content_hash_contract$algorithm, serialize = FALSE
  )
}

# Bytes of one realized block in row-major order. Realization follows the
# source dtype exactly as reads do (.realized_dtype_mode), so an integer R
# matrix in a float64 memory source hashes as the doubles a read returns.
.content_hash_block_bytes <- function(block, mode) {
  values <- as.vector(t(block))
  if (identical(mode, "logical")) {
    values <- as.integer(as.logical(values))
    values[is.na(values)] <- 2L
    return(as.raw(values))
  }
  if (identical(mode, "complex")) {
    values <- as.complex(values)
    values <- c(rbind(Re(values), Im(values)))
  }
  values <- as.double(values)
  # One representation for NaN and one for NA. Arithmetic NaNs carry
  # platform-dependent sign and payload bits; R decides which of the two a
  # value is, and the encoding records that decision, nothing more.
  values[is.nan(values)] <- NaN
  values[is.na(values) & !is.nan(values)] <- NA_real_
  writeBin(values, raw(), size = 8L, endian = "big")
}

.content_hash_push <- function(state, bytes) {
  buffer <- c(state$carry, bytes)
  leaf_bytes <- .content_hash_contract$leaf_values * state$width
  n_leaf <- length(buffer) %/% leaf_bytes
  for (leaf in seq_len(n_leaf)) {
    at <- ((leaf - 1L) * leaf_bytes + 1L):(leaf * leaf_bytes)
    state$digest <- .content_hash_digest(c(charToRaw(state$digest), buffer[at]))
  }
  state$carry <- if (n_leaf * leaf_bytes < length(buffer)) {
    buffer[(n_leaf * leaf_bytes + 1L):length(buffer)]
  } else {
    raw()
  }
  state
}

.content_hash_finish <- function(state) {
  expected <- prod(as.double(state$shape))
  if (state$values != expected) {
    .frame_abort(
      sprintf(
        "Content hashing read %s values but the source declares %s.",
        format(state$values, scientific = FALSE),
        format(expected, scientific = FALSE)
      ),
      "fmridataset_error_source_contract",
      field = "shape",
      expected = expected,
      actual = state$values
    )
  }
  if (length(state$carry)) {
    state$digest <- .content_hash_digest(c(charToRaw(state$digest), state$carry))
  }
  state$digest
}

# Row blocks aligned to the source chunk grid. A block holds as many whole
# rows as fit the byte bound, rounded down to a multiple of the observation
# chunk when it holds at least one chunk. A row wider than the bound is read
# as consecutive feature ranges, aligned to the feature chunk the same way.
.content_hash_plan <- function(shape, chunks, width, block_bytes) {
  block_bytes <- .validate_budget_scalar(block_bytes, "block_bytes")
  capacity <- max(1, floor(block_bytes / width))
  n_row <- shape[[1L]]
  n_col <- shape[[2L]]
  if (!n_row || !n_col) {
    return(list(rows = integer(), cols = n_col))
  }
  rows <- max(1L, as.integer(min(n_row, floor(capacity / n_col))))
  if (rows >= chunks[[1L]]) rows <- (rows %/% chunks[[1L]]) * chunks[[1L]]
  cols <- n_col
  if (rows == 1L && n_col > capacity) {
    cols <- max(1L, as.integer(capacity))
    if (cols >= chunks[[2L]]) cols <- (cols %/% chunks[[2L]]) * chunks[[2L]]
  }
  list(rows = as.integer(rows), cols = as.integer(cols))
}

.content_hash_feed <- function(state, reader, block_bytes) {
  shape <- as.integer(source_shape(reader))
  if (!identical(shape[[2L]], state$shape[[2L]])) {
    .frame_abort(
      "Content hashing requires every streamed part to share the feature count.",
      "fmridataset_error_source_contract",
      field = "shape",
      expected = state$shape[[2L]],
      actual = shape[[2L]]
    )
  }
  chunks <- pmin(as.integer(source_chunks(reader)), pmax(1L, shape))
  plan <- .content_hash_plan(shape, chunks, state$width, block_bytes)
  if (!length(plan$rows) || !shape[[1L]] || !shape[[2L]]) {
    return(state)
  }
  row_starts <- seq.int(1L, shape[[1L]], by = plan$rows)
  col_starts <- seq.int(1L, shape[[2L]], by = plan$cols)
  for (row_start in row_starts) {
    row_end <- min(shape[[1L]], row_start + plan$rows - 1L)
    for (col_start in col_starts) {
      col_end <- min(shape[[2L]], col_start + plan$cols - 1L)
      block <- source_read(
        reader,
        observations = row_start:row_end,
        features = col_start:col_end
      )
      state$values <- state$values + as.double(length(block))
      state <- .content_hash_push(
        state, .content_hash_block_bytes(block, state$mode)
      )
    }
  }
  state
}
