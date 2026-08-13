.zarr_supported_version <- "0.4.2"

.zarr_assert_available <- function() {
  if (!requireNamespace("zarr", quietly = TRUE)) {
    .frame_abort(
      "The optional zarr package is required to open a Zarr ArraySource.",
      "fmridataset_error_backend_io",
      operation = "open",
      package = "zarr"
    )
  }
  installed <- utils::packageVersion("zarr")
  if (installed < .zarr_supported_version) {
    .frame_abort(
      sprintf(
        "Zarr ArraySource requires zarr >= %s; version %s is installed.",
        .zarr_supported_version,
        installed
      ),
      "fmridataset_error_backend_io",
      operation = "open",
      package = "zarr"
    )
  }
  invisible(TRUE)
}

.zarr_provider_open <- function(uri, array_path) {
  .zarr_assert_available()
  store <- tryCatch(
    zarr::open_zarr(uri, read_only = TRUE),
    error = function(error) {
      if (inherits(error, "fmridataset_error")) stop(error)
      .frame_abort(
        sprintf("Failed to open Zarr store: %s", conditionMessage(error)),
        "fmridataset_error_backend_io",
        operation = "open",
        uri = uri
      )
    }
  )
  runtime <- new.env(parent = emptyenv())
  runtime$store <- store
  runtime$array <- NULL
  runtime$closed <- FALSE
  keep <- FALSE
  on.exit(if (!keep) .zarr_provider_close(runtime), add = TRUE)
  array <- store[[array_path]]
  if (!inherits(array, "zarr_array")) {
    .frame_abort(
      sprintf("Zarr node '%s' is missing or is not an array.", array_path),
      "fmridataset_error_backend_io",
      operation = "open",
      uri = uri,
      array_path = array_path
    )
  }
  runtime$array <- array
  keep <- TRUE
  runtime
}

.zarr_normalize_dtype <- function(dtype) {
  if (is.list(dtype) && !is.null(dtype$data_type)) dtype <- dtype$data_type
  dtype <- as.character(dtype)[[1L]]
  aliases <- c(
    bool = "logical", boolean = "logical",
    byte = "uint8", double = "float64", float = "float32",
    complex = "complex128"
  )
  if (dtype %in% names(aliases)) dtype <- unname(aliases[[dtype]])
  dtype
}

.zarr_provider_metadata <- function(handle) {
  array <- handle$array
  list(
    shape = as.integer(array$shape),
    chunks = as.integer(array$chunking$chunk_shape),
    dtype = .zarr_normalize_dtype(array$data_type$data_type)
  )
}

.zarr_provider_read <- function(handle, selection) {
  handle$array$read(selection)
}

.zarr_provider_close <- function(handle) {
  if (is.environment(handle) && !isTRUE(handle$closed)) {
    store <- handle$store
    close_method <- tryCatch(store$close, error = function(error) NULL)
    if (is.function(close_method)) close_method()
    handle$array <- NULL
    handle$store <- NULL
    handle$closed <- TRUE
  }
  invisible(TRUE)
}

.zarr_physical_metadata <- function(uri, array_path) {
  runtime <- .zarr_provider_open(uri, array_path)
  on.exit(.zarr_provider_close(runtime), add = TRUE)
  .zarr_provider_metadata(runtime)
}

.zarr_validate_metadata <- function(metadata) {
  if (!is.list(metadata) || !identical(sort(names(metadata)), sort(c("shape", "chunks", "dtype")))) {
    .frame_abort(
      "Zarr metadata must describe shape, chunks, and dtype.",
      "fmridataset_error_source_contract",
      field = "metadata"
    )
  }
  shape <- as.integer(metadata$shape)
  chunks <- as.integer(metadata$chunks)
  if (length(shape) != 2L || anyNA(shape) || any(shape < 0L)) {
    .frame_abort(
      "A Zarr frame source must be a two-dimensional array.",
      "fmridataset_error_source_contract",
      field = "shape",
      actual = metadata$shape
    )
  }
  if (length(chunks) != 2L || anyNA(chunks) || any(chunks <= 0L) ||
    any(chunks > pmax(1L, shape))) {
    .frame_abort(
      "Zarr chunks must be two positive integers bounded by the physical shape.",
      "fmridataset_error_source_contract",
      field = "chunks",
      actual = metadata$chunks,
      shape = shape
    )
  }
  dtype <- .zarr_normalize_dtype(metadata$dtype)
  .dtype_bytes(dtype)
  list(shape = shape, chunks = chunks, dtype = dtype)
}

.zarr_logical_order <- function(values, physical_axes) {
  as.integer(values[match(c("observation", "feature"), physical_axes)])
}

.zarr_physical_order <- function(values, physical_axes) {
  as.integer(values[match(physical_axes, c("observation", "feature"))])
}

#' Construct an experimental Zarr array source
#'
#' `zarr_array_source()` describes one two-dimensional Zarr array whose logical
#' axes are observations and features. The descriptor contains no open store,
#' R6 object, external pointer, or loader function. Runtime handles are opened
#' only for metadata discovery and numerical reads.
#'
#' The optional `zarr` package is needed only when metadata
#' must be discovered or data are read. Supplying `shape`, `dtype`, and `chunks`
#' together therefore permits metadata-only construction and serialization on
#' workers where Zarr is not installed.
#'
#' @param uri One local path, file URI, or HTTP(S) location understood by
#'   `zarr::open_zarr()`.
#' @param array_path Absolute path of the array within the Zarr hierarchy. The
#'   default `"/"` denotes a single-array store.
#' @param shape Optional logical observation-by-feature shape.
#' @param dtype Optional logical storage dtype supported by `ArraySource`.
#' @param chunks Optional logical observation-by-feature chunk shape.
#' @param physical_axes Names of the two physical Zarr dimensions, permitting
#'   either observation-first or feature-first storage.
#' @return A serializable `zarr_array_source` descriptor.
#' @export
zarr_array_source <- function(uri, array_path = "/", shape = NULL,
                              dtype = NULL, chunks = NULL,
                              physical_axes = c("observation", "feature")) {
  if (!is.character(uri) || length(uri) != 1L || is.na(uri) || !nzchar(uri)) {
    .frame_abort(
      "uri must be one non-empty character string.",
      "fmridataset_error_source_contract",
      field = "uri"
    )
  }
  if (!is.character(array_path) || length(array_path) != 1L ||
    is.na(array_path) || !startsWith(array_path, "/")) {
    .frame_abort(
      "array_path must be one absolute Zarr node path beginning with '/'.",
      "fmridataset_error_source_contract",
      field = "array_path"
    )
  }
  if (!is.character(physical_axes) || length(physical_axes) != 2L ||
    !identical(sort(physical_axes), c("feature", "observation"))) {
    .frame_abort(
      "physical_axes must be a permutation of observation and feature.",
      "fmridataset_error_source_contract",
      field = "physical_axes",
      actual = physical_axes
    )
  }

  supplied <- c(!is.null(shape), !is.null(dtype), !is.null(chunks))
  if (any(supplied) && !all(supplied)) {
    .frame_abort(
      "shape, dtype, and chunks must be provided together or all discovered from the store.",
      "fmridataset_error_source_contract",
      field = "metadata"
    )
  }
  if (all(supplied)) {
    logical_metadata <- .zarr_validate_metadata(list(
      shape = shape,
      chunks = chunks,
      dtype = dtype
    ))
  } else {
    physical_metadata <- .zarr_validate_metadata(
      .zarr_physical_metadata(uri, array_path)
    )
    logical_metadata <- list(
      shape = .zarr_logical_order(physical_metadata$shape, physical_axes),
      chunks = .zarr_logical_order(physical_metadata$chunks, physical_axes),
      dtype = physical_metadata$dtype
    )
  }

  out <- structure(
    list(
      uri = uri,
      array_path = array_path,
      shape = logical_metadata$shape,
      dtype = logical_metadata$dtype,
      chunks = logical_metadata$chunks,
      physical_axes = physical_axes,
      capabilities = c(
        "row_slice", "column_slice", "block_slice", "serializable"
      ),
      schema_version = 1L,
      experimental = TRUE
    ),
    class = c("zarr_array_source", "array_source")
  )
  out$fingerprint <- .canonical_digest(list(
    type = "zarr_array_source",
    schema_version = out$schema_version,
    uri = out$uri,
    array_path = out$array_path,
    shape = out$shape,
    dtype = out$dtype,
    chunks = out$chunks,
    physical_axes = out$physical_axes
  ))
  validate_array_source(out)
  out
}

#' @export
source_shape.zarr_array_source <- function(x, ...) x$shape
#' @export
source_dtype.zarr_array_source <- function(x, ...) x$dtype
#' @export
source_chunks.zarr_array_source <- function(x, ...) x$chunks
#' @export
source_capabilities.zarr_array_source <- function(x, ...) x$capabilities
#' @export
source_fingerprint.zarr_array_source <- function(x, ...) x$fingerprint

.zarr_assert_fresh <- function(source, metadata) {
  metadata <- .zarr_validate_metadata(metadata)
  expected_shape <- .zarr_physical_order(source$shape, source$physical_axes)
  expected_chunks <- .zarr_physical_order(source$chunks, source$physical_axes)
  if (!identical(metadata$shape, expected_shape) ||
    !identical(metadata$chunks, expected_chunks) ||
    !identical(metadata$dtype, source$dtype)) {
    .frame_abort(
      "Zarr array metadata changed after the source descriptor was created.",
      "fmridataset_error_source_stale",
      operation = "open",
      expected = list(
        shape = expected_shape,
        chunks = expected_chunks,
        dtype = source$dtype
      ),
      actual = metadata
    )
  }
  invisible(TRUE)
}

#' @export
source_open.zarr_array_source <- function(x, ...) {
  runtime <- .zarr_provider_open(x$uri, x$array_path)
  keep <- FALSE
  on.exit(if (!keep) .zarr_provider_close(runtime), add = TRUE)
  .zarr_assert_fresh(x, .zarr_provider_metadata(runtime))
  handle <- structure(
    list(source = x, runtime = runtime),
    class = c("zarr_array_source_handle", "array_source_handle")
  )
  keep <- TRUE
  handle
}

.zarr_consecutive_runs <- function(index) {
  index <- sort(unique(as.integer(index)))
  if (!length(index)) {
    return(list())
  }
  split(index, cumsum(c(TRUE, diff(index) != 1L)))
}

.zarr_empty_matrix <- function(dtype, nrow, ncol) {
  prototype <- if (identical(dtype, "logical")) {
    logical()
  } else if (startsWith(dtype, "complex")) {
    complex()
  } else {
    numeric()
  }
  matrix(prototype, nrow = nrow, ncol = ncol)
}

.zarr_read_handle <- function(handle, observations, features) {
  source <- handle$source
  runtime <- handle$runtime
  if (!is.environment(runtime) || isTRUE(runtime$closed)) {
    .frame_abort(
      "Zarr source handle is closed.",
      "fmridataset_error_backend_io",
      operation = "read"
    )
  }
  observations <- .normalize_source_index(observations, source$shape[[1L]])
  features <- .normalize_source_index(features, source$shape[[2L]])
  if (!length(observations) || !length(features)) {
    return(.zarr_empty_matrix(
      source$dtype,
      length(observations),
      length(features)
    ))
  }

  observation_index <- sort(unique(observations))
  feature_index <- sort(unique(features))
  selected <- .zarr_empty_matrix(
    source$dtype,
    length(observation_index),
    length(feature_index)
  )
  observation_runs <- .zarr_consecutive_runs(observation_index)
  feature_runs <- .zarr_consecutive_runs(feature_index)
  logical_axes <- c("observation", "feature")

  for (observation_run in observation_runs) {
    for (feature_run in feature_runs) {
      logical_selection <- list(
        observation = range(observation_run),
        feature = range(feature_run)
      )
      physical_selection <- unname(
        logical_selection[match(source$physical_axes, logical_axes)]
      )
      block <- .zarr_provider_read(runtime, physical_selection)
      physical_dim <- vapply(physical_selection, function(axis) {
        as.integer(diff(axis) + 1L)
      }, integer(1))
      block <- array(block, dim = physical_dim)
      if (identical(source$physical_axes, c("feature", "observation"))) {
        block <- t(block)
      }
      selected[
        match(observation_run, observation_index),
        match(feature_run, feature_index)
      ] <- block
    }
  }
  selected[
    match(observations, observation_index),
    match(features, feature_index),
    drop = FALSE
  ]
}

#' @export
source_read.zarr_array_source <- function(x, observations = NULL,
                                          features = NULL, ...) {
  handle <- source_open(x)
  on.exit(source_close(handle), add = TRUE)
  source_read(handle, observations = observations, features = features, ...)
}

#' @export
source_read.zarr_array_source_handle <- function(x, observations = NULL,
                                                 features = NULL, ...) {
  .zarr_read_handle(x, observations, features)
}

#' @export
source_read_native.zarr_array_source <- function(x, observations = NULL, ...) {
  .frame_abort(
    "zarr_array_source has no native spatial read path.",
    "fmridataset_error_backend_io",
    operation = "native_read"
  )
}

#' @export
source_close.zarr_array_source <- function(x, ...) invisible(TRUE)

#' @export
source_close.zarr_array_source_handle <- function(x, ...) {
  .zarr_provider_close(x$runtime)
}
