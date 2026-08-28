.has_complete_feature_selection <- function(x) {
  selection <- .frame_selection(x)
  identical(selection$features, seq_len(ncol(selection$base)))
}

#' Select a matrix or spatial execution path
#'
#' Matrix operations always use bounded observation-by-feature blocks. Spatial
#' operations use a source's native-image capability only when the frame view
#' retains the complete feature domain; otherwise they reconstruct maps from
#' packed assay values through the frame's feature space.
#'
#' @param x An `fmri_frame` or view.
#' @param operation Either `"matrix"` or `"spatial"`.
#' @param assay Assay name.
#' @param path For spatial operations, one of `"auto"`, `"native"`, or
#'   `"reconstruct"`.
#' @return One of `"matrix"`, `"native"`, or `"reconstruct"`.
#' @export
execution_path <- function(
  x,
  operation = c("matrix", "spatial"),
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct")
) {
  if (!inherits(x, "fmri_frame")) {
    .frame_abort("x must be an fmri_frame or fmri_view.", "fmridataset_error_alignment")
  }
  operation <- match.arg(operation)
  path <- match.arg(path)
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  if (operation == "matrix") {
    return("matrix")
  }
  native_available <- "native_read" %in% source_capabilities(descriptor$source) &&
    .has_complete_feature_selection(x)
  if (path == "native" && !native_available) {
    .frame_abort(
      "A native spatial path is unavailable for this source or feature selection.",
      "fmridataset_error_backend_io",
      operation = "native_read",
      complete_feature_domain = .has_complete_feature_selection(x),
      capabilities = source_capabilities(descriptor$source)
    )
  }
  if (path == "native" || (path == "auto" && native_available)) {
    "native"
  } else {
    "reconstruct"
  }
}

.native_realization_values <- function(spatial) {
  if (inherits(spatial, "composite_space")) {
    return(sum(vapply(
      composite_parts(spatial),
      .native_realization_values, numeric(1)
    )))
  }
  if (inherits(spatial, "parcel_space") ||
    (inherits(spatial, "basis_space") && !is.null(basis_synthesis(spatial)))) {
    return(.native_realization_values(parent_space(spatial)))
  }
  shape <- native_shape(spatial)
  if (!is.numeric(shape) || anyNA(shape) || any(shape < 0)) {
    .frame_abort(
      "The feature space does not expose a valid native shape.",
      "fmridataset_error_space_mismatch"
    )
  }
  prod(as.double(shape))
}

.spatial_output_bytes <- function(x, n_map = 1L) {
  .native_realization_values(space(x)) * 8 * as.double(n_map)
}

.spatial_realization_cost <- function(x, n_map, assay, path) {
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  traits <- .source_cost_traits(descriptor$source)
  packed <- if (n_map > 0L && length(selection$observations)) {
    source_realization_cost(
      descriptor$source,
      observations = selection$observations[[1L]],
      features = selection$features
    )
  } else {
    .realization_cost_from_shape(
      c(0L, length(selection$features)),
      descriptor$dtype,
      already_realized = traits$already_realized,
      compressed = traits$compressed
    )
  }
  output_per_map <- .spatial_output_bytes(x, 1L)
  output_bytes <- output_per_map * as.double(n_map)

  # Reconstruction holds the packed row and its read buffers while allocating
  # the native map. The native fast path additionally holds the source-native
  # map while vectorizing and rebuilding the returned object.
  temporary_bytes <- packed$estimated_peak_bytes
  if (identical(path, "native") && n_map > 0L) {
    temporary_bytes <- temporary_bytes + output_per_map
  }
  structure(
    list(
      storage_dtype = descriptor$dtype,
      storage_bytes = packed$storage_bytes,
      realized_dtype = "double native map",
      realized_dtype_bytes = 8,
      estimated_output_bytes = output_bytes,
      packed_read_peak_bytes = packed$estimated_peak_bytes,
      native_input_bytes = if (identical(path, "native")) output_per_map else 0,
      estimated_temporary_bytes = temporary_bytes,
      estimated_peak_bytes = output_bytes + temporary_bytes
    ),
    class = "source_realization_cost"
  )
}

.assert_spatial_budget <- function(x, n_map, assay, path, memory_budget) {
  cost <- .spatial_realization_cost(x, n_map, assay, path)
  .assert_realization_budget(cost, memory_budget, "spatial realization")
  invisible(cost)
}

.one_native_map <- function(value) {
  if (methods::is(value, "NeuroVec")) {
    return(value[[1L]])
  }
  if (is.list(value)) {
    if (length(value) != 1L) {
      .frame_abort(
        "A one-observation native read returned multiple spatial objects.",
        "fmridataset_error_backend_io",
        operation = "native_read"
      )
    }
    return(value[[1L]])
  }
  value
}

.read_one_spatial_map <- function(x, position, assay, path) {
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  if (path == "native") {
    native <- .one_native_map(source_read_native(
      descriptor$source,
      observations = selection$observations[[position]]
    ))
    return(reconstruct_space(
      space(x),
      vectorize_space(space(x), native)
    ))
  }
  values <- source_read(
    descriptor$source,
    observations = selection$observations[[position]],
    features = selection$features
  )
  reconstruct_space(space(x), as.numeric(values))
}

#' Collect spatial maps through native or reconstructed reads
#'
#' @param x An `fmri_frame` or view.
#' @param observations Observation IDs or integer positions. The requested
#'   order is preserved. Duplicated selectors are rejected, as they are on
#'   every other frame axis selection.
#' @param assay Assay name.
#' @param path One of `"auto"`, `"native"`, or `"reconstruct"`.
#' @param memory_budget Maximum estimated peak bytes for all returned native
#'   maps plus the current packed read, conversion, and reconstruction buffers.
#' @return A named list with one native spatial object per observation.
#' @export
collect_spatial_maps <- function(
  x,
  observations = NULL,
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct"),
  memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)
) {
  positions <- .normalize_frame_selector(
    observations %||% seq_len(nrow(x)),
    observation_ids(x),
    "observation"
  )
  selected_path <- execution_path(x, operation = "spatial", assay = assay, path = path)
  .assert_spatial_budget(
    x, length(positions), assay, selected_path, memory_budget
  )
  out <- lapply(
    positions,
    function(position) .read_one_spatial_map(x, position, assay, selected_path)
  )
  names(out) <- observation_ids(x)[positions]
  out
}

#' Stream an operation over spatial maps
#'
#' Unlike `collect_spatial_maps()`, `execute_spatial()` holds only one input
#' map at a time. The callback result is retained, so callers remain
#' responsible for keeping returned values appropriately small.
#'
#' @param x An `fmri_frame` or view.
#' @param observations Observation IDs or integer positions.
#' @param FUN Function receiving `map` and `observation_id`.
#' @param ... Additional arguments passed to `FUN`.
#' @param assay Assay name.
#' @param path One of `"auto"`, `"native"`, or `"reconstruct"`.
#' @param memory_budget Maximum estimated peak bytes for one input spatial map
#'   plus its packed read, conversion, and reconstruction buffers.
#' @return A list of callback results in requested observation order.
#' @export
execute_spatial <- function(
  x,
  observations = NULL,
  FUN,
  ...,
  assay = active_assay(x),
  path = c("auto", "native", "reconstruct"),
  memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)
) {
  if (!is.function(FUN)) {
    .frame_abort("FUN must be a function.", "fmridataset_error_source_contract")
  }
  positions <- .normalize_frame_selector(
    observations %||% seq_len(nrow(x)),
    observation_ids(x),
    "observation"
  )
  selected_path <- execution_path(x, operation = "spatial", assay = assay, path = path)
  .assert_spatial_budget(
    x, min(1L, length(positions)), assay, selected_path, memory_budget
  )
  ids <- observation_ids(x)
  lapply(positions, function(position) {
    FUN(
      map = .read_one_spatial_map(x, position, assay, selected_path),
      observation_id = ids[[position]],
      ...
    )
  })
}
