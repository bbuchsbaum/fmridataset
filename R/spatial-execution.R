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
    path = c("auto", "native", "reconstruct")) {
  if (!inherits(x, "fmri_frame")) {
    .frame_abort("x must be an fmri_frame or fmri_view.", "fmridataset_error_alignment")
  }
  operation <- match.arg(operation)
  path <- match.arg(path)
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  if (operation == "matrix") return("matrix")
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

.spatial_output_bytes <- function(x, n_map = 1L) {
  shape <- native_shape(space(x))
  if (!is.numeric(shape) || anyNA(shape) || any(shape < 0)) {
    .frame_abort(
      "The feature space does not expose a valid native shape.",
      "fmridataset_error_space_mismatch"
    )
  }
  prod(as.double(shape)) * 8 * as.double(n_map)
}

.assert_spatial_budget <- function(x, n_map, memory_budget) {
  memory_budget <- .validate_budget_scalar(memory_budget, "memory_budget")
  bytes <- .spatial_output_bytes(x, n_map)
  if (bytes > memory_budget) {
    .frame_abort(
      sprintf(
        "Spatial realization requires at least %s bytes, above memory_budget.",
        format(bytes, scientific = FALSE)
      ),
      "fmridataset_error_budget",
      required_bytes = bytes,
      memory_budget = memory_budget,
      n_map = n_map
    )
  }
  invisible(bytes)
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
#'   order and duplicates are preserved.
#' @param assay Assay name.
#' @param path One of `"auto"`, `"native"`, or `"reconstruct"`.
#' @param memory_budget Maximum estimated bytes for all returned native maps.
#' @return A named list with one native spatial object per observation.
#' @export
collect_spatial_maps <- function(
    x,
    observations = NULL,
    assay = active_assay(x),
    path = c("auto", "native", "reconstruct"),
    memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)) {
  positions <- .normalize_frame_selector(
    observations %||% seq_len(nrow(x)),
    observation_ids(x),
    "observation"
  )
  selected_path <- execution_path(x, operation = "spatial", assay = assay, path = path)
  .assert_spatial_budget(x, length(positions), memory_budget)
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
#' @param memory_budget Maximum estimated bytes for one input spatial map.
#' @return A list of callback results in requested observation order.
#' @export
execute_spatial <- function(
    x,
    observations = NULL,
    FUN,
    ...,
    assay = active_assay(x),
    path = c("auto", "native", "reconstruct"),
    memory_budget = getOption("fmridataset.spatial_budget", 512 * 1024^2)) {
  if (!is.function(FUN)) {
    .frame_abort("FUN must be a function.", "fmridataset_error_source_contract")
  }
  positions <- .normalize_frame_selector(
    observations %||% seq_len(nrow(x)),
    observation_ids(x),
    "observation"
  )
  selected_path <- execution_path(x, operation = "spatial", assay = assay, path = path)
  .assert_spatial_budget(x, min(1L, length(positions)), memory_budget)
  ids <- observation_ids(x)
  lapply(positions, function(position) {
    FUN(
      map = .read_one_spatial_map(x, position, assay, selected_path),
      observation_id = ids[[position]],
      ...
    )
  })
}
