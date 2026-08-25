.axis_digest <- function(x) .canonical_digest(axis_ids(x))

.frame_assay_source <- function(x, name) {
  source <- assay(x, name)$source
  if (inherits(x, "fmri_view")) {
    source_view(
      source,
      observations = x$observation_index,
      features = x$feature_index
    )
  } else {
    source
  }
}

#' Construct a strictly aligned assay set
#'
#' @param assays Named matrices or `ArraySource` descriptors.
#' @param observations Observation axis.
#' @param features Feature axis.
#' @return An `aligned_assay_set`.
#' @export
aligned_assay_set <- function(assays, observations, features) {
  if (!is.list(assays) || !length(assays) || is.null(names(assays)) || any(!nzchar(names(assays)))) {
    .frame_abort("assays must be a non-empty named list.", "fmridataset_error_alignment")
  }
  if (anyDuplicated(names(assays))) {
    .frame_abort("Assay names must be unique.", "fmridataset_error_alignment")
  }
  expected <- c(length(observations), length(features))
  obs_digest <- .axis_digest(observations)
  feature_digest <- .axis_digest(features)
  out <- lapply(names(assays), function(nm) {
    value <- assays[[nm]]
    source <- if (inherits(value, "aligned_assay")) value$source else as_array_source(value)
    validate_array_source(source)
    annotation <- if (is.list(value) && !inherits(value, "array_source")) value else list()
    if (!identical(as.integer(source_shape(source)), as.integer(expected))) {
      .frame_abort(
        sprintf("Assay '%s' shape does not match the frame axes.", nm),
        "fmridataset_error_alignment",
        assay = nm,
        expected = expected,
        actual = source_shape(source)
      )
    }
    structure(
      list(
        name = nm,
        source = source,
        dtype = source_dtype(source),
        observation_digest = obs_digest,
        feature_digest = feature_digest,
        role = annotation$role %||% NULL,
        units = annotation$units %||% NULL,
        metadata = annotation$metadata %||% list()
      ),
      class = "aligned_assay"
    )
  })
  names(out) <- names(assays)
  class(out) <- c("aligned_assay_set", "list")
  out
}

#' Construct a spatially typed annotated matrix
#'
#' @param assays Named matrices or serializable array sources.
#' @param observations Observation metadata or an observation `axis_frame`.
#' @param features Feature metadata or a spatial feature axis.
#' @param space Feature space used when `features` is not already spatial.
#' @param entities A named `entity_registry` or entries normalizable by
#'   `entity_registry()`.
#' @param relations Named relation registry.
#' @param tables Named auxiliary tables.
#' @param active_assay Active assay name.
#' @param metadata Frame metadata.
#' @param provenance Serializable provenance records.
#' @return An `fmri_frame`.
#' @export
fmri_frame <- function(assays, observations, features = NULL, space = NULL,
                       entities = list(), relations = list(), tables = list(),
                       active_assay = NULL, metadata = list(), provenance = NULL) {
  if (!is.list(assays) || !length(assays)) {
    .frame_abort("assays must be a non-empty list.", "fmridataset_error_alignment")
  }
  first <- if (inherits(assays[[1L]], "aligned_assay")) assays[[1L]]$source else as_array_source(assays[[1L]])
  shape <- source_shape(first)
  if (length(shape) != 2L) {
    .frame_abort("Frame assays must be two dimensional.", "fmridataset_error_alignment")
  }

  if (!inherits(observations, "axis_frame")) {
    observations <- axis_frame(observations, axis = "observation")
  }
  if (!identical(observations$axis, "observation")) {
    .frame_abort("observations must be an observation axis.", "fmridataset_error_alignment")
  }

  if (inherits(features, "spatial_axis_frame")) {
    feature_axis_value <- features
    space <- features$space
  } else {
    if (is.null(space)) {
      space <- index_space(shape[2L])
    }
    if (is.null(features)) features <- feature_data(space)
    feature_axis_value <- feature_axis(features, space = space)
  }

  if (length(observations) != shape[1L]) {
    .frame_abort("Observation metadata does not match assay rows.", "fmridataset_error_alignment")
  }
  if (length(feature_axis_value) != shape[2L]) {
    .frame_abort("Feature metadata does not match assay columns.", "fmridataset_error_alignment")
  }

  assay_values <- aligned_assay_set(assays, observations, feature_axis_value)
  active_assay <- active_assay %||% names(assay_values)[1L]
  if (!active_assay %in% names(assay_values)) {
    .frame_abort("active_assay is not present in assays.", "fmridataset_error_alignment")
  }
  entities <- entity_registry(entities)
  if (inherits(provenance, "provenance_graph")) {
    validate_provenance_graph(provenance)
  }
  if (.source_contains_runtime_state(provenance)) {
    .frame_abort(
      "Frame provenance cannot contain runtime state.",
      "fmridataset_error_feature_map",
      field = "provenance"
    )
  }
  relations <- .resolve_relation_registry(
    relation_registry(relations),
    observations,
    feature_axis_value,
    entities
  )

  structure(
    list(
      assays = assay_values,
      observations = observations,
      features = feature_axis_value,
      entities = entities,
      relations = relations,
      tables = tables,
      active_assay = active_assay,
      metadata = metadata,
      provenance = provenance,
      schema_version = 1L
    ),
    class = c("fmri_frame", "fmri_dataset")
  )
}

#' Frame accessors
#'
#' @param x An `fmri_frame` or `fmri_view`.
#' @param resolve Whether to append reachable, namespaced entity annotations or
#'   lazily lifted entity blocks.
#' @param ... Additional method arguments.
#' @name frame-accessors
NULL

#' @rdname frame-accessors
#' @export
assays <- function(x, ...) UseMethod("assays")
#' @export
assays.fmri_frame <- function(x, ...) x$assays

#' @rdname frame-accessors
#' @param name Assay name.
#' @export
assay <- function(x, name = active_assay(x), ...) UseMethod("assay")
#' @export
assay.fmri_frame <- function(x, name = active_assay(x), ...) {
  value <- assays(x)[[name]]
  if (is.null(value)) {
    .frame_abort(sprintf("Unknown assay '%s'.", name), "fmridataset_error_alignment")
  }
  value
}

#' @rdname frame-accessors
#' @export
active_assay <- function(x, ...) UseMethod("active_assay")
#' @export
active_assay.fmri_frame <- function(x, ...) x$active_assay

#' @rdname frame-accessors
#' @export
observation_axis <- function(x, ...) UseMethod("observation_axis")
#' @export
observation_axis.fmri_frame <- function(x, ...) x$observations

#' @rdname frame-accessors
#' @export
observations <- function(x, resolve = FALSE, ...) UseMethod("observations")
#' @export
observations.fmri_frame <- function(x, resolve = FALSE, ...) {
  resolve <- .validate_resolve_flag(resolve)
  if (resolve) .resolved_observation_data(x) else axis_data(observation_axis(x))
}

#' @export
entities.fmri_frame <- function(x, ...) x$entities

#' @export
entity.fmri_frame <- function(x, name, ...) entity(entities(x), name)

#' @export
relations.fmri_frame <- function(x, ...) x$relations

#' @export
relation.fmri_frame <- function(x, name, ...) relation(relations(x), name)

#' @rdname frame-accessors
#' @export
features <- function(x, ...) UseMethod("features")
#' @export
features.fmri_frame <- function(x, ...) axis_data(feature_axis(x))

#' @rdname frame-accessors
#' @export
observation_ids <- function(x, ...) UseMethod("observation_ids")
#' @export
observation_ids.fmri_frame <- function(x, ...) axis_ids(observation_axis(x))

#' @export
feature_ids.fmri_frame <- function(x, ...) axis_ids(feature_axis(x))

#' @rdname frame-accessors
#' @export
obs_blocks <- function(x, resolve = FALSE, ...) UseMethod("obs_blocks")
#' @export
obs_blocks.fmri_frame <- function(x, resolve = FALSE, ...) {
  resolve <- .validate_resolve_flag(resolve)
  if (resolve) .resolved_observation_blocks(x) else axis_blocks(observation_axis(x))
}

#' @rdname frame-accessors
#' @export
feature_blocks <- function(x, ...) UseMethod("feature_blocks")
#' @export
feature_blocks.fmri_frame <- function(x, ...) axis_blocks(feature_axis(x))

#' Feature-space accessor
#'
#' @param x An object with spatial identity.
#' @param ... Additional arguments.
#' @return A `FeatureSpace`.
#' @export
space <- function(x, ...) UseMethod("space")
#' @export
space.fmri_frame <- function(x, ...) feature_axis(x)$space
#' @export
space.spatial_axis_frame <- function(x, ...) x$space
#' @export
space.default <- function(x, ...) neuroim2::space(x, ...)

#' @rdname frame-accessors
#' @export
dim.fmri_frame <- function(x) c(length(observation_axis(x)), length(feature_axis(x)))
#' @rdname frame-accessors
#' @export
nrow.fmri_frame <- function(x) dim(x)[1L]
#' @rdname frame-accessors
#' @export
ncol.fmri_frame <- function(x) dim(x)[2L]

#' @export
print.fmri_frame <- function(x, ...) {
  cat("<fmri_frame>", dim(x)[1L], "observations x", dim(x)[2L], "features\n")
  cat("  assays:", paste(names(assays(x)), collapse = ", "), "\n")
  cat("  active:", active_assay(x), "\n")
  cat("  space:", class(space(x))[1L], substr(space_digest(space(x)), 1L, 12L), "\n")
  invisible(x)
}

.frame_selection <- function(x) {
  if (inherits(x, "fmri_view")) {
    list(base = x$base, observations = x$observation_index, features = x$feature_index)
  } else {
    list(base = x, observations = seq_len(nrow(x)), features = seq_len(ncol(x)))
  }
}

#' Collect one frame assay under an explicit memory budget
#'
#' @param x An `fmri_frame` or view.
#' @param assay Assay name.
#' @param memory_budget Maximum output bytes.
#' @param force Allow collection above the budget.
#' @return A dense matrix.
#' @export
collect_assay <- function(x, assay = active_assay(x),
                          memory_budget = getOption("fmridataset.collect_budget", 2 * 1024^3),
                          force = FALSE) {
  selection <- .frame_selection(x)
  descriptor <- assay(selection$base, assay)
  # Realized width, not storage width: the result is an R matrix.
  bytes <- length(selection$observations) * length(selection$features) *
    .realized_dtype_bytes(descriptor$dtype)
  if (!isTRUE(force) && bytes > memory_budget) {
    .frame_abort(
      sprintf(
        paste0(
          "Collecting this assay requires %s bytes (%d x %d %s values ",
          "realized as R doubles), above the memory_budget of %s bytes. ",
          "Raise memory_budget, select fewer rows or columns, or pass force = TRUE."
        ),
        format(bytes, scientific = FALSE),
        length(selection$observations), length(selection$features),
        descriptor$dtype,
        format(memory_budget, scientific = FALSE)
      ),
      "fmridataset_error_budget",
      required_bytes = bytes,
      memory_budget = memory_budget
    )
  }
  source_read(
    descriptor$source,
    observations = selection$observations,
    features = selection$features
  )
}

#' Apply a function to bounded feature blocks
#'
#' @param x An `fmri_frame` or view.
#' @param FUN Function receiving an observation-by-feature matrix and feature IDs.
#' @param block_size Number of features per block.
#' @param assay Assay name.
#' @param ... Additional arguments passed to `FUN`.
#' @return A list of block results.
#' @export
block_apply <- function(x, FUN, block_size = 4096L, assay = active_assay(x), ...) {
  block_size <- as.integer(block_size)
  if (length(block_size) != 1L || is.na(block_size) || block_size <= 0L) {
    .frame_abort("block_size must be positive.", "fmridataset_error_budget")
  }
  ids <- feature_ids(x)
  # An empty feature selection is a valid frame; it yields no blocks rather than
  # tripping seq.int()'s wrong-sign error on seq.int(1L, 0L, by = block_size).
  starts <- seq.int(1L, by = block_size, length.out = ceiling(length(ids) / block_size))
  lapply(starts, function(start) {
    idx <- start:min(length(ids), start + block_size - 1L)
    FUN(collect_assay(x[, idx], assay = assay), ids[idx], ...)
  })
}

#' Recover a spatial map for one observation
#'
#' @param x An `fmri_frame` or view.
#' @param observation Observation ID or one integer position.
#' @param assay Assay name.
#' @return A reconstructed spatial object.
#' @export
spatial_map <- function(x, observation, assay = active_assay(x)) {
  index <- .normalize_frame_selector(observation, observation_ids(x), "observation")
  if (length(index) != 1L) {
    .frame_abort("spatial_map requires one observation.", "fmridataset_error_alignment")
  }
  collect_spatial_maps(x, observations = index, assay = assay)[[1L]]
}

#' Explain a frame without reading assay values
#'
#' @param x An `fmri_frame` or view.
#' @return A serializable execution summary.
#' @export
explain <- function(x) {
  list(
    class = class(x)[1L],
    shape = dim(x),
    assays = lapply(assays(x), function(a) {
      list(
        dtype = a$dtype,
        chunks = source_chunks(a$source),
        capabilities = source_capabilities(a$source),
        fingerprint = source_fingerprint(a$source)
      )
    }),
    observation_ids = observation_ids(x),
    feature_ids = feature_ids(x),
    space_digest = space_digest(space(x))
  )
}

.assert_bind_agreement <- function(reference, candidate, what) {
  if (isTRUE(all.equal(reference, candidate))) {
    return(invisible(TRUE))
  }
  .frame_abort(
    sprintf(
      "Frames disagree on %s; bind_observations() cannot choose between them.",
      what
    ),
    "fmridataset_error_alignment",
    field = what
  )
}

# Return a block's data with its component axis permuted into `proto`'s
# component order, refusing any block whose component identities differ.
# Binding rbinds these positionally, so a block whose components are merely
# ordered differently would otherwise file each frame's values under the
# previous frame's labels.
.aligned_block_data <- function(block, proto, block_name) {
  ref <- block_components(proto)
  cur <- block_components(block)
  ref_ids <- ref$.component_id
  cur_ids <- cur$.component_id

  if (length(cur_ids) != length(ref_ids) || !setequal(cur_ids, ref_ids)) {
    .frame_abort(
      sprintf(
        "Block %s has different components across bound frames (%s vs %s).",
        encodeString(block_name, quote = "\""),
        paste(ref_ids, collapse = ", "),
        paste(cur_ids, collapse = ", ")
      ),
      "fmridataset_error_alignment",
      block = block_name,
      expected = ref_ids,
      actual = cur_ids
    )
  }

  perm <- match(ref_ids, cur_ids)
  reordered <- cur[perm, , drop = FALSE]
  if (!isTRUE(all.equal(as.data.frame(reordered), as.data.frame(ref)))) {
    .frame_abort(
      sprintf(
        "Block %s has conflicting component metadata across bound frames.",
        encodeString(block_name, quote = "\"")
      ),
      "fmridataset_error_alignment",
      block = block_name
    )
  }

  .permute_block_columns(axis_block_data(block), perm)
}

.permute_block_columns <- function(data, perm) {
  if (identical(perm, seq_along(perm))) {
    return(data)
  }
  if (inherits(data, "array_source")) {
    return(source_view(data, features = perm))
  }
  data[, perm, drop = FALSE]
}

.bind_axis_frames <- function(xs) {
  first <- xs[[1L]]
  if (length(first$blocks)) {
    block_names <- names(first$blocks)
    if (!all(vapply(xs, function(x) identical(names(x$blocks), block_names), logical(1)))) {
      .frame_abort("Bound axes must have identical block names.", "fmridataset_error_alignment")
    }
    blocks <- lapply(block_names, function(nm) {
      proto <- first$blocks[[nm]]
      # Block data is row-bound positionally, so the component axes must be
      # brought into a common order FIRST. Aligning by component ID rather than
      # by column position is what keeps values under the label they belong to.
      values <- lapply(xs, function(x) .aligned_block_data(x$blocks[[nm]], proto, nm))
      if (any(vapply(values, inherits, logical(1), what = "array_source"))) {
        source <- row_bound_source(lapply(values, as_array_source))
      } else {
        source <- do.call(rbind, values)
      }
      axis_block(source, proto$components, proto$role, proto$units, proto$metadata)
    })
    names(blocks) <- block_names
  } else {
    blocks <- list()
  }
  data <- do.call(rbind, lapply(xs, axis_data))
  axis_frame(data, blocks = blocks, id = data[[first$id_col]], axis = first$axis, id_col = first$id_col)
}

#' Bind frames along observations
#'
#' @param ... Frames with identical feature IDs, spaces, and assay names.
#' @return A lazily row-bound `fmri_frame`.
#' @export
bind_observations <- function(...) {
  xs <- list(...)
  if (!length(xs)) .frame_abort("At least one frame is required.", "fmridataset_error_alignment")
  first <- xs[[1L]]
  for (x in xs[-1L]) {
    assert_compatible_space(space(first), space(x))
    if (!identical(feature_ids(first), feature_ids(x)) || !identical(names(assays(first)), names(assays(x)))) {
      .frame_abort("Frames have incompatible feature or assay identities.", "fmridataset_error_alignment")
    }
    if (!identical(entity_registry_digest(first), entity_registry_digest(x))) {
      .entity_abort(
        "Frames have incompatible entity registries.",
        operation = "bind_observations"
      )
    }
    # The bound frame keeps the first frame's feature annotations, tables, and
    # metadata. That is only sound when the others agree; otherwise the result
    # would silently assert the first frame's description of shared objects.
    .assert_bind_agreement(
      axis_data(feature_axis(first)), axis_data(feature_axis(x)),
      "feature metadata"
    )
    .assert_bind_agreement(first$tables, x$tables, "tables")
    .assert_bind_agreement(first$metadata, x$metadata, "metadata")
  }
  relation_values <- .bind_relation_registries(lapply(xs, relations))
  obs <- .bind_axis_frames(lapply(xs, observation_axis))
  if (anyDuplicated(axis_ids(obs))) {
    .frame_abort("Observation IDs collide across frames.", "fmridataset_error_alignment")
  }
  assay_sources <- lapply(names(assays(first)), function(nm) {
    row_bound_source(lapply(xs, function(x) .frame_assay_source(x, nm)))
  })
  names(assay_sources) <- names(assays(first))
  fmri_frame(
    assays = assay_sources,
    observations = obs,
    features = feature_axis(first),
    entities = entities(first),
    relations = relation_values,
    tables = first$tables,
    active_assay = active_assay(first),
    metadata = first$metadata,
    provenance = first$provenance
  )
}
