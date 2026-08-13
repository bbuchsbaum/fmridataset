.axis_digest <- function(x) .canonical_digest(axis_ids(x))

.frame_assay_source <- function(x, name) {
  assay(x, name)$source
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
#' @param tables Named typed tables created by `event_table()` or
#'   `auxiliary_table()`.
#' @param active_assay Active assay name.
#' @param metadata Unaligned frame-level record. Aligned values belong on an
#'   axis, entity, block, assay, relation, typed table, or linked frame.
#' @param provenance `NULL` or a validated `provenance_graph`.
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
      # Shape alone cannot establish feature identity. Exploratory construction
      # remains possible only as an explicitly ephemeral space; persistence and
      # semantic certification reject it until a durable domain is supplied.
      space <- index_space(shape[2L], id_policy = "ephemeral")
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
  relations <- .resolve_relation_registry(
    relation_registry(relations),
    observations,
    feature_axis_value,
    entities
  )
  tables <- .validate_table_registry(tables, "Frame")
  metadata <- .normalize_container_metadata(
    metadata,
    .frame_alignment_domains(observations, feature_axis_value, entities)
  )
  provenance <- .validate_container_provenance(provenance, "Frame")

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
    class = "fmri_frame"
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

#' @export
ids_are_durable.fmri_frame <- function(x) {
  ids_are_durable(observation_axis(x)) && ids_are_durable(feature_axis(x)$space)
}

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
  list(
    base = x,
    observations = seq_len(nrow(x)),
    features = seq_len(ncol(x))
  )
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
  bytes <- length(selection$observations) * length(selection$features) * .dtype_bytes(descriptor$dtype)
  if (!isTRUE(force) && bytes > memory_budget) {
    .frame_abort(
      sprintf("Collecting this assay requires %s bytes, above memory_budget.", format(bytes, scientific = FALSE)),
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
  starts <- seq.int(1L, length(ids), by = block_size)
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

.explain_ids <- function(values, mode, sample_size) {
  if (identical(mode, "complete")) {
    return(list(mode = mode, values = values))
  }
  if (identical(mode, "none") || !length(values) || sample_size == 0L) {
    return(list(mode = mode, values = values[integer()]))
  }
  leading <- seq_len(min(sample_size, length(values)))
  trailing <- seq.int(max(1L, length(values) - sample_size + 1L), length(values))
  list(mode = mode, values = values[unique(c(leading, trailing))])
}

.explain_source_type <- function(source) {
  while (inherits(source, "counting_source") || inherits(source, "fault_source")) {
    source <- source$source
  }
  class(source)[1L]
}

#' Explain a frame without reading numerical values
#'
#' `explain()` returns a bounded, serializable summary of the visible frame.
#' Axis IDs are sampled by default so inspecting a large study does not create
#' another large object. Set `ids = "complete"` only when every visible ID is
#' required.
#'
#' The schema digest describes column, block, assay, relation, entity, table,
#' and space contracts. The semantic digest covers the complete source-free FDS
#' manifest. Physical source fingerprints are reported separately.
#'
#' @param x An `fmri_frame` or view.
#' @param ids One of `"sample"`, `"none"`, or `"complete"`.
#' @param sample_size Number of IDs sampled from each end of each axis.
#' @return A bounded serializable execution summary. No assay or aligned-block
#'   values are read.
#' @export
explain <- function(x, ids = c("sample", "none", "complete"), sample_size = 3L) {
  if (!inherits(x, "fmri_frame")) {
    .frame_abort("x must be an fmri_frame or view.", "fmridataset_error_alignment")
  }
  ids <- match.arg(ids)
  sample_size <- as.integer(sample_size)
  if (length(sample_size) != 1L || is.na(sample_size) || sample_size < 0L) {
    .frame_abort("sample_size must be one non-negative integer.", "fmridataset_error_alignment")
  }
  assay_values <- assays(x)
  assay_summary <- lapply(assay_values, function(value) {
    source <- value$source
    shape <- source_shape(source)
    list(
      source_type = .explain_source_type(source),
      shape = shape,
      dtype = value$dtype,
      chunks = source_chunks(source),
      capabilities = source_capabilities(source),
      fingerprint = source_fingerprint(source),
      realization_bytes = prod(as.double(shape)) * .dtype_bytes(value$dtype)
    )
  })
  observation_values <- observation_ids(x)
  feature_values <- feature_ids(x)
  manifest <- fds_frame_manifest(x)
  spatial <- space(x)
  list(
    class = class(x)[1L],
    schema_version = x$schema_version %||% x$base$schema_version,
    shape = stats::setNames(as.integer(dim(x)), c("observation", "feature")),
    active_assay = active_assay(x),
    counts = list(
      assays = length(assay_values),
      observation_blocks = length(obs_blocks(x)),
      feature_blocks = length(feature_blocks(x)),
      entities = length(entities(x)),
      relations = length(relations(x)),
      tables = length(x$tables %||% x$base$tables)
    ),
    digests = list(
      schema = frame_schema_digest(x),
      semantic = fds_manifest_digest(manifest),
      observation = .canonical_digest(observation_values),
      feature = .canonical_digest(feature_values)
    ),
    axes = list(
      observation = list(
        count = length(observation_values),
        ids = .explain_ids(observation_values, ids, sample_size)
      ),
      feature = list(
        count = length(feature_values),
        ids = .explain_ids(feature_values, ids, sample_size)
      )
    ),
    space = list(
      type = class(spatial)[1L],
      features = n_features(spatial),
      digest = space_digest(spatial)
    ),
    assays = assay_summary,
    realization = list(
      active_assay_bytes = assay_summary[[active_assay(x)]]$realization_bytes,
      all_assays_bytes = sum(vapply(
        assay_summary, function(value) value$realization_bytes, numeric(1)
      ))
    )
  )
}

.bind_axis_frames <- function(xs) {
  first <- xs[[1L]]
  if (length(first$blocks)) {
    block_names <- names(first$blocks)
    if (!all(vapply(xs, function(x) identical(names(x$blocks), block_names), logical(1)))) {
      .frame_abort("Bound axes must have identical block names.", "fmridataset_error_alignment")
    }
    blocks <- lapply(block_names, function(nm) {
      values <- lapply(xs, function(x) axis_block_data(x$blocks[[nm]]))
      if (any(vapply(values, inherits, logical(1), what = "array_source"))) {
        source <- row_bound_source(lapply(values, as_array_source))
      } else {
        source <- do.call(rbind, values)
      }
      proto <- first$blocks[[nm]]
      axis_block(source, proto$components, proto$role, proto$units, proto$metadata)
    })
    names(blocks) <- block_names
  } else {
    blocks <- list()
  }
  data <- do.call(rbind, lapply(xs, axis_data))
  out <- axis_frame(
    data, blocks = blocks, id = data[[first$id_col]], axis = first$axis,
    id_col = first$id_col, metadata = first$metadata
  )
  policies <- lapply(xs, axis_id_policy)
  out$id_policy <- if (all(vapply(
    policies, identical, logical(1), policies[[1L]]
  ))) policies[[1L]] else .id_policy("require")
  out
}

.frame_container_value <- function(x, name) x[[name]] %||% x$base[[name]]

.merge_unaligned_records <- function(values, path = "metadata") {
  out <- unclass(values[[1L]])
  if (length(values) == 1L) {
    return(structure(out, class = c("unaligned_record", "list")))
  }
  for (value in values[-1L]) {
    value <- unclass(value)
    for (name in names(value)) {
      child_path <- paste(path, name, sep = ".")
      if (!name %in% names(out)) {
        out[[name]] <- value[[name]]
      } else if (inherits(out[[name]], "unaligned_record") &&
                 inherits(value[[name]], "unaligned_record")) {
        out[[name]] <- .merge_unaligned_records(
          list(out[[name]], value[[name]]), child_path
        )
      } else if (!identical(out[[name]], value[[name]])) {
        .frame_abort(
          sprintf("Cannot merge conflicting frame metadata at '%s'.", child_path),
          "fmridataset_error_alignment", field = child_path
        )
      }
    }
  }
  structure(out, class = c("unaligned_record", "list"))
}

.reconcile_frame_metadata <- function(xs, policy) {
  values <- lapply(xs, .frame_container_value, name = "metadata")
  if (identical(policy, "identical")) {
    if (!all(vapply(values, identical, logical(1), values[[1L]]))) {
      .frame_abort(
        "Bound frame metadata differ; use metadata_policy = 'merge' for a conflict-free record merge.",
        "fmridataset_error_alignment", field = "metadata"
      )
    }
    return(values[[1L]])
  }
  .merge_unaligned_records(values)
}

.table_row_equal <- function(x, i, y, j) {
  all(vapply(names(x), function(name) {
    identical(x[[name]][i], y[[name]][j])
  }, logical(1)))
}

.merge_typed_table <- function(values, name) {
  prototype <- values[[1L]]
  keys <- vapply(values, function(value) table_key(value) %||% NA_character_,
                 character(1), USE.NAMES = FALSE)
  if (anyNA(keys) || any(!nzchar(keys))) {
    if (!all(vapply(values, identical, logical(1), prototype))) {
      .table_abort(
        sprintf(
          "Bound table '%s' has no declared key and differs across operands.",
          name
        ),
        paste0("tables.", name)
      )
    }
    return(prototype)
  }
  if (any(keys != keys[[1L]])) {
    .table_abort(
      sprintf("Bound table '%s' declares inconsistent keys.", name),
      paste0("tables.", name, ".key")
    )
  }
  rows <- table_data(prototype)
  key <- keys[[1L]]
  if (length(values) > 1L) {
    for (value in values[-1L]) {
      incoming <- table_data(value)
      positions <- match(as.character(incoming[[key]]), as.character(rows[[key]]))
      overlap <- which(!is.na(positions))
      if (length(overlap)) {
        equal <- vapply(overlap, function(i) {
          .table_row_equal(incoming, i, rows, positions[[i]])
        }, logical(1))
        if (!all(equal)) {
          .table_abort(
            sprintf("Bound table '%s' contains conflicting rows for declared keys.", name),
            paste0("tables.", name),
            keys = as.character(incoming[[key]][overlap[!equal]])
          )
        }
      }
      append <- which(is.na(positions))
      if (length(append)) rows <- rbind(rows, incoming[append, , drop = FALSE])
    }
  }
  if (inherits(prototype, "fmri_event_table")) {
    event_table(rows, key = key, metadata = prototype$metadata)
  } else {
    auxiliary_table(
      rows, key = key, role = table_role(prototype),
      metadata = prototype$metadata
    )
  }
}

.merge_frame_tables <- function(xs) {
  registries <- lapply(xs, .frame_container_value, name = "tables")
  first_names <- names(registries[[1L]])
  if (!all(vapply(registries, function(value) {
    identical(names(value), first_names)
  }, logical(1)))) {
    .table_abort(
      "Bound frames must have identical typed-table names.",
      "tables"
    )
  }
  out <- lapply(first_names, function(name) {
    .merge_typed_table(lapply(registries, `[[`, name), name)
  })
  names(out) <- first_names
  out
}

.merge_bind_provenance <- function(xs) {
  graphs <- lapply(xs, .frame_container_value, name = "provenance")
  records <- list()
  parents <- character()
  for (graph in graphs) {
    if (is.null(graph)) next
    current <- provenance_records(graph)
    for (id in names(current)) {
      if (!is.null(records[[id]]) && !identical(records[[id]], current[[id]])) {
        .provenance_abort("A provenance record ID has conflicting content.")
      }
      records[[id]] <- current[[id]]
    }
    parents <- c(parents, provenance_tips(graph))
  }
  bind <- provenance_record(
    "bind_observations",
    parents = unique(parents),
    inputs = list(observation_ids = lapply(xs, observation_ids)),
    parameters = list(operand_count = length(xs)),
    outputs = list(
      observation_ids = unlist(lapply(xs, observation_ids), use.names = FALSE),
      feature_ids = feature_ids(xs[[1L]])
    )
  )
  provenance_graph(c(records, list(bind)))
}

.flatten_bound_sources <- function(sources) {
  unlist(lapply(sources, function(source) {
    if (inherits(source, "row_bound_source")) {
      .flatten_bound_sources(source$sources)
    } else {
      list(source)
    }
  }), recursive = FALSE)
}

.bind_assay_sources <- function(sources) {
  flattened <- .flatten_bound_sources(sources)
  nonempty <- vapply(
    flattened, function(source) source_shape(source)[[1L]] > 0L, logical(1)
  )
  if (!any(nonempty)) {
    return(source_view(flattened[[1L]], observations = integer()))
  }
  flattened <- flattened[nonempty]
  if (length(flattened) == 1L) flattened[[1L]] else row_bound_source(flattened)
}

#' Bind frames along observations
#'
#' @param ... Frames with identical feature IDs, spaces, and assay semantics.
#' @param metadata_policy Frame-metadata reconciliation. `"identical"`
#'   requires exact equality; `"merge"` recursively combines non-conflicting
#'   unaligned records.
#' @param active_assay Optional active assay for the result. Required when
#'   operands have different active assays.
#' @return A lazily row-bound `fmri_frame`.
#' @export
bind_observations <- function(...,
                              metadata_policy = c("identical", "merge"),
                              active_assay = NULL) {
  xs <- list(...)
  if (!length(xs)) .frame_abort("At least one frame is required.", "fmridataset_error_alignment")
  if (!all(vapply(xs, inherits, logical(1), "fmri_frame"))) {
    .frame_abort("Every bound operand must be an fmri_frame or view.",
                 "fmridataset_error_alignment")
  }
  metadata_policy <- match.arg(metadata_policy)
  first <- xs[[1L]]
  for (x in xs[-1L]) {
    assert_compatible_space(space(first), space(x))
    if (!identical(entity_registry_digest(first), entity_registry_digest(x))) {
      .entity_abort(
        "Frames have incompatible entity registries.",
        operation = "bind_observations"
      )
    }
    validate_against_schema(x, first, mode = "bind")
  }
  relation_values <- .bind_relation_registries(lapply(xs, relations))
  obs <- .bind_axis_frames(lapply(xs, observation_axis))
  if (anyDuplicated(axis_ids(obs))) {
    .frame_abort("Observation IDs collide across frames.", "fmridataset_error_alignment")
  }
  active_values <- vapply(xs, function(value) active_assay(value), character(1))
  if (is.null(active_assay)) {
    if (any(active_values != active_values[[1L]])) {
      .frame_abort(
        "Bound frames have different active assays; supply active_assay explicitly.",
        "fmridataset_error_alignment", field = "active_assay"
      )
    }
    active_assay <- active_values[[1L]]
  } else if (!is.character(active_assay) || length(active_assay) != 1L ||
             is.na(active_assay) || !active_assay %in% names(assays(first))) {
    .frame_abort(
      "active_assay must name one assay shared by all bound frames.",
      "fmridataset_error_alignment", field = "active_assay"
    )
  }
  metadata <- .reconcile_frame_metadata(xs, metadata_policy)
  tables <- .merge_frame_tables(xs)
  provenance <- .merge_bind_provenance(xs)
  assay_sources <- lapply(names(assays(first)), function(nm) {
    prototype <- assay(first, nm)
    structure(
      list(
        source = .bind_assay_sources(lapply(
          xs, function(x) .frame_assay_source(x, nm)
        )),
        role = prototype$role,
        units = prototype$units,
        metadata = prototype$metadata
      ),
      class = "aligned_assay"
    )
  })
  names(assay_sources) <- names(assays(first))
  fmri_frame(
    assays = assay_sources,
    observations = obs,
    features = feature_axis(first),
    entities = entities(first),
    relations = relation_values,
    tables = tables,
    active_assay = active_assay,
    metadata = metadata,
    provenance = provenance
  )
}
