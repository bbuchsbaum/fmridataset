.study_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_study", ...)
}

.event_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_event", ...)
}

#' Construct a keyed event table
#'
#' Event rows retain their natural cardinality and are not expanded to acquired
#' volumes. Entity-key columns are validated against a study when attached.
#'
#' @param data Scalar event annotations.
#' @param key Stable event-key column.
#' @param metadata Serializable event-table metadata.
#' @return An `fmri_event_table`.
#' @export
event_table <- function(data, key = "event_id", metadata = list()) {
  if (!is.data.frame(data)) .event_abort("Event data must be a data frame.")
  data <- tibble::as_tibble(data)
  if (!is.character(key) || length(key) != 1L || is.na(key) || !nzchar(key) ||
      !key %in% names(data)) {
    .event_abort("Event key must name one scalar data column.", field = "key")
  }
  non_scalar <- vapply(data, function(value) {
    is.list(value) || !is.null(dim(value)) || length(value) != nrow(data)
  }, logical(1))
  if (any(non_scalar)) {
    .event_abort("Event columns must contain scalar values.", columns = names(data)[non_scalar])
  }
  ids <- as.character(data[[key]])
  if (anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
    .event_abort("Event keys must be unique, non-missing, and non-empty.", field = key)
  }
  data[[key]] <- ids
  if ("onset" %in% names(data) &&
      (!is.numeric(data$onset) || anyNA(data$onset) ||
       any(!is.finite(data$onset)) || any(data$onset < 0))) {
    .event_abort("Event onset values must be finite and non-negative.", field = "onset")
  }
  if ("duration" %in% names(data) &&
      (!is.numeric(data$duration) || anyNA(data$duration) ||
       any(!is.finite(data$duration)) || any(data$duration < 0))) {
    .event_abort("Event duration values must be finite and non-negative.", field = "duration")
  }
  out <- structure(
    list(data = data, key = key, metadata = metadata, schema_version = 1L),
    class = "fmri_event_table"
  )
  if (.source_contains_runtime_state(out)) {
    .event_abort("Event tables cannot contain runtime state.")
  }
  out
}

#' Validate a keyed event table
#'
#' @param x An `fmri_event_table`.
#' @return `x`, invisibly.
#' @export
validate_event_table <- function(x) {
  required <- c("data", "key", "metadata", "schema_version")
  if (!inherits(x, "fmri_event_table") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .event_abort("x is not a valid fmri_event_table.")
  }
  event_table(x$data, key = x$key, metadata = x$metadata)
  invisible(x)
}

#' Event-table accessors
#'
#' @param x An `fmri_event_table`.
#' @return Event scalar data or the stable key name.
#' @name event-accessors
NULL

#' @rdname event-accessors
#' @export
event_data <- function(x) {
  validate_event_table(x)
  x$data
}

#' @rdname event-accessors
#' @export
event_key <- function(x) {
  validate_event_table(x)
  x$key
}

#' Describe a typed link between study representations
#'
#' @param from Source representation name.
#' @param to Target representation name.
#' @param type Link type: derivation, feature mapping, correspondence, or
#'   alignment.
#' @param map Optional scalar table with `.from_id` and `.to_id` columns.
#' @param from_axis Axis addressed by `.from_id`.
#' @param to_axis Axis addressed by `.to_id`.
#' @param metadata Serializable link metadata.
#' @return A `frame_link` descriptor.
#' @export
frame_link <- function(from, to,
                       type = c("derived_from", "mapped_from", "corresponds_to", "aligned_from"),
                       map = NULL,
                       from_axis = c("observation", "feature"),
                       to_axis = c("observation", "feature"), metadata = list()) {
  scalar_string <- function(value, field) {
    if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
      .study_abort(sprintf("%s must be one non-empty string.", field), field = field)
    }
    value
  }
  from <- scalar_string(from, "from")
  to <- scalar_string(to, "to")
  allowed <- c("derived_from", "mapped_from", "corresponds_to", "aligned_from")
  type <- if (length(type)) type[[1L]] else type
  type <- scalar_string(type, "type")
  if (!type %in% allowed) {
    .study_abort("Unknown frame-link type.", type = type, allowed = allowed)
  }
  from_axis <- match.arg(from_axis)
  to_axis <- match.arg(to_axis)
  if (!is.null(map)) {
    if (!is.data.frame(map)) .study_abort("Link map must be a data frame.", field = "map")
    map <- tibble::as_tibble(map)
    non_scalar <- vapply(map, function(value) {
      is.list(value) || !is.null(dim(value)) || length(value) != nrow(map)
    }, logical(1))
    if (any(non_scalar)) {
      .study_abort("Link map columns must contain scalar values.", field = "map")
    }
    required <- c(".from_id", ".to_id")
    if (!all(required %in% names(map))) {
      .study_abort("Link maps require .from_id and .to_id columns.", field = "map")
    }
    map$.from_id <- as.character(map$.from_id)
    map$.to_id <- as.character(map$.to_id)
    if (anyNA(map$.from_id) || any(!nzchar(map$.from_id)) ||
        anyNA(map$.to_id) || any(!nzchar(map$.to_id)) ||
        anyDuplicated(map[required])) {
      .study_abort("Link map IDs must be non-missing with unique pairs.", field = "map")
    }
  }
  out <- structure(
    list(
      from = from, to = to, type = type, map = map,
      from_axis = from_axis, to_axis = to_axis,
      metadata = metadata, schema_version = 1L
    ),
    class = "frame_link"
  )
  if (.source_contains_runtime_state(out)) .study_abort("Frame links cannot contain runtime state.")
  out
}

.validate_frame_link <- function(x, name = NULL) {
  required <- c(
    "from", "to", "type", "map", "from_axis", "to_axis", "metadata",
    "schema_version"
  )
  if (!inherits(x, "frame_link") || !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .study_abort("Invalid frame_link descriptor.", link = name)
  }
  frame_link(
    x$from, x$to, x$type, map = x$map,
    from_axis = x$from_axis, to_axis = x$to_axis, metadata = x$metadata
  )
  invisible(x)
}

.study_member_frames <- function(value) {
  if (inherits(value, "fmri_collection")) collection_frames(value) else list(value)
}

.study_vector_equal <- function(x, y) {
  isTRUE(all.equal(x, y, check.attributes = FALSE))
}

.validate_member_entities <- function(frame, shared, representation) {
  local <- entities(frame)
  missing_entities <- setdiff(entity_names(local), entity_names(shared))
  if (length(missing_entities)) {
    .study_abort(
      sprintf("Representation '%s' references an entity absent from the study.", representation),
      representation = representation,
      entities = missing_entities
    )
  }
  for (entity_name in entity_names(local)) {
    local_value <- local[[entity_name]]
    shared_value <- shared[[entity_name]]
    positions <- match(entity_ids(local_value), entity_ids(shared_value))
    if (anyNA(positions)) {
      .study_abort(
        sprintf("Representation '%s' contains unknown %s entity IDs.", representation, entity_name),
        representation = representation,
        entity = entity_name
      )
    }
    local_data <- entity_data(local_value)
    shared_data <- entity_data(shared_value)
    missing_columns <- setdiff(names(local_data), names(shared_data))
    if (length(missing_columns)) {
      .study_abort(
        sprintf("Representation '%s' has entity fields absent from shared entities.", representation),
        entity = entity_name,
        columns = missing_columns
      )
    }
    for (column in names(local_data)) {
      if (!.study_vector_equal(local_data[[column]], shared_data[[column]][positions])) {
        .study_abort(
          sprintf("Representation '%s' entity '%s' disagrees with shared metadata.", representation, entity_name),
          representation = representation,
          entity = entity_name,
          column = column
        )
      }
    }
    local_blocks <- entity_blocks(local_value)
    shared_blocks <- entity_blocks(shared_value)
    if (!all(names(local_blocks) %in% names(shared_blocks))) {
      .study_abort(
        sprintf("Representation '%s' has entity blocks absent from shared entities.", representation),
        entity = entity_name
      )
    }
    for (block_name in names(local_blocks)) {
      x <- local_blocks[[block_name]]
      y <- shared_blocks[[block_name]]
      if (!identical(block_components(x), block_components(y)) ||
          !identical(x$role, y$role) || !identical(x$units, y$units)) {
        .study_abort(
          sprintf("Representation '%s' entity block '%s.%s' is incompatible.", representation, entity_name, block_name),
          entity = entity_name,
          block = block_name
        )
      }
    }
  }
  invisible(TRUE)
}

.study_axis_ids <- function(value, axis, endpoint) {
  if (inherits(value, "fmri_collection")) {
    .study_abort(
      sprintf("Mapped link endpoint '%s' is a collection and has no single %s axis.", endpoint, axis),
      endpoint = endpoint
    )
  }
  if (identical(axis, "observation")) observation_ids(value) else feature_ids(value)
}

.validate_study_links <- function(links, frames) {
  if (!is.list(links)) .study_abort("links must be a named list.", field = "links")
  if (length(links)) {
    ids <- names(links)
    if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
      .study_abort("Study links require unique non-empty names.", field = "links")
    }
    for (id in ids) {
      value <- links[[id]]
      .validate_frame_link(value, id)
      if (!all(c(value$from, value$to) %in% names(frames))) {
        .study_abort(sprintf("Study link '%s' has an unknown endpoint.", id), link = id)
      }
      if (!is.null(value$map)) {
        from_ids <- .study_axis_ids(frames[[value$from]], value$from_axis, value$from)
        to_ids <- .study_axis_ids(frames[[value$to]], value$to_axis, value$to)
        if (any(!value$map$.from_id %in% from_ids) || any(!value$map$.to_id %in% to_ids)) {
          .study_abort(sprintf("Study link '%s' map contains unknown axis IDs.", id), link = id)
        }
      }
    }
  }
  links
}

.validate_study_tables <- function(tables, entities_value) {
  if (!is.list(tables)) .study_abort("tables must be a named list.", field = "tables")
  if (length(tables)) {
    ids <- names(tables)
    if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
      .study_abort("Study tables require unique non-empty names.", field = "tables")
    }
    valid <- vapply(tables, function(value) {
      inherits(value, "fmri_event_table") || is.data.frame(value)
    }, logical(1))
    if (!all(valid)) .study_abort("Study tables must be data frames or event tables.")
    for (table_name in ids[vapply(tables, inherits, logical(1), "fmri_event_table")]) {
      validate_event_table(tables[[table_name]])
      data <- event_data(tables[[table_name]])
      for (entity_name in entity_names(entities_value)) {
        entity_value <- entities_value[[entity_name]]
        key <- entity_key(entity_value)
        if (!key %in% names(data)) next
        values <- as.character(data[[key]])
        present <- !is.na(values)
        if (any(!values[present] %in% entity_ids(entity_value))) {
          .study_abort(
            sprintf("Event table '%s' contains unknown %s entity IDs.", table_name, entity_name),
            table = table_name,
            entity = entity_name
          )
        }
      }
    }
  }
  if (.source_contains_runtime_state(tables)) .study_abort("Study tables cannot contain runtime state.")
  tables
}

#' Construct a linked fMRI study
#'
#' @param frames Named `fmri_frame` or `fmri_collection` representations.
#' @param entities Shared authoritative entity registry.
#' @param links Named `frame_link` descriptors.
#' @param tables Named relational tables, including `event_table` objects.
#' @param metadata Serializable study metadata.
#' @param provenance Serializable provenance records.
#' @return An `fmri_study`.
#' @export
fmri_study <- function(frames, entities = list(), links = list(), tables = list(),
                       metadata = list(), provenance = NULL) {
  if (!is.list(frames) || !length(frames)) {
    .study_abort("frames must be a non-empty named list.", field = "frames")
  }
  ids <- names(frames)
  if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
    .study_abort("Study representation names must be unique stable IDs.", field = "frames")
  }
  valid <- vapply(frames, function(value) {
    inherits(value, "fmri_frame") || inherits(value, "fmri_collection")
  }, logical(1))
  if (!all(valid)) .study_abort("Study representations must be frames or collections.")
  entities_value <- entity_registry(entities)
  for (representation in ids) {
    members <- .study_member_frames(frames[[representation]])
    for (member in members) {
      .validate_member_entities(member, entities_value, representation)
    }
  }
  links <- .validate_study_links(links, frames)
  tables <- .validate_study_tables(tables, entities_value)
  out <- structure(
    list(
      frames = frames, entities = entities_value, links = links, tables = tables,
      metadata = metadata, provenance = provenance, schema_version = 1L
    ),
    class = "fmri_study"
  )
  if (.source_contains_runtime_state(out)) .study_abort("Studies cannot contain runtime state.")
  out
}

.contextualize_study_frame <- function(value, shared) {
  if (inherits(value, "fmri_collection")) {
    members <- lapply(collection_frames(value), .contextualize_study_frame, shared = shared)
    return(fmri_collection(members, metadata = value$metadata, provenance = value$provenance))
  }
  assay_sources <- lapply(names(assays(value)), function(name) .frame_assay_source(value, name))
  names(assay_sources) <- names(assays(value))
  fmri_frame(
    assays = assay_sources,
    observations = observation_axis(value),
    features = feature_axis(value),
    entities = shared,
    relations = relations(value),
    tables = value$tables %||% value$base$tables,
    active_assay = active_assay(value),
    metadata = value$metadata %||% value$base$metadata,
    provenance = value$provenance %||% value$base$provenance
  )
}

.study_base <- function(x) if (inherits(x, "fmri_study_view")) x$base else x
.study_raw_frames <- function(x) if (inherits(x, "fmri_study_view")) x$frames else x$frames

#' Validate a study or filtered study view
#'
#' @param x An `fmri_study` or `fmri_study_view`.
#' @return `x`, invisibly.
#' @export
validate_fmri_study <- function(x) {
  if (inherits(x, "fmri_study_view")) {
    required <- c("base", "frames", "entity_selections", "schema_version")
    if (!identical(names(unclass(x)), required) || !identical(x$schema_version, 1L)) {
      .study_abort("x is not a valid fmri_study_view.")
    }
    validate_fmri_study(x$base)
    if (!is.list(x$frames) || !identical(names(x$frames), names(x$base$frames)) ||
        !is.list(x$entity_selections)) {
      .study_abort("x is not a valid fmri_study_view.")
    }
    valid_frames <- vapply(x$frames, function(value) {
      inherits(value, "fmri_frame") || inherits(value, "fmri_collection")
    }, logical(1))
    selection_names <- names(x$entity_selections)
    valid_selection_names <- !is.null(selection_names) &&
      !anyNA(selection_names) && !any(!nzchar(selection_names)) &&
      !anyDuplicated(selection_names) &&
      all(selection_names %in% entity_names(x$base$entities))
    valid_selections <- valid_selection_names && all(vapply(
      selection_names,
      function(name) {
        ids <- x$entity_selections[[name]]
        is.character(ids) && !anyNA(ids) && !any(!nzchar(ids)) &&
          !anyDuplicated(ids) && all(ids %in% entity_ids(x$base$entities[[name]]))
      },
      logical(1)
    ))
    if (!all(valid_frames) || !valid_selections || .source_contains_runtime_state(x)) {
      .study_abort("x is not a valid fmri_study_view.")
    }
    return(invisible(x))
  }
  required <- c("frames", "entities", "links", "tables", "metadata", "provenance", "schema_version")
  if (!inherits(x, "fmri_study") || !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .study_abort("x is not a valid fmri_study.")
  }
  fmri_study(x$frames, x$entities, x$links, x$tables, x$metadata, x$provenance)
  invisible(x)
}

#' Study representation accessors
#'
#' @param x An `fmri_study` or filtered view.
#' @param name Stable representation name.
#' @param contextual Replace frame-local entity stubs with shared study entities.
#' @return Named representations, one representation, or representation IDs.
#' @name study-accessors
NULL

#' @rdname study-accessors
#' @export
study_frames <- function(x, contextual = TRUE) {
  validate_fmri_study(x)
  if (!is.logical(contextual) || length(contextual) != 1L || is.na(contextual)) {
    .study_abort("contextual must be TRUE or FALSE.", field = "contextual")
  }
  frames <- .study_raw_frames(x)
  if (!isTRUE(contextual)) return(frames)
  shared <- entities(.study_base(x))
  lapply(frames, .contextualize_study_frame, shared = shared)
}

#' @rdname study-accessors
#' @export
study_frame <- function(x, name, contextual = TRUE) {
  frames <- study_frames(x, contextual = contextual)
  if (!is.character(name) || length(name) != 1L || is.na(name) || !name %in% names(frames)) {
    label <- if (length(name)) as.character(name)[[1L]] else ""
    .study_abort(sprintf("Unknown study representation '%s'.", label), representation = name)
  }
  frames[[name]]
}

#' @rdname study-accessors
#' @export
study_ids <- function(x) names(study_frames(x, contextual = FALSE))

#' @export
entities.fmri_study <- function(x, ...) x$entities
#' @export
entity.fmri_study <- function(x, name, ...) entity(entities(x), name)
#' @export
entities.fmri_study_view <- function(x, ...) {
  registry <- entities(x$base)
  for (name in names(x$entity_selections)) {
    value <- registry[[name]]
    registry[[name]] <- value[match(x$entity_selections[[name]], entity_ids(value))]
  }
  class(registry) <- c("entity_registry", "list")
  registry
}
#' @export
entity.fmri_study_view <- function(x, name, ...) entity(entities(x), name)

#' Study link and table accessors
#'
#' @param x An `fmri_study` or filtered view.
#' @param name Stable link or table name.
#' @return A registry or one descriptor/table.
#' @name study-registries
NULL

#' @rdname study-registries
#' @export
study_links <- function(x) {
  validate_fmri_study(x)
  values <- .study_base(x)$links
  if (!inherits(x, "fmri_study_view")) return(values)
  frames <- .study_raw_frames(x)
  lapply(values, function(value) {
    if (is.null(value$map)) return(value)
    from_ids <- .study_axis_ids(frames[[value$from]], value$from_axis, value$from)
    to_ids <- .study_axis_ids(frames[[value$to]], value$to_axis, value$to)
    keep <- value$map$.from_id %in% from_ids & value$map$.to_id %in% to_ids
    value$map <- value$map[keep, , drop = FALSE]
    value
  })
}
#' @rdname study-registries
#' @export
study_link <- function(x, name) {
  values <- study_links(x)
  if (!is.character(name) || length(name) != 1L || is.na(name) || !name %in% names(values)) {
    .study_abort("Unknown study link.", link = name)
  }
  values[[name]]
}

.filter_event_table <- function(value, selections, shared) {
  data <- event_data(value)
  keep <- rep(TRUE, nrow(data))
  for (entity_name in names(selections)) {
    key <- entity_key(shared[[entity_name]])
    if (key %in% names(data)) keep <- keep & !is.na(data[[key]]) & data[[key]] %in% selections[[entity_name]]
  }
  event_table(data[keep, , drop = FALSE], key = event_key(value), metadata = value$metadata)
}

#' @rdname study-registries
#' @export
study_tables <- function(x) {
  values <- .study_base(x)$tables
  if (!inherits(x, "fmri_study_view")) return(values)
  shared <- entities(x$base)
  lapply(values, function(value) {
    if (inherits(value, "fmri_event_table")) {
      .filter_event_table(value, x$entity_selections, shared)
    } else value
  })
}
#' @rdname study-registries
#' @export
study_table <- function(x, name) {
  values <- study_tables(x)
  if (!is.character(name) || length(name) != 1L || is.na(name) || !name %in% names(values)) {
    .study_abort("Unknown study table.", table = name)
  }
  values[[name]]
}
#' @rdname study-registries
#' @export
events <- function(x, name = "events") study_table(x, name)

.filter_study_frame_entity <- function(value, entity_name, selected_ids, shared) {
  if (inherits(value, "fmri_collection")) {
    members <- lapply(collection_frames(value), .filter_study_frame_entity,
                      entity_name = entity_name, selected_ids = selected_ids,
                      shared = shared)
    return(fmri_collection(members, metadata = value$metadata, provenance = value$provenance))
  }
  contextual <- .contextualize_study_frame(value, shared)
  maps <- .resolve_entity_maps(contextual)
  mapped <- maps[[paste0("entity:", entity_name)]]
  if (is.null(mapped)) return(value)
  contextual[which(!is.na(mapped) & mapped %in% selected_ids), ]
}

#' Filter every study representation through one shared entity selection
#'
#' @param x An `fmri_study` or filtered view.
#' @param entity Bare or quoted shared entity name.
#' @param predicate A scalar-metadata predicate evaluated on that entity table.
#' @return A lazy `fmri_study_view`.
#' @export
filter_entities <- function(x, entity, predicate) {
  validate_fmri_study(x)
  entity_name <- rlang::as_name(rlang::ensym(entity))
  visible_registry <- entities(x)
  if (!entity_name %in% entity_names(visible_registry)) {
    .study_abort(sprintf("Unknown study entity '%s'.", entity_name), entity = entity_name)
  }
  visible <- visible_registry[[entity_name]]
  keep <- rlang::eval_tidy(rlang::enquo(predicate), data = entity_data(visible))
  if (!is.logical(keep) || length(keep) != length(visible) || anyNA(keep)) {
    .study_abort("Entity predicate must return one non-missing logical value per entity.")
  }
  selected_ids <- entity_ids(visible)[which(keep)]
  base <- .study_base(x)
  shared <- entities(base)
  frames <- lapply(
    .study_raw_frames(x),
    .filter_study_frame_entity,
    entity_name = entity_name,
    selected_ids = selected_ids,
    shared = shared
  )
  selections <- if (inherits(x, "fmri_study_view")) x$entity_selections else list()
  selections[[entity_name]] <- selected_ids
  out <- structure(
    list(base = base, frames = frames, entity_selections = selections, schema_version = 1L),
    class = c("fmri_study_view", "fmri_study")
  )
  if (.source_contains_runtime_state(out)) .study_abort("Study views cannot contain runtime state.")
  out
}

.study_representation_digest <- function(value) {
  if (inherits(value, "fmri_collection")) collection_digest(value) else .canonical_digest(.collection_frame_descriptor(value))
}

#' Compute a deterministic study digest
#'
#' @param x An `fmri_study` or filtered view.
#' @return A SHA-256 digest computed without numerical reads.
#' @export
study_digest <- function(x) {
  validate_fmri_study(x)
  base <- .study_base(x)
  .canonical_digest(list(
    schema_version = 1L,
    frames = lapply(.study_raw_frames(x), .study_representation_digest),
    entities = entities(x),
    links = study_links(x),
    tables = study_tables(x),
    metadata = base$metadata,
    provenance = base$provenance
  ))
}

#' @export
print.fmri_study <- function(x, ...) {
  validate_fmri_study(x)
  label <- if (inherits(x, "fmri_study_view")) " filtered view" else ""
  cat("<fmri_study>", length(study_ids(x)), "representations", label, "\n")
  cat("  entities:", paste(entity_names(x), collapse = ", "), "\n")
  cat("  links:", length(study_links(x)), "\n")
  invisible(x)
}
