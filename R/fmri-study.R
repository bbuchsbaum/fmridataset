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
  data <- tryCatch(
    .validate_scalar_table_data(data, "Event table"),
    fmridataset_error_table = function(error) {
      .event_abort(conditionMessage(error), field = error$field %||% "data")
    }
  )
  if (!is.character(key) || length(key) != 1L || is.na(key) || !nzchar(key) ||
      !key %in% names(data)) {
    .event_abort("Event key must name one scalar data column.", field = "key")
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
  metadata <- unaligned_record(metadata)
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
#' @param source Source representation name.
#' @param target Target representation name.
#' @param type Link type: derivation, feature mapping, correspondence, or
#'   alignment.
#' @param map Optional scalar table with `.source_id` and `.target_id` columns.
#' @param source_axis Axis addressed by `.source_id`.
#' @param target_axis Axis addressed by `.target_id`.
#' @param metadata Serializable link metadata.
#' @param operator Optional typed feature operator. This is valid only for a
#'   feature-to-feature mapping or alignment and remains a first-class field.
#' @return A `frame_link` descriptor.
#' @export
frame_link <- function(source, target,
                       type = c("derivation", "mapping", "correspondence", "alignment"),
                       map = NULL,
                       source_axis = c("observation", "feature"),
                       target_axis = c("observation", "feature"),
                       metadata = list(), operator = NULL) {
  scalar_string <- function(value, field) {
    if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
      .study_abort(sprintf("%s must be one non-empty string.", field), field = field)
    }
    value
  }
  source <- scalar_string(source, "source")
  target <- scalar_string(target, "target")
  allowed <- c("derivation", "mapping", "correspondence", "alignment")
  type <- if (length(type)) type[[1L]] else type
  type <- scalar_string(type, "type")
  if (!type %in% allowed) {
    .study_abort("Unknown frame-link type.", type = type, allowed = allowed)
  }
  source_axis <- match.arg(source_axis)
  target_axis <- match.arg(target_axis)
  metadata <- tryCatch(
    unaligned_record(metadata),
    error = function(error) {
      .study_abort("Frame-link metadata must be a serializable unaligned record.",
                   field = "metadata")
    }
  )
  if (type %in% c("mapping", "alignment") &&
      (!identical(source_axis, "feature") ||
       !identical(target_axis, "feature"))) {
    .study_abort(
      "Mapping and alignment links must address feature axes.",
      field = "type"
    )
  }
  if (!is.null(operator)) {
    validate_feature_map(operator)
    if (!type %in% c("mapping", "alignment") ||
        !identical(source_axis, "feature") ||
        !identical(target_axis, "feature")) {
      .study_abort(
        "operator is valid only for a feature-to-feature mapping or alignment link.",
        field = "operator"
      )
    }
  }
  if (!is.null(map)) {
    if (!is.data.frame(map)) .study_abort("Link map must be a data frame.", field = "map")
    map <- tibble::as_tibble(map)
    non_scalar <- vapply(map, function(value) {
      is.list(value) || !is.null(dim(value)) || length(value) != nrow(map)
    }, logical(1))
    if (any(non_scalar)) {
      .study_abort("Link map columns must contain scalar values.", field = "map")
    }
    required <- c(".source_id", ".target_id")
    if (!all(required %in% names(map))) {
      .study_abort("Link maps require .source_id and .target_id columns.", field = "map")
    }
    map$.source_id <- as.character(map$.source_id)
    map$.target_id <- as.character(map$.target_id)
    if (anyNA(map$.source_id) || any(!nzchar(map$.source_id)) ||
        anyNA(map$.target_id) || any(!nzchar(map$.target_id)) ||
        anyDuplicated(map[required])) {
      .study_abort("Link map IDs must be non-missing with unique pairs.", field = "map")
    }
  }
  out <- structure(
    list(
      source = source, target = target, type = type, map = map,
      source_axis = source_axis, target_axis = target_axis,
      operator = operator, metadata = metadata, schema_version = 2L
    ),
    class = "frame_link"
  )
  if (.source_contains_runtime_state(out)) .study_abort("Frame links cannot contain runtime state.")
  out
}

.validate_frame_link <- function(x, name = NULL) {
  required <- c(
    "source", "target", "type", "map", "source_axis", "target_axis",
    "operator", "metadata", "schema_version"
  )
  if (!inherits(x, "frame_link") || !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 2L)) {
    .study_abort("Invalid frame_link descriptor.", link = name)
  }
  frame_link(
    x$source, x$target, x$type, map = x$map,
    source_axis = x$source_axis, target_axis = x$target_axis,
    metadata = x$metadata, operator = x$operator
  )
  invisible(x)
}

#' Upgrade a provisional frame-link descriptor
#'
#' Version-one links used reverse `*_from` endpoints for derivation, mapping,
#' and alignment and stored feature operators in metadata. This explicit
#' migration converts them to the canonical source-to-target version-two form.
#'
#' @param x A provisional version-one or canonical version-two `frame_link`.
#' @return A canonical version-two `frame_link`.
#' @export
upgrade_frame_link <- function(x) {
  if (inherits(x, "frame_link") && identical(x$schema_version, 2L)) {
    .validate_frame_link(x)
    return(x)
  }
  required <- c(
    "from", "to", "type", "map", "from_axis", "to_axis", "metadata",
    "schema_version"
  )
  if (!inherits(x, "frame_link") || !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .study_abort("x is not a supported provisional frame_link.", field = "x")
  }
  legacy_types <- c(
    derived_from = "derivation", mapped_from = "mapping",
    corresponds_to = "correspondence", aligned_from = "alignment"
  )
  if (!x$type %in% names(legacy_types)) {
    .study_abort("Unknown provisional frame-link type.", type = x$type)
  }
  reverse <- !identical(x$type, "corresponds_to")
  source <- if (reverse) x$to else x$from
  target <- if (reverse) x$from else x$to
  source_axis <- if (reverse) x$to_axis else x$from_axis
  target_axis <- if (reverse) x$from_axis else x$to_axis
  map <- x$map
  if (!is.null(map)) {
    if (!all(c(".from_id", ".to_id") %in% names(map))) {
      .study_abort("Provisional link map lacks .from_id and .to_id.", field = "map")
    }
    source_id <- if (reverse) map$.to_id else map$.from_id
    target_id <- if (reverse) map$.from_id else map$.to_id
    extra <- map[setdiff(names(map), c(".from_id", ".to_id"))]
    map <- tibble::as_tibble(c(
      list(.source_id = source_id, .target_id = target_id), extra
    ))
  }
  metadata <- x$metadata
  operator <- metadata$feature_map %||% NULL
  metadata$feature_map <- NULL
  frame_link(
    source, target, unname(legacy_types[[x$type]]), map = map,
    source_axis = source_axis, target_axis = target_axis,
    metadata = metadata, operator = operator
  )
}

#' Compose source-to-target frame links
#'
#' The target of `first` must be the source of `second`, and their addressed
#' axes must agree. Explicit ID maps are joined through the intermediate IDs.
#' Feature operators are composed as `second %*% first` without changing link
#' direction.
#'
#' @param first A canonical source-to-intermediate `frame_link`.
#' @param second A canonical intermediate-to-target `frame_link`.
#' @param type Result link type. It may be omitted when both inputs have the
#'   same type.
#' @param metadata Unaligned result-link metadata.
#' @return A canonical source-to-target `frame_link`.
#' @export
compose_frame_links <- function(first, second, type = NULL, metadata = list()) {
  .validate_frame_link(first, "first")
  .validate_frame_link(second, "second")
  if (!identical(first$target, second$source) ||
      !identical(first$target_axis, second$source_axis)) {
    .study_abort(
      "Frame links do not share one compatible intermediate endpoint and axis.",
      field = "links"
    )
  }
  if (is.null(type)) {
    if (!identical(first$type, second$type)) {
      .study_abort("Composed links with different types require an explicit type.",
                   field = "type")
    }
    type <- first$type
  }
  map <- NULL
  if (!is.null(first$map) && !is.null(second$map)) {
    left <- first$map[c(".source_id", ".target_id")]
    names(left) <- c(".source_id", ".intermediate_id")
    right <- second$map[c(".source_id", ".target_id")]
    names(right) <- c(".intermediate_id", ".target_id")
    rows <- merge(left, right, by = ".intermediate_id", sort = FALSE)
    map <- unique(tibble::as_tibble(rows[c(".source_id", ".target_id")]))
  }
  operator <- NULL
  if (!is.null(first$operator) || !is.null(second$operator)) {
    if (is.null(first$operator) || is.null(second$operator)) {
      .study_abort("Both links require operators for operator composition.",
                   field = "operator")
    }
    first_operator <- feature_map_operator(first$operator)
    second_operator <- feature_map_operator(second$operator)
    if (inherits(first_operator, "array_source") ||
        inherits(second_operator, "array_source")) {
      .study_abort(
        "Array-source feature operators cannot be composed implicitly; materialize or provide a composed operator explicitly.",
        field = "operator"
      )
    }
    assert_compatible_space(
      feature_map_target_space(first$operator),
      feature_map_source_space(second$operator)
    )
    operator <- feature_map(
      feature_map_source_space(first$operator),
      feature_map_target_space(second$operator),
      second_operator %*% first_operator,
      map_type = "composition",
      provenance = list(
        first = feature_map_digest(first$operator),
        second = feature_map_digest(second$operator)
      )
    )
  }
  frame_link(
    first$source, second$target, type = type, map = map,
    source_axis = first$source_axis, target_axis = second$target_axis,
    metadata = metadata, operator = operator
  )
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
      if (!all(c(value$source, value$target) %in% names(frames))) {
        .study_abort(sprintf("Study link '%s' has an unknown endpoint.", id), link = id)
      }
      if (!is.null(value$map)) {
        source_ids <- .study_axis_ids(
          frames[[value$source]], value$source_axis, value$source
        )
        target_ids <- .study_axis_ids(
          frames[[value$target]], value$target_axis, value$target
        )
        if (any(!value$map$.source_id %in% source_ids) ||
            any(!value$map$.target_id %in% target_ids)) {
          .study_abort(sprintf("Study link '%s' map contains unknown axis IDs.", id), link = id)
        }
      }
      typed_map <- value$operator
      if (!is.null(typed_map)) {
        validate_feature_map(typed_map)
        if (!value$type %in% c("mapping", "alignment") ||
            !identical(value$source_axis, "feature") ||
            !identical(value$target_axis, "feature")) {
          .study_abort(
            sprintf("Study link '%s' uses an operator outside a feature mapping or alignment.", id),
            link = id
          )
        }
        source_frame <- frames[[value$source]]
        target_frame <- frames[[value$target]]
        if (inherits(source_frame, "fmri_collection") ||
            inherits(target_frame, "fmri_collection")) {
          .study_abort(
            sprintf("Study link '%s' operator endpoints must be single frames.", id),
            link = id
          )
        }
        assert_compatible_space(feature_map_source_space(typed_map),
                                space(source_frame))
        assert_compatible_space(feature_map_target_space(typed_map),
                                space(target_frame))
      }
    }
  }
  links
}

.validate_study_tables <- function(tables, entities_value) {
  tables <- tryCatch(
    .validate_table_registry(tables, "Study"),
    fmridataset_error_table = function(error) {
      .study_abort(conditionMessage(error), field = error$field %||% "tables")
    }
  )
  if (length(tables)) {
    ids <- names(tables)
    for (table_name in ids) {
      data <- table_data(tables[[table_name]])
      for (entity_name in entity_names(entities_value)) {
        entity_value <- entities_value[[entity_name]]
        key <- entity_key(entity_value)
        if (!key %in% names(data)) next
        values <- as.character(data[[key]])
        present <- !is.na(values)
        if (any(!values[present] %in% entity_ids(entity_value))) {
          .study_abort(
            sprintf("Study table '%s' contains unknown %s entity IDs.", table_name, entity_name),
            table = table_name,
            entity = entity_name
          )
        }
      }
    }
  }
  tables
}

#' Construct a linked fMRI study
#'
#' @param frames Named `fmri_frame` or `fmri_collection` representations.
#' @param entities Shared authoritative entity registry.
#' @param links Named `frame_link` descriptors.
#' @param tables Named typed relational tables.
#' @param metadata Unaligned study-level metadata.
#' @param provenance `NULL` or a validated `provenance_graph`.
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
  metadata <- .normalize_container_metadata(
    metadata, .study_alignment_domains(frames, entities_value)
  )
  provenance <- .validate_container_provenance(provenance, "Study")
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

.study_base <- function(x) x
.study_raw_frames <- function(x) x$frames

#' Validate a study
#'
#' @param x An `fmri_study`.
#' @return `x`, invisibly.
#' @export
validate_fmri_study <- function(x) {
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
#' @param x An `fmri_study`.
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
  shared <- entities(x)
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
#' Study link and table accessors
#'
#' @param x An `fmri_study`.
#' @param name Stable link or table name.
#' @return A registry or one descriptor/table.
#' @name study-registries
NULL

#' @rdname study-registries
#' @export
study_links <- function(x) {
  validate_fmri_study(x)
  x$links
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

.filter_study_table <- function(value, selections, shared) {
  data <- table_data(value)
  keep <- rep(TRUE, nrow(data))
  for (entity_name in names(selections)) {
    key <- entity_key(shared[[entity_name]])
    if (key %in% names(data)) keep <- keep & !is.na(data[[key]]) & data[[key]] %in% selections[[entity_name]]
  }
  data <- data[keep, , drop = FALSE]
  if (inherits(value, "fmri_event_table")) {
    return(event_table(data, key = event_key(value), metadata = value$metadata))
  }
  auxiliary_table(
    data, key = table_key(value), role = table_role(value), metadata = value$metadata
  )
}

#' @rdname study-registries
#' @export
study_tables <- function(x) {
  x$tables
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
#' @param x An `fmri_study`.
#' @param entity Bare or quoted shared entity name.
#' @param predicate A scalar-metadata predicate evaluated on that entity table.
#' @return A self-contained `fmri_study` whose numerical sources remain lazy.
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
  shared <- entities(x)
  frames <- lapply(
    x$frames,
    .filter_study_frame_entity,
    entity_name = entity_name,
    selected_ids = selected_ids,
    shared = shared
  )
  restricted <- visible_registry
  restricted[[entity_name]] <- visible[which(keep)]
  class(restricted) <- c("entity_registry", "list")
  frames <- lapply(frames, .contextualize_study_frame, shared = restricted)
  links <- lapply(x$links, function(value) {
    if (is.null(value$map)) return(value)
    source_ids <- .study_axis_ids(
      frames[[value$source]], value$source_axis, value$source
    )
    target_ids <- .study_axis_ids(
      frames[[value$target]], value$target_axis, value$target
    )
    keep_map <- value$map$.source_id %in% source_ids &
      value$map$.target_id %in% target_ids
    value$map <- value$map[keep_map, , drop = FALSE]
    value
  })
  selections <- stats::setNames(list(selected_ids), entity_name)
  tables <- lapply(
    x$tables, .filter_study_table, selections = selections, shared = visible_registry
  )
  fmri_study(
    frames = frames, entities = restricted, links = links, tables = tables,
    metadata = x$metadata, provenance = x$provenance
  )
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
  .canonical_digest(list(
    schema_version = 1L,
    frames = lapply(x$frames, .study_representation_digest),
    entities = entities(x),
    links = study_links(x),
    tables = study_tables(x),
    metadata = x$metadata,
    provenance = x$provenance
  ))
}

#' @export
print.fmri_study <- function(x, ...) {
  validate_fmri_study(x)
  cat("<fmri_study>", length(study_ids(x)), "representations\n")
  cat("  entities:", paste(entity_names(x), collapse = ", "), "\n")
  cat("  links:", length(study_links(x)), "\n")
  invisible(x)
}
