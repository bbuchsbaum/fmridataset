.frame_schema_version <- 1L

# Column descriptors record the storage type plus the attributes that change
# a column's meaning: factor levels and ordering, the time zone of a POSIXct
# instant, and the units of a difftime duration. `tzone` and `units` are only
# present for those column classes so other descriptors keep their encoding.
.schema_column <- function(value) {
  out <- list(
    class = class(value),
    typeof = typeof(value),
    levels = if (is.factor(value)) levels(value) else NULL,
    ordered = is.ordered(value)
  )
  if (inherits(value, "POSIXct")) {
    tzone <- attr(value, "tzone", exact = TRUE)
    out$tzone <- if (is.null(tzone)) "" else as.character(tzone)[[1L]]
  }
  if (inherits(value, "difftime")) {
    out$units <- as.character(units(value))
  }
  out
}

.schema_columns <- function(data) {
  out <- lapply(data, .schema_column)
  names(out) <- names(data)
  out
}

.schema_data_shape <- function(data) {
  if (inherits(data, "array_source")) {
    return(as.integer(source_shape(data)))
  }
  shape <- dim(data)
  if (is.null(shape)) c(length(data), 1L) else as.integer(shape)
}

.schema_blocks <- function(blocks) {
  out <- lapply(blocks, function(block) {
    shape <- .schema_data_shape(axis_block_data(block))
    list(
      trailing_shape = as.integer(shape[-1L]),
      component_ids = block_component_ids(block),
      components = .schema_columns(block_components(block)),
      role = block$role,
      units = block$units,
      metadata = block$metadata
    )
  })
  names(out) <- names(blocks)
  out
}

.schema_relations <- function(registry) {
  out <- lapply(registry, function(value) {
    if (inherits(value, "key_relation")) {
      return(list(
        type = "key", key = value$key, source = value$source,
        target = value$target, allow_missing = value$allow_missing,
        metadata = value$metadata
      ))
    }
    list(
      type = "sparse", from = value$from, to = value$to,
      from_col = value$from_col, to_col = value$to_col,
      weight = value$weight, directed = value$directed,
      columns = .schema_columns(value$data), metadata = value$metadata
    )
  })
  names(out) <- names(registry)
  out
}

.schema_tables <- function(tables) {
  out <- lapply(tables, function(value) {
    if (inherits(value, "fmri_event_table")) {
      return(list(
        type = "event", key = event_key(value),
        columns = .schema_columns(event_data(value)), metadata = value$metadata
      ))
    }
    if (inherits(value, "fmri_auxiliary_table")) {
      return(list(
        type = "auxiliary", key = table_key(value), role = table_role(value),
        columns = .schema_columns(table_data(value)), metadata = value$metadata
      ))
    }
    list(
      type = class(value)[1L], key = attr(value, "key", exact = TRUE),
      columns = if (is.data.frame(value)) .schema_columns(value) else NULL
    )
  })
  names(out) <- names(tables)
  out
}

#' Compute the canonical metadata-only frame schema
#'
#' The schema describes aligned assay annotations, scalar column types and
#' factor levels, multivariate block trailing axes and components, spatial
#' identity, entities, relations, auxiliary tables, and active-assay policy.
#' It inspects descriptors and metadata only and never reads numerical values.
#'
#' @param x An `fmri_frame`, synchronized view, or canonical frame schema.
#' @return A serializable `fmri_frame_schema`.
#' @examples
#' sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
#'   observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
#'   space = sp
#' )
#' schema <- frame_schema(frame)
#' schema$active_assay
#' @export
frame_schema <- function(x) {
  if (inherits(x, "fmri_frame_schema")) {
    validate_frame_schema(x)
    return(x)
  }
  if (!inherits(x, "fmri_frame")) {
    .fds_schema_abort("x must be an fmri_frame, view, or frame schema.", "object_type")
  }
  observation <- observation_axis(x)
  feature <- feature_axis(x)
  assay_values <- assays(x)
  entity_values <- entities(x)
  out <- list(
    schema = list(id = "org.fmridataset.frame-schema/v1", version = .frame_schema_version),
    assays = lapply(assay_values, function(value) {
      list(
        dtype = value$dtype, role = value$role, units = value$units,
        metadata = value$metadata
      )
    }),
    observation = list(
      count = length(observation), id_column = observation$id_col,
      columns = .schema_columns(axis_data(observation)),
      blocks = .schema_blocks(axis_blocks(observation)),
      metadata = observation$metadata
    ),
    feature = list(
      count = length(feature), id_column = feature$id_col,
      columns = .schema_columns(axis_data(feature)),
      blocks = .schema_blocks(axis_blocks(feature)),
      metadata = feature$metadata,
      space = list(
        type = class(space(x))[1L], digest = space_digest(space(x)),
        feature_ids = feature_ids(x)
      )
    ),
    entities = lapply(entity_values, function(value) {
      list(
        key = entity_key(value), entity_type = value$entity_type,
        columns = .schema_columns(entity_data(value)),
        blocks = .schema_blocks(entity_blocks(value)), metadata = value$metadata
      )
    }),
    relations = .schema_relations(relations(x)),
    tables = .schema_tables(x$tables %||% x$base$tables),
    active_assay = list(policy = "named", name = active_assay(x))
  )
  class(out) <- c("fmri_frame_schema", "list")
  validate_frame_schema(out)
  out
}

#' Validate and compare canonical frame schemas
#'
#' @param schema A canonical frame schema.
#' @param x A frame, view, or schema to validate.
#' @param reference A frame, view, or schema used as the reference contract.
#' @param mode Comparison mode. `same` compares the complete schema;
#'   `collection` permits different observation counts and feature identities;
#'   `bind` permits different observation counts but requires one feature
#'   identity.
#' @details `compare_frame_schema()` treats named records, such as column
#'   descriptors and the assay, entity, relation, and table registries, as
#'   unordered: column order and assay registry order are not semantic. This
#'   matches `frame_schema_digest()`, whose canonical encoding sorts record
#'   fields by name. Unnamed sequences such as feature and component IDs remain
#'   order-sensitive. `validate_frame_schema()` checks the shape of every
#'   section and rejects missing or unknown top-level fields.
#' @return Validators return their input invisibly. `compare_frame_schema()`
#'   returns a structured `frame_schema_compatibility` report.
#'   `frame_schema_digest()` returns a single hex digest string.
#' @examples
#' sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
#'   observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
#'   space = sp
#' )
#' schema <- frame_schema(frame)
#' validate_frame_schema(schema)
#' compare_frame_schema(frame, frame)$compatible
#' frame_schema_digest(frame)
#' @name frame-schema-validation
NULL

#' @rdname frame-schema-validation
#' @export
validate_frame_schema <- function(schema) {
  required <- c(
    "schema", "assays", "observation", "feature", "entities", "relations",
    "tables", "active_assay"
  )
  if (!inherits(schema, "fmri_frame_schema")) {
    .fds_schema_abort("Invalid canonical frame schema.", "frame_schema")
  }
  value <- unclass(schema)
  .fds_assert_fields(value, required, "frame_schema", "The canonical frame schema")
  if (!is.list(value$schema) ||
    !identical(value$schema$id, "org.fmridataset.frame-schema/v1") ||
    !identical(value$schema$version, .frame_schema_version)) {
    .fds_schema_abort("Invalid canonical frame schema.", "frame_schema")
  }
  if (!is.list(value$assays) || !length(value$assays) ||
    !.has_unique_names(value$assays)) {
    .fds_schema_abort("Frame schema assays must be uniquely named.", "assays")
  }
  for (name in names(value$assays)) {
    assay <- value$assays[[name]]
    if (!is.list(assay) ||
      !all(c("dtype", "role", "units", "metadata") %in% names(assay)) ||
      !.is_one_string(assay$dtype)) {
      .fds_schema_abort(
        sprintf("Frame schema assay '%s' is malformed.", name),
        paste0("assays.", name)
      )
    }
  }
  .validate_schema_axis(value$observation, "observation")
  .validate_schema_axis(value$feature, "feature")
  space <- value$feature$space
  if (!is.list(space) || !.is_one_string(space$type) ||
    !.is_one_string(space$digest) || !is.character(space$feature_ids) ||
    length(space$feature_ids) != value$feature$count) {
    .fds_schema_abort(
      "Frame schema feature space must record a type, digest, and one ID per feature.",
      "feature.space"
    )
  }
  .validate_schema_registry(value$entities, "entities", function(entity, field) {
    if (!is.list(entity) || !.is_one_string(entity$key)) {
      .fds_schema_abort("Frame schema entity is malformed.", field)
    }
    .validate_schema_columns(entity$columns, paste0(field, ".columns"))
    .validate_schema_registry(entity$blocks, paste0(field, ".blocks"), .validate_schema_block)
  })
  .validate_schema_registry(value$relations, "relations", function(relation, field) {
    if (!is.list(relation) || !.is_one_string(relation$type)) {
      .fds_schema_abort("Frame schema relation is malformed.", field)
    }
  })
  .validate_schema_registry(value$tables, "tables", function(table, field) {
    if (!is.list(table) || !.is_one_string(table$type)) {
      .fds_schema_abort("Frame schema table is malformed.", field)
    }
  })
  active <- value$active_assay
  if (!is.list(active) || !identical(active$policy, "named") ||
    !.is_one_string(active$name) || !active$name %in% names(value$assays)) {
    .fds_schema_abort("Frame schema active assay is invalid.", "active_assay")
  }
  invisible(schema)
}

.validate_schema_registry <- function(value, field, validate_item) {
  if (!is.list(value) || is.data.frame(value) || !.has_unique_names(value)) {
    .fds_schema_abort(
      sprintf("Frame schema '%s' must be a uniquely named list.", field), field
    )
  }
  for (name in names(value)) validate_item(value[[name]], paste0(field, ".", name))
  invisible(TRUE)
}

.validate_schema_columns <- function(columns, field) {
  .validate_schema_registry(columns, field, function(column, column_field) {
    if (!is.list(column) || !is.character(column$class) || !length(column$class) ||
      !.is_one_string(column$typeof) ||
      length(setdiff(names(column), c("class", "typeof", "levels", "ordered", "tzone", "units")))) {
      .fds_schema_abort("Frame schema column descriptor is malformed.", column_field)
    }
  })
}

.validate_schema_block <- function(block, field) {
  if (!is.list(block) ||
    !all(c("trailing_shape", "component_ids", "components") %in% names(block)) ||
    !is.character(block$component_ids)) {
    .fds_schema_abort("Frame schema block is malformed.", field)
  }
  .validate_schema_columns(block$components, paste0(field, ".components"))
}

.validate_schema_axis <- function(axis, field) {
  required <- c("count", "id_column", "columns", "blocks", "metadata")
  count <- if (is.list(axis)) axis$count else NULL
  if (!is.list(axis) || is.data.frame(axis) || !all(required %in% names(axis)) ||
    !is.numeric(count) || length(count) != 1L || !is.finite(count) ||
    count < 0 || count > .Machine$integer.max || count != trunc(count) ||
    !.is_one_string(axis$id_column)) {
    .fds_schema_abort(
      sprintf("Frame schema %s axis must record a count, ID column, columns, blocks, and metadata.", field),
      field
    )
  }
  .validate_schema_columns(axis$columns, paste0(field, ".columns"))
  .validate_schema_registry(axis$blocks, paste0(field, ".blocks"), .validate_schema_block)
  invisible(TRUE)
}

.schema_projection <- function(schema, mode) {
  schema <- unclass(frame_schema(schema))
  if (!identical(mode, "same")) schema$observation$count <- NULL
  if (identical(mode, "bind")) {
    schema$active_assay <- NULL
    # Binding rebuilds observation blocks by component identity and permutes
    # each operand's columns into the leading frame's order, so the ordered
    # component list is not part of the bind contract. Component identity is
    # enforced by the bind path itself with an alignment error that names the
    # differing components.
    schema$observation$blocks <- lapply(schema$observation$blocks, function(block) {
      block$component_ids <- NULL
      block
    })
  }
  if (identical(mode, "collection")) {
    schema$feature$count <- NULL
    schema$feature$space$digest <- NULL
    schema$feature$space$feature_ids <- NULL
  }
  schema
}

# Named records (column descriptors, assay and other registries, record
# fields) are compared as unordered sets of fields, matching the canonical
# encoding behind frame_schema_digest(), which sorts record fields by name.
# Unnamed sequences, including ID vectors, stay order-sensitive.
.schema_is_record <- function(x) {
  is.list(x) && length(x) && .has_unique_names(x)
}

.schema_first_mismatch <- function(expected, actual, path = "schema") {
  if (identical(expected, actual)) {
    return(NULL)
  }
  if (.schema_is_record(expected) && .schema_is_record(actual)) {
    expected_names <- sort(names(expected), method = "radix")
    actual_names <- sort(names(actual), method = "radix")
    if (!identical(expected_names, actual_names)) {
      return(list(path = path, expected = expected_names, actual = actual_names))
    }
    for (name in expected_names) {
      mismatch <- .schema_first_mismatch(
        expected[[name]], actual[[name]], paste0(path, ".", name)
      )
      if (!is.null(mismatch)) {
        return(mismatch)
      }
    }
    strip_names <- function(x) {
      values <- attributes(x)
      values$names <- NULL
      values
    }
    if (identical(strip_names(expected), strip_names(actual))) {
      return(NULL)
    }
    return(list(path = path, expected = expected, actual = actual))
  }
  if (is.list(expected) && is.list(actual)) {
    if (!identical(names(expected), names(actual))) {
      return(list(path = path, expected = names(expected), actual = names(actual)))
    }
    for (name in names(expected)) {
      mismatch <- .schema_first_mismatch(
        expected[[name]], actual[[name]], paste0(path, ".", name)
      )
      if (!is.null(mismatch)) {
        return(mismatch)
      }
    }
  }
  list(path = path, expected = expected, actual = actual)
}

#' @rdname frame-schema-validation
#' @export
compare_frame_schema <- function(x, reference,
                                 mode = c("same", "collection", "bind")) {
  mode <- match.arg(mode)
  expected <- .schema_projection(frame_schema(reference), mode)
  actual <- .schema_projection(frame_schema(x), mode)
  mismatch <- .schema_first_mismatch(expected, actual)
  structure(
    list(
      compatible = is.null(mismatch), mode = mode,
      path = mismatch$path %||% NULL,
      expected = mismatch$expected %||% NULL,
      actual = mismatch$actual %||% NULL
    ),
    class = "frame_schema_compatibility"
  )
}

#' @rdname frame-schema-validation
#' @export
validate_against_schema <- function(x, reference,
                                    mode = c("same", "collection", "bind")) {
  report <- compare_frame_schema(x, reference, mode = match.arg(mode))
  if (!isTRUE(report$compatible)) {
    .fds_schema_abort(
      sprintf("Frame schema mismatch at '%s'.", report$path), report$path,
      expected = report$expected, actual = report$actual, mode = report$mode
    )
  }
  invisible(x)
}

#' @rdname frame-schema-validation
#' @export
frame_schema_digest <- function(x) .canonical_digest(unclass(frame_schema(x)))
