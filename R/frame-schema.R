.frame_schema_version <- 1L

.schema_column <- function(value) {
  list(
    class = class(value),
    typeof = typeof(value),
    levels = if (is.factor(value)) levels(value) else NULL,
    ordered = is.ordered(value)
  )
}

.schema_columns <- function(data) {
  out <- lapply(data, .schema_column)
  names(out) <- names(data)
  out
}

.schema_data_shape <- function(data) {
  if (inherits(data, "array_source")) return(as.integer(source_shape(data)))
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
    assays = lapply(assay_values, function(value) list(
      dtype = value$dtype, role = value$role, units = value$units,
      metadata = value$metadata
    )),
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
    entities = lapply(entity_values, function(value) list(
      key = entity_key(value), entity_type = value$entity_type,
      columns = .schema_columns(entity_data(value)),
      blocks = .schema_blocks(entity_blocks(value)), metadata = value$metadata
    )),
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
#' @return Validators return their input invisibly. `compare_frame_schema()`
#'   returns a structured compatibility report.
#' @name frame-schema-validation
NULL

#' @rdname frame-schema-validation
#' @export
validate_frame_schema <- function(schema) {
  required <- c(
    "schema", "assays", "observation", "feature", "entities", "relations",
    "tables", "active_assay"
  )
  if (!inherits(schema, "fmri_frame_schema") ||
      !identical(names(unclass(schema)), required) ||
      !identical(schema$schema$id, "org.fmridataset.frame-schema/v1") ||
      !identical(schema$schema$version, .frame_schema_version)) {
    .fds_schema_abort("Invalid canonical frame schema.", "frame_schema")
  }
  if (!length(schema$assays) || is.null(names(schema$assays)) ||
      any(!nzchar(names(schema$assays))) || anyDuplicated(names(schema$assays))) {
    .fds_schema_abort("Frame schema assays must be uniquely named.", "assays")
  }
  if (!identical(schema$active_assay$policy, "named") ||
      !schema$active_assay$name %in% names(schema$assays)) {
    .fds_schema_abort("Frame schema active assay is invalid.", "active_assay")
  }
  invisible(schema)
}

.schema_projection <- function(schema, mode) {
  schema <- unclass(frame_schema(schema))
  if (!identical(mode, "same")) schema$observation$count <- NULL
  if (identical(mode, "collection")) {
    schema$feature$count <- NULL
    schema$feature$space$digest <- NULL
    schema$feature$space$feature_ids <- NULL
  }
  schema
}

.schema_first_mismatch <- function(expected, actual, path = "schema") {
  if (identical(expected, actual)) return(NULL)
  if (is.list(expected) && is.list(actual)) {
    if (!identical(names(expected), names(actual))) {
      return(list(path = path, expected = names(expected), actual = names(actual)))
    }
    for (name in names(expected)) {
      mismatch <- .schema_first_mismatch(
        expected[[name]], actual[[name]], paste0(path, ".", name)
      )
      if (!is.null(mismatch)) return(mismatch)
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
