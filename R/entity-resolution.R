.resolution_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_resolution", ...)
}

.validate_resolve_flag <- function(resolve) {
  if (!is.logical(resolve) || length(resolve) != 1L || is.na(resolve)) {
    .resolution_abort("resolve must be TRUE or FALSE.", field = "resolve")
  }
  resolve
}

.resolution_domain <- function(x, domain) {
  if (identical(domain, "observation")) {
    return(list(ids = observation_ids(x), data = axis_data(observation_axis(x))))
  }
  if (!startsWith(domain, "entity:")) {
    return(NULL)
  }
  name <- sub("^entity:", "", domain)
  value <- entity(x, name)
  list(ids = entity_ids(value), data = entity_data(value))
}

.merge_resolved_ids <- function(existing, candidate, entity_name) {
  both <- !is.na(existing) & !is.na(candidate)
  if (any(existing[both] != candidate[both])) {
    .resolution_abort(
      sprintf("Entity '%s' has conflicting paths from observations.", entity_name),
      entity = entity_name
    )
  }
  fill <- is.na(existing) & !is.na(candidate)
  existing[fill] <- candidate[fill]
  list(value = existing, changed = any(fill))
}

.resolve_entity_maps <- function(x) {
  registry <- relations(x)
  maps <- list(observation = observation_ids(x))
  key_names <- relation_names(registry)[vapply(
    registry, inherits, logical(1), "key_relation"
  )]
  if (!length(key_names)) {
    return(maps)
  }

  max_passes <- length(key_names) + length(entity_names(x)) + 1L
  for (pass in seq_len(max_passes)) {
    changed <- FALSE
    for (relation_name in key_names) {
      descriptor <- registry[[relation_name]]
      if (!descriptor$source %in% names(maps) ||
        !startsWith(descriptor$target, "entity:")) {
        next
      }
      source_domain <- .resolution_domain(x, descriptor$source)
      source_ids <- maps[[descriptor$source]]
      positions <- match(source_ids, source_domain$ids)
      target_ids <- rep(NA_character_, length(source_ids))
      present <- !is.na(positions)
      if (any(present)) {
        target_ids[present] <- as.character(
          source_domain$data[[descriptor$key]][positions[present]]
        )
      }
      target <- descriptor$target
      entity_name <- sub("^entity:", "", target)
      if (is.null(maps[[target]])) {
        maps[[target]] <- target_ids
        changed <- TRUE
      } else {
        merged <- .merge_resolved_ids(maps[[target]], target_ids, entity_name)
        maps[[target]] <- merged$value
        changed <- changed || merged$changed
      }
    }
    if (!changed) break
  }
  maps
}

.resolved_observation_data <- function(x, maps = .resolve_entity_maps(x)) {
  out <- axis_data(observation_axis(x))
  registry <- entities(x)
  for (entity_name in entity_names(registry)) {
    domain <- paste0("entity:", entity_name)
    mapped_ids <- maps[[domain]]
    if (is.null(mapped_ids)) next
    entity_value <- registry[[entity_name]]
    positions <- match(mapped_ids, entity_ids(entity_value))
    data <- entity_data(entity_value)
    lifted_names <- paste(entity_name, names(data), sep = ".")
    collision <- intersect(lifted_names, names(out))
    if (length(collision)) {
      .resolution_abort(
        sprintf("Resolved entity column '%s' collides with observation metadata.", collision[[1L]]),
        column = collision[[1L]],
        entity = entity_name
      )
    }
    for (i in seq_along(data)) {
      out[[lifted_names[[i]]]] <- data[[i]][positions]
    }
  }
  out
}

.row_index_source <- function(source, rows) {
  source <- as_array_source(source)
  shape <- source_shape(source)
  if (length(shape) != 2L) {
    .resolution_abort("Entity blocks must be two dimensional for lazy lifting.")
  }
  if (!is.numeric(rows) || any(!is.na(rows) & rows != as.integer(rows)) ||
    any(!is.na(rows) & (rows < 1L | rows > shape[[1L]]))) {
    .resolution_abort("Lifted entity row indices are invalid.", field = "rows")
  }
  out <- structure(
    list(
      source = source,
      rows = as.integer(rows),
      shape = c(length(rows), shape[[2L]]),
      schema_version = 1L
    ),
    class = c("row_index_source", "array_source")
  )
  validate_array_source(out)
  out
}

.sparse_entity_source <- function(data) {
  if (!inherits(data, "Matrix") || length(dim(data)) != 2L) {
    .resolution_abort("Sparse entity blocks must be two-dimensional Matrix objects.")
  }
  out <- structure(
    list(
      data = data,
      shape = as.integer(dim(data)),
      schema_version = 1L
    ),
    class = c("sparse_entity_source", "array_source")
  )
  validate_array_source(out)
  out
}

.entity_block_source <- function(data) {
  if (inherits(data, "Matrix")) .sparse_entity_source(data) else as_array_source(data)
}

#' @export
source_shape.sparse_entity_source <- function(x, ...) as.integer(x$shape)

#' @export
source_dtype.sparse_entity_source <- function(x, ...) "float64"

#' @export
source_chunks.sparse_entity_source <- function(x, ...) pmax(1L, source_shape(x))

#' @export
source_capabilities.sparse_entity_source <- function(x, ...) {
  c("row_slice", "column_slice", "block_slice", "serializable")
}

#' @export
source_fingerprint.sparse_entity_source <- function(x, ...) {
  .canonical_digest(list(
    type = "sparse_entity_source",
    schema_version = x$schema_version,
    data = x$data
  ))
}

#' @export
source_open.sparse_entity_source <- function(x, ...) {
  structure(
    list(source = x),
    class = c("sparse_entity_source_handle", "array_source_handle")
  )
}

#' @export
source_read.sparse_entity_source <- function(x, observations = NULL,
                                             features = NULL, ...) {
  shape <- source_shape(x)
  observations <- .normalize_source_index(observations, shape[[1L]])
  features <- .normalize_source_index(features, shape[[2L]])
  if (!length(observations) || !length(features)) {
    return(matrix(numeric(), nrow = length(observations), ncol = length(features)))
  }
  as.matrix(x$data[observations, features, drop = FALSE])
}

#' @export
source_close.sparse_entity_source <- function(x, ...) invisible(TRUE)

#' @export
source_shape.row_index_source <- function(x, ...) as.integer(x$shape)

#' @export
source_dtype.row_index_source <- function(x, ...) source_dtype(x$source)

#' @export
source_chunks.row_index_source <- function(x, ...) {
  pmin(source_chunks(x$source), pmax(1L, source_shape(x)))
}

#' @export
source_capabilities.row_index_source <- function(x, ...) {
  c("row_slice", "column_slice", "block_slice", "serializable")
}

#' @export
source_fingerprint.row_index_source <- function(x, ...) {
  .canonical_digest(list(
    type = "row_index_source",
    schema_version = x$schema_version,
    source = source_fingerprint(x$source),
    rows = x$rows
  ))
}

#' @export
source_open.row_index_source <- function(x, ...) {
  structure(
    list(source = x),
    class = c("row_index_source_handle", "array_source_handle")
  )
}

#' @export
source_read.row_index_source <- function(x, observations = NULL,
                                         features = NULL, ...) {
  shape <- source_shape(x)
  observations <- .normalize_source_index(observations, shape[[1L]])
  features <- .normalize_source_index(features, shape[[2L]])
  # Typed by the declared dtype: matrix(NA, ...) is logical, so a lift with no
  # present rows used to return a logical matrix while source_dtype() reported
  # the child's dtype.
  out <- .realized_na_matrix(source_dtype(x), length(observations), length(features))
  if (!length(observations) || !length(features)) {
    return(out)
  }

  mapped <- x$rows[observations]
  present <- !is.na(mapped)
  if (!any(present)) {
    return(out)
  }
  unique_rows <- unique(mapped[present])
  values <- source_read(
    x$source,
    observations = unique_rows,
    features = features,
    ...
  )
  out[present, ] <- values[match(mapped[present], unique_rows), , drop = FALSE]
  out
}

#' @export
source_close.row_index_source <- function(x, ...) invisible(TRUE)

.resolved_observation_blocks <- function(x, maps = .resolve_entity_maps(x)) {
  out <- axis_blocks(observation_axis(x))
  registry <- entities(x)
  for (entity_name in entity_names(registry)) {
    domain <- paste0("entity:", entity_name)
    mapped_ids <- maps[[domain]]
    if (is.null(mapped_ids)) next
    entity_value <- registry[[entity_name]]
    rows <- match(mapped_ids, entity_ids(entity_value))
    for (block_name in names(entity_blocks(entity_value))) {
      lifted_name <- paste(entity_name, block_name, sep = ".")
      if (lifted_name %in% names(out)) {
        .resolution_abort(
          sprintf("Resolved entity block '%s' collides with an observation block.", lifted_name),
          block = lifted_name,
          entity = entity_name
        )
      }
      block <- entity_blocks(entity_value)[[block_name]]
      metadata <- block$metadata
      metadata$.fmridataset_lift <- list(
        entity = entity_name,
        block = block_name,
        entity_digest = .canonical_digest(entity_value),
        relation_digest = relation_registry_digest(relations(x))
      )
      out[[lifted_name]] <- axis_block(
        data = .row_index_source(.entity_block_source(axis_block_data(block)), rows),
        components = block_components(block),
        role = block$role,
        units = block$units,
        metadata = metadata
      )
    }
  }
  out
}
