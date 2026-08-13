.entity_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_entity", ...)
}

.validate_entity_scalar_data <- function(data, key) {
  if (!is.data.frame(data)) {
    .entity_abort("Entity data must be a data frame.", field = "data")
  }
  if (!is.character(key) || length(key) != 1L || is.na(key) || !nzchar(key)) {
    .entity_abort("Entity key must be one non-empty column name.", field = "key")
  }
  if (!key %in% names(data)) {
    .entity_abort(
      sprintf("Entity key column '%s' is absent from data.", key),
      field = "key",
      key = key
    )
  }
  non_scalar <- vapply(data, function(value) {
    is.list(value) || !is.null(dim(value)) || length(value) != nrow(data)
  }, logical(1))
  if (any(non_scalar)) {
    .entity_abort(
      "Entity data columns must be scalar annotations; use axis_block for multivariate values.",
      field = "data",
      columns = names(data)[non_scalar]
    )
  }
  invisible(TRUE)
}

#' Construct a keyed entity frame
#'
#' An entity frame stores one row per subject, session, run, stimulus, item, or
#' other study entity. Scalar annotations live in `data`; multivariate values
#' live in named, first-axis-aligned `axis_block` objects.
#'
#' @param data Scalar entity annotations with one row per entity.
#' @param key Name of the stable primary-key column.
#' @param blocks Named entity-aligned `axis_block` objects.
#' @param entity_type Optional semantic type such as `"subject"` or
#'   `"stimulus"`.
#' @param metadata Additional serializable metadata.
#' @return An `entity_frame`, also implementing the `axis_frame` contract.
#' @export
entity_frame <- function(data, key, blocks = list(), entity_type = NULL,
                         metadata = list()) {
  data <- tibble::as_tibble(data)
  .validate_entity_scalar_data(data, key)
  if (!is.null(entity_type) &&
    (!is.character(entity_type) || length(entity_type) != 1L ||
      is.na(entity_type) || !nzchar(entity_type))) {
    .entity_abort("entity_type must be NULL or one non-empty string.", field = "entity_type")
  }
  out <- axis_frame(
    data = data,
    blocks = blocks,
    id = as.character(data[[key]]),
    axis = "entity",
    id_col = key,
    metadata = metadata
  )
  out$key <- key
  out$entity_type <- entity_type
  class(out) <- c("entity_frame", class(out))
  if (.source_contains_runtime_state(out)) {
    .entity_abort(
      "Entity frames cannot contain runtime functions, environments, or external pointers.",
      field = "runtime_state"
    )
  }
  out
}

#' Entity-frame accessors
#'
#' @param x An `entity_frame`.
#' @return The stable key name, entity IDs, scalar data, or aligned blocks.
#' @name entity-frame-accessors
NULL

#' @rdname entity-frame-accessors
#' @export
entity_key <- function(x) {
  if (!inherits(x, "entity_frame")) .entity_abort("x must be an entity_frame.")
  x$key
}

#' @rdname entity-frame-accessors
#' @export
entity_ids <- function(x) {
  if (!inherits(x, "entity_frame")) .entity_abort("x must be an entity_frame.")
  axis_ids(x)
}

#' @rdname entity-frame-accessors
#' @export
entity_data <- function(x) {
  if (!inherits(x, "entity_frame")) .entity_abort("x must be an entity_frame.")
  axis_data(x)
}

#' @rdname entity-frame-accessors
#' @export
entity_blocks <- function(x) {
  if (!inherits(x, "entity_frame")) .entity_abort("x must be an entity_frame.")
  axis_blocks(x)
}

#' @export
`[.entity_frame` <- function(x, i, ...) {
  if (missing(i)) i <- seq_len(length(x))
  value <- `[.axis_frame`(x, i, ...)
  value$key <- x$key
  value$entity_type <- x$entity_type
  class(value) <- c("entity_frame", class(value))
  value
}

#' @export
print.entity_frame <- function(x, ...) {
  label <- x$entity_type %||% "entity"
  cat("<entity_frame>", length(x), label, "records\n")
  cat("  key:", entity_key(x), "\n")
  if (length(entity_blocks(x))) {
    cat("  blocks:", paste(names(entity_blocks(x)), collapse = ", "), "\n")
  }
  invisible(x)
}

.coerce_entity_registry_entry <- function(value, name) {
  if (inherits(value, "entity_frame")) {
    return(value)
  }
  if (!is.list(value) || is.null(value$data)) {
    .entity_abort(
      sprintf("Registry entry '%s' must be an entity_frame.", name),
      entity = name
    )
  }
  key <- value$key
  if (is.null(key)) {
    conventional <- paste0(name, "_id")
    if (conventional %in% names(value$data)) key <- conventional
  }
  if (is.null(key)) {
    .entity_abort(
      sprintf("Legacy registry entry '%s' must declare its key.", name),
      entity = name,
      field = "key"
    )
  }
  entity_frame(
    data = value$data,
    key = key,
    blocks = value$blocks %||% list(),
    entity_type = value$entity_type %||% name,
    metadata = value$metadata %||% list()
  )
}

#' Construct and validate an entity registry
#'
#' @param entities A named list of `entity_frame` objects. For evolutionary
#'   compatibility, a named legacy entry with `data`, `blocks`, and either
#'   `key` or a conventional `<name>_id` column is normalized immediately.
#' @param ... Alternatively, named `entity_frame` objects.
#' @return A named `entity_registry`.
#' @export
entity_registry <- function(entities = list(), ...) {
  dots <- list(...)
  if (inherits(entities, "entity_registry") && !length(dots)) {
    validate_entity_registry(entities)
    return(entities)
  }
  if (length(dots)) {
    if (length(entities)) {
      .entity_abort("Supply entity registries either as a list or as named arguments, not both.")
    }
    entities <- dots
  }
  if (!is.list(entities)) {
    .entity_abort("entities must be a named list.", field = "entities")
  }
  if (length(entities)) {
    names_value <- names(entities)
    if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
      anyDuplicated(names_value)) {
      .entity_abort("Entity registries must be named with unique, non-empty values.", field = "names")
    }
    entities <- lapply(names_value, function(name) {
      .coerce_entity_registry_entry(entities[[name]], name)
    })
    names(entities) <- names_value
  }
  class(entities) <- c("entity_registry", "list")
  validate_entity_registry(entities)
  entities
}

#' @param x An entity registry.
#' @rdname entity_registry
#' @export
validate_entity_registry <- function(x) {
  if (!inherits(x, "entity_registry") || !is.list(x)) {
    .entity_abort("x must be an entity_registry.", field = "class")
  }
  if (length(x)) {
    names_value <- names(x)
    if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
      anyDuplicated(names_value)) {
      .entity_abort("Entity registries must be named with unique, non-empty values.", field = "names")
    }
    valid <- vapply(x, inherits, logical(1), "entity_frame")
    if (!all(valid)) {
      .entity_abort(
        "Every entity registry entry must be an entity_frame.",
        field = "entities",
        entities = names_value[!valid]
      )
    }
    for (name in names_value) {
      value <- x[[name]]
      .validate_entity_scalar_data(value$data, value$key)
      if (!identical(value$axis, "entity") || !identical(value$id_col, value$key)) {
        .entity_abort(
          sprintf("Entity registry entry '%s' does not implement the entity axis contract.", name),
          entity = name,
          field = "axis"
        )
      }
      .validate_stable_ids(entity_ids(value), paste0("entity:", name))
      if (!identical(as.character(value$data[[value$key]]), entity_ids(value))) {
        .entity_abort(
          sprintf("Entity registry entry '%s' has keys out of alignment.", name),
          entity = name,
          field = "key"
        )
      }
      blocks <- entity_blocks(value)
      if (length(blocks) &&
        (is.null(names(blocks)) || any(!nzchar(names(blocks))) || anyDuplicated(names(blocks)))) {
        .entity_abort(
          sprintf("Entity registry entry '%s' has unnamed or duplicate blocks.", name),
          entity = name,
          field = "blocks"
        )
      }
      for (block_name in names(blocks)) {
        block <- blocks[[block_name]]
        if (!inherits(block, "axis_block") ||
          .data_leading_dim(axis_block_data(block)) != nrow(value$data)) {
          .entity_abort(
            sprintf("Entity block '%s.%s' is not aligned with its entity keys.", name, block_name),
            entity = name,
            block = block_name,
            field = "blocks"
          )
        }
      }
    }
  }
  if (.source_contains_runtime_state(x)) {
    .entity_abort(
      "Entity registries cannot contain runtime functions, environments, or external pointers.",
      field = "runtime_state"
    )
  }
  invisible(x)
}

#' Access entities from a frame or registry
#'
#' @param x An `fmri_frame`, view, or `entity_registry`.
#' @param name One registered entity name.
#' @param ... Additional method arguments.
#' @return `entities()` returns the registry; `entity()` returns one
#'   `entity_frame`; `entity_names()` returns registry names.
#' @name entity-accessors
NULL

#' @rdname entity-accessors
#' @export
entities <- function(x, ...) UseMethod("entities")

#' @export
entities.entity_registry <- function(x, ...) x

#' @rdname entity-accessors
#' @export
entity <- function(x, name, ...) UseMethod("entity")

#' @export
entity.entity_registry <- function(x, name, ...) {
  if (!is.character(name) || length(name) != 1L || is.na(name) || !name %in% names(x)) {
    label <- if (length(name)) as.character(name)[[1L]] else ""
    .entity_abort(sprintf("Unknown entity '%s'.", label), entity = name)
  }
  x[[name]]
}

#' @rdname entity-accessors
#' @export
entity_names <- function(x) names(entities(x))

#' Compute a stable entity-registry digest
#'
#' @param x A frame, view, or entity registry.
#' @return A hexadecimal digest over the normalized registry.
#' @export
entity_registry_digest <- function(x) {
  x <- entities(x)
  validate_entity_registry(x)
  .canonical_digest(x)
}

#' @export
print.entity_registry <- function(x, ...) {
  cat("<entity_registry>", length(x), "entity types\n")
  if (length(x)) cat("  ", paste(names(x), collapse = ", "), "\n", sep = "")
  invisible(x)
}
