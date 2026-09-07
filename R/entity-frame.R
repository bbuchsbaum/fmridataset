.entity_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_entity", ...)
}

.validate_entity_scalar_data <- function(data, key) {
  if (!is.data.frame(data)) {
    .entity_abort("Entity data must be a data frame.", field = "data")
  }
  .assert_one_string(
    key, "key", .entity_abort,
    message = "Entity key must be one non-empty column name."
  )
  if (!key %in% names(data)) {
    .entity_abort(
      sprintf("Entity key column '%s' is absent from data.", key),
      field = "key",
      key = key
    )
  }
  .assert_scalar_columns(
    data, .entity_abort,
    "Entity data columns must be scalar annotations; use axis_block for multivariate values."
  )
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
#' @examples
#' x <- entity_frame(
#'   data = tibble::tibble(
#'     stimulus_id = c("stim-1", "stim-2", "stim-3"),
#'     category = c("face", "scene", "object")
#'   ),
#'   key = "stimulus_id"
#' )
#' entity_ids(x)
#' @export
entity_frame <- function(data, key, blocks = list(), entity_type = NULL,
                         metadata = list()) {
  data <- tibble::as_tibble(data)
  .validate_entity_scalar_data(data, key)
  .assert_optional_string(
    entity_type, "entity_type", .entity_abort,
    message = "entity_type must be NULL or one non-empty string."
  )
  .assert_no_runtime_state(
    list(blocks = blocks, metadata = metadata), .entity_abort,
    "Entity frames cannot contain runtime functions, environments, or external pointers."
  )
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
  .assert_no_runtime_state(
    out, .entity_abort,
    "Entity frames cannot contain runtime functions, environments, or external pointers."
  )
  out
}

#' Entity-frame accessors
#'
#' @param x An `entity_frame`.
#' @return The stable key name, entity IDs, scalar data, or aligned blocks.
#' @examples
#' embedding <- axis_block(matrix(as.double(1:8), 2, 4), role = "embedding")
#' x <- entity_frame(
#'   data = tibble::tibble(stimulus_id = c("stim-1", "stim-2")),
#'   key = "stimulus_id",
#'   blocks = list(semantic = embedding)
#' )
#' entity_key(x)
#' entity_ids(x)
#' entity_data(x)
#' entity_blocks(x)
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
#' @examples
#' subjects <- entity_frame(
#'   data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
#'   key = "subject_id"
#' )
#' registry <- entity_registry(subject = subjects)
#' entity_names(registry)
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
    .assert_unique_names(
      entities, .entity_abort,
      "Entity registries must be named with unique, non-empty values."
    )
    names_value <- names(entities)
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
    .assert_unique_names(
      x, .entity_abort,
      "Entity registries must be named with unique, non-empty values."
    )
    names_value <- names(x)
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
      .assert_aligned_blocks(
        entity_blocks(value), nrow(value$data), .entity_abort,
        what = sprintf("Entity '%s'", name), entity = name
      )
    }
  }
  .assert_no_runtime_state(
    x, .entity_abort,
    "Entity registries cannot contain runtime functions, environments, or external pointers."
  )
  invisible(x)
}

#' Access entities from a frame or registry
#'
#' @param x An `fmri_frame`, view, or `entity_registry`.
#' @param name One registered entity name.
#' @param ... Additional method arguments.
#' @return `entities()` returns the registry; `entity()` returns one
#'   `entity_frame`; `entity_names()` returns registry names.
#' @examples
#' subjects <- entity_frame(
#'   data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
#'   key = "subject_id"
#' )
#' registry <- entity_registry(subject = subjects)
#' entities(registry)
#' entity(registry, "subject")
#' entity_names(registry)
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
#' @examples
#' subjects <- entity_frame(
#'   data = tibble::tibble(subject_id = c("sub-1", "sub-2")),
#'   key = "subject_id"
#' )
#' registry <- entity_registry(subject = subjects)
#' entity_registry_digest(registry)
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
