.relation_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_relation", ...)
}

.one_relation_string <- function(x, field) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    .relation_abort(sprintf("%s must be one non-empty string.", field), field = field)
  }
  x
}

#' Describe a symbolic foreign-key relation
#'
#' A key relation does not duplicate a mapping. It declares that one scalar
#' column on an observation, feature, or entity table references the stable
#' IDs of one entity frame.
#'
#' @param key Foreign-key column on the source domain.
#' @param target Target entity name. When `NULL`, frame validation infers the
#'   unique entity whose primary key has the same name as `key`.
#' @param source Source domain: `"observation"`, `"feature"`, an entity name,
#'   or an explicit `"entity:<name>"` domain.
#' @param allow_missing Whether missing foreign-key values are permitted.
#' @param metadata Additional serializable metadata.
#' @return A serializable `key_relation` descriptor.
#' @export
key_relation <- function(key, target = NULL, source = "observation",
                         allow_missing = FALSE, metadata = list()) {
  key <- .one_relation_string(key, "key")
  source <- .one_relation_string(source, "source")
  if (!is.null(target)) target <- .one_relation_string(target, "target")
  if (!is.logical(allow_missing) || length(allow_missing) != 1L || is.na(allow_missing)) {
    .relation_abort("allow_missing must be TRUE or FALSE.", field = "allow_missing")
  }
  out <- structure(
    list(
      type = "key",
      key = key,
      source = source,
      target = target,
      allow_missing = allow_missing,
      metadata = metadata,
      schema_version = 1L
    ),
    class = c("key_relation", "fmri_relation")
  )
  if (.source_contains_runtime_state(out)) {
    .relation_abort("Relations cannot contain runtime state.", field = "runtime_state")
  }
  out
}

.validate_sparse_scalar_data <- function(data) {
  non_scalar <- vapply(data, function(value) {
    is.list(value) || !is.null(dim(value)) || length(value) != nrow(data)
  }, logical(1))
  if (any(non_scalar)) {
    .relation_abort(
      "Sparse relation columns must contain scalar values.",
      field = "data",
      columns = names(data)[non_scalar]
    )
  }
}

#' Describe an explicit sparse or many-to-many relation
#'
#' @param data Scalar edge table.
#' @param from Source domain.
#' @param to Target domain.
#' @param from_col Column containing source stable IDs.
#' @param to_col Column containing target stable IDs.
#' @param weight Optional numeric weight column.
#' @param directed Whether edge direction is semantically meaningful.
#' @param metadata Additional serializable metadata.
#' @return A serializable `sparse_relation` descriptor.
#' @export
sparse_relation <- function(data, from, to, from_col = ".from_id",
                            to_col = ".to_id", weight = NULL,
                            directed = TRUE, metadata = list()) {
  data <- tibble::as_tibble(data)
  from <- .one_relation_string(from, "from")
  to <- .one_relation_string(to, "to")
  from_col <- .one_relation_string(from_col, "from_col")
  to_col <- .one_relation_string(to_col, "to_col")
  if (identical(from_col, to_col)) {
    .relation_abort("from_col and to_col must be distinct.", field = "columns")
  }
  if (!all(c(from_col, to_col) %in% names(data))) {
    .relation_abort("Sparse relation edge columns are absent from data.", field = "data")
  }
  .validate_sparse_scalar_data(data)
  data[[from_col]] <- as.character(data[[from_col]])
  data[[to_col]] <- as.character(data[[to_col]])
  if (anyNA(data[[from_col]]) || any(!nzchar(data[[from_col]])) ||
      anyNA(data[[to_col]]) || any(!nzchar(data[[to_col]]))) {
    .relation_abort("Sparse relation edge IDs must be non-missing and non-empty.", field = "data")
  }
  if (!is.logical(directed) || length(directed) != 1L || is.na(directed)) {
    .relation_abort("directed must be TRUE or FALSE.", field = "directed")
  }
  pairs <- data[c(from_col, to_col)]
  duplicate_pairs <- anyDuplicated(pairs)
  if (!isTRUE(directed) && identical(from, to)) {
    undirected_pairs <- data.frame(
      from = pmin(data[[from_col]], data[[to_col]]),
      to = pmax(data[[from_col]], data[[to_col]]),
      stringsAsFactors = FALSE
    )
    duplicate_pairs <- duplicate_pairs || anyDuplicated(undirected_pairs)
  }
  if (duplicate_pairs) {
    .relation_abort("Sparse relation edges contain duplicate source-target pairs.", field = "data")
  }
  if (!is.null(weight)) {
    weight <- .one_relation_string(weight, "weight")
    if (!weight %in% names(data) || !is.numeric(data[[weight]]) ||
        anyNA(data[[weight]]) || any(!is.finite(data[[weight]]))) {
      .relation_abort("Sparse relation weight must name a finite numeric column.", field = "weight")
    }
  }
  out <- structure(
    list(
      type = "sparse",
      data = data,
      from = from,
      to = to,
      from_col = from_col,
      to_col = to_col,
      weight = weight,
      directed = directed,
      metadata = metadata,
      schema_version = 1L
    ),
    class = c("sparse_relation", "fmri_relation")
  )
  if (.source_contains_runtime_state(out)) {
    .relation_abort("Relations cannot contain runtime state.", field = "runtime_state")
  }
  out
}

#' Construct a relation registry
#'
#' @param relations A named list of `key_relation` or `sparse_relation`
#'   descriptors.
#' @param ... Alternatively, named relation descriptors.
#' @return A named `relation_registry`.
#' @export
relation_registry <- function(relations = list(), ...) {
  dots <- list(...)
  if (inherits(relations, "relation_registry") && !length(dots)) {
    validate_relation_registry(relations)
    return(relations)
  }
  if (length(dots)) {
    if (length(relations)) {
      .relation_abort("Supply relations either as a list or as named arguments, not both.")
    }
    relations <- dots
  }
  if (!is.list(relations)) {
    .relation_abort("relations must be a named list.", field = "relations")
  }
  if (length(relations)) {
    names_value <- names(relations)
    if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
        anyDuplicated(names_value)) {
      .relation_abort("Relation registries must be named with unique, non-empty values.", field = "names")
    }
    valid <- vapply(relations, inherits, logical(1), "fmri_relation")
    if (!all(valid)) {
      .relation_abort(
        "Every registry entry must be a key_relation or sparse_relation.",
        field = "relations",
        relations = names_value[!valid]
      )
    }
  }
  class(relations) <- c("relation_registry", "list")
  validate_relation_registry(relations)
  relations
}

.validate_relation_descriptor <- function(value, name) {
  if (inherits(value, "key_relation")) {
    required <- c(
      "type", "key", "source", "target", "allow_missing", "metadata",
      "schema_version"
    )
    if (!identical(names(value), required) || !identical(value$type, "key") ||
        !identical(value$schema_version, 1L)) {
      .relation_abort(sprintf("Key relation '%s' has an invalid descriptor.", name), relation = name)
    }
    key_relation(
      key = value$key,
      target = value$target,
      source = value$source,
      allow_missing = value$allow_missing,
      metadata = value$metadata
    )
    return(invisible(TRUE))
  }
  if (inherits(value, "sparse_relation")) {
    required <- c(
      "type", "data", "from", "to", "from_col", "to_col", "weight",
      "directed", "metadata", "schema_version"
    )
    if (!identical(names(value), required) || !identical(value$type, "sparse") ||
        !identical(value$schema_version, 1L)) {
      .relation_abort(sprintf("Sparse relation '%s' has an invalid descriptor.", name), relation = name)
    }
    sparse_relation(
      data = value$data,
      from = value$from,
      to = value$to,
      from_col = value$from_col,
      to_col = value$to_col,
      weight = value$weight,
      directed = value$directed,
      metadata = value$metadata
    )
    return(invisible(TRUE))
  }
  .relation_abort(sprintf("Relation '%s' has an unknown descriptor type.", name), relation = name)
}

.normalize_relation_domain <- function(domain, entities) {
  domain <- .one_relation_string(domain, "domain")
  if (domain %in% c("observation", "feature")) return(domain)
  name <- if (startsWith(domain, "entity:")) sub("^entity:", "", domain) else domain
  if (!name %in% entity_names(entities)) {
    .relation_abort(sprintf("Unknown relation domain '%s'.", domain), domain = domain)
  }
  paste0("entity:", name)
}

.relation_domain_value <- function(domain, observations, features, entities) {
  if (identical(domain, "observation")) {
    return(list(ids = axis_ids(observations), data = axis_data(observations)))
  }
  if (identical(domain, "feature")) {
    return(list(ids = axis_ids(features), data = axis_data(features)))
  }
  name <- sub("^entity:", "", domain)
  value <- entity(entities, name)
  list(ids = entity_ids(value), data = entity_data(value))
}

.resolve_key_relation <- function(value, observations, features, entities, name) {
  source <- .normalize_relation_domain(value$source, entities)
  target <- value$target
  if (is.null(target)) {
    candidates <- entity_names(entities)[vapply(entities, function(entity_value) {
      identical(entity_key(entity_value), value$key)
    }, logical(1))]
    if (length(candidates) != 1L) {
      .relation_abort(
        sprintf("Relation '%s' cannot uniquely infer a target entity for key '%s'.", name, value$key),
        relation = name,
        field = "target",
        candidates = candidates
      )
    }
    target <- candidates[[1L]]
  }
  target <- .normalize_relation_domain(target, entities)
  if (!startsWith(target, "entity:")) {
    .relation_abort("Key relations must target an entity domain.", relation = name, field = "target")
  }
  source_value <- .relation_domain_value(source, observations, features, entities)
  if (!value$key %in% names(source_value$data)) {
    .relation_abort(
      sprintf("Relation '%s' source domain has no '%s' column.", name, value$key),
      relation = name,
      field = "key"
    )
  }
  keys <- source_value$data[[value$key]]
  if (is.list(keys) || !is.null(dim(keys))) {
    .relation_abort("Foreign-key columns must contain scalar values.", relation = name, field = "key")
  }
  missing <- is.na(keys)
  if (any(missing) && !isTRUE(value$allow_missing)) {
    .relation_abort(
      sprintf("Relation '%s' contains missing foreign keys.", name),
      relation = name,
      field = "key"
    )
  }
  target_ids <- .relation_domain_value(target, observations, features, entities)$ids
  unknown <- !missing & !as.character(keys) %in% target_ids
  if (any(unknown)) {
    .relation_abort(
      sprintf("Relation '%s' contains unknown target IDs.", name),
      relation = name,
      field = "key",
      values = unique(as.character(keys[unknown]))
    )
  }
  value$source <- source
  value$target <- target
  value
}

.resolve_sparse_relation <- function(value, observations, features, entities, name) {
  from <- .normalize_relation_domain(value$from, entities)
  to <- .normalize_relation_domain(value$to, entities)
  from_ids <- .relation_domain_value(from, observations, features, entities)$ids
  to_ids <- .relation_domain_value(to, observations, features, entities)$ids
  unknown_from <- !value$data[[value$from_col]] %in% from_ids
  unknown_to <- !value$data[[value$to_col]] %in% to_ids
  if (any(unknown_from) || any(unknown_to)) {
    .relation_abort(
      sprintf("Relation '%s' contains unknown source or target IDs.", name),
      relation = name,
      field = "data",
      unknown_from = unique(value$data[[value$from_col]][unknown_from]),
      unknown_to = unique(value$data[[value$to_col]][unknown_to])
    )
  }
  value$from <- from
  value$to <- to
  value
}

.resolve_relation_registry <- function(x, observations, features, entities) {
  x <- relation_registry(x)
  out <- lapply(names(x), function(name) {
    value <- x[[name]]
    if (inherits(value, "key_relation")) {
      .resolve_key_relation(value, observations, features, entities, name)
    } else {
      .resolve_sparse_relation(value, observations, features, entities, name)
    }
  })
  names(out) <- names(x)
  class(out) <- c("relation_registry", "list")
  out
}

#' Validate a relation registry
#'
#' @param x A `relation_registry`.
#' @param observations Optional observation `axis_frame`.
#' @param features Optional feature `axis_frame`.
#' @param entities Optional `entity_registry`.
#' @return Invisibly returns `x`; contextual validation also enforces all
#'   foreign-key and edge identities.
#' @export
validate_relation_registry <- function(x, observations = NULL, features = NULL,
                                       entities = NULL) {
  if (!inherits(x, "relation_registry") || !is.list(x)) {
    .relation_abort("x must be a relation_registry.", field = "class")
  }
  if (length(x)) {
    names_value <- names(x)
    if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
        anyDuplicated(names_value)) {
      .relation_abort("Relation registries must be named with unique, non-empty values.", field = "names")
    }
    valid <- vapply(x, inherits, logical(1), "fmri_relation")
    if (!all(valid)) {
      .relation_abort("Every relation registry entry must be an fmri_relation.", field = "relations")
    }
    for (name in names_value) .validate_relation_descriptor(x[[name]], name)
  }
  context <- c(!is.null(observations), !is.null(features), !is.null(entities))
  if (any(context) && !all(context)) {
    .relation_abort("Contextual validation requires observations, features, and entities together.", field = "context")
  }
  if (all(context)) {
    if (!inherits(observations, "axis_frame") || !identical(observations$axis, "observation") ||
        !inherits(features, "spatial_axis_frame")) {
      .relation_abort("Relation validation received invalid frame axes.", field = "context")
    }
    validate_entity_registry(entities)
    .resolve_relation_registry(x, observations, features, entities)
  }
  if (.source_contains_runtime_state(x)) {
    .relation_abort("Relation registries cannot contain runtime state.", field = "runtime_state")
  }
  invisible(x)
}

#' Access frame relations
#'
#' @param x An `fmri_frame`, view, or relation registry.
#' @param name One registered relation name.
#' @param ... Additional method arguments.
#' @return `relations()` returns the registry; `relation()` returns one
#'   descriptor; `relation_names()` returns registry names.
#' @name relation-accessors
NULL

#' @rdname relation-accessors
#' @export
relations <- function(x, ...) UseMethod("relations")

#' @export
relations.relation_registry <- function(x, ...) x

#' @rdname relation-accessors
#' @export
relation <- function(x, name, ...) UseMethod("relation")

#' @export
relation.relation_registry <- function(x, name, ...) {
  if (!is.character(name) || length(name) != 1L || is.na(name) || !name %in% names(x)) {
    label <- if (length(name)) as.character(name)[[1L]] else ""
    .relation_abort(sprintf("Unknown relation '%s'.", label), relation = name)
  }
  x[[name]]
}

#' @rdname relation-accessors
#' @export
relation_names <- function(x) names(relations(x))

#' Compute a stable relation-registry digest
#'
#' @param x A frame, view, or relation registry.
#' @return A hexadecimal digest over the normalized registry.
#' @export
relation_registry_digest <- function(x) {
  x <- relations(x)
  validate_relation_registry(x)
  .canonical_digest(x)
}

.restrict_relation_registry <- function(x, observation_ids, feature_ids) {
  out <- lapply(x, function(value) {
    if (!inherits(value, "sparse_relation")) return(value)
    keep <- rep(TRUE, nrow(value$data))
    if (identical(value$from, "observation")) {
      keep <- keep & value$data[[value$from_col]] %in% observation_ids
    } else if (identical(value$from, "feature")) {
      keep <- keep & value$data[[value$from_col]] %in% feature_ids
    }
    if (identical(value$to, "observation")) {
      keep <- keep & value$data[[value$to_col]] %in% observation_ids
    } else if (identical(value$to, "feature")) {
      keep <- keep & value$data[[value$to_col]] %in% feature_ids
    }
    value$data <- value$data[keep, , drop = FALSE]
    value
  })
  names(out) <- names(x)
  class(out) <- c("relation_registry", "list")
  out
}

.bind_relation_registries <- function(xs) {
  first <- xs[[1L]]
  if (!all(vapply(xs, function(x) identical(names(x), names(first)), logical(1)))) {
    .relation_abort("Bound frames must have identical relation names.", operation = "bind_observations")
  }
  out <- lapply(names(first), function(name) {
    values <- lapply(xs, `[[`, name)
    prototype <- values[[1L]]
    if (inherits(prototype, "key_relation")) {
      if (!all(vapply(values, identical, logical(1), prototype))) {
        .relation_abort(sprintf("Key relation '%s' differs across bound frames.", name), relation = name)
      }
      return(prototype)
    }
    fields <- c("type", "from", "to", "from_col", "to_col", "weight", "directed", "metadata", "schema_version")
    same_structure <- vapply(values, function(value) {
      identical(value[fields], prototype[fields])
    }, logical(1))
    if (!all(same_structure)) {
      .relation_abort(sprintf("Sparse relation '%s' differs across bound frames.", name), relation = name)
    }
    observation_bound <- "observation" %in% c(prototype$from, prototype$to)
    if (!observation_bound) {
      if (!all(vapply(values, function(value) identical(value$data, prototype$data), logical(1)))) {
        .relation_abort(sprintf("Non-observation relation '%s' differs across bound frames.", name), relation = name)
      }
      return(prototype)
    }
    data <- do.call(rbind, lapply(values, `[[`, "data"))
    sparse_relation(
      data = data,
      from = prototype$from,
      to = prototype$to,
      from_col = prototype$from_col,
      to_col = prototype$to_col,
      weight = prototype$weight,
      directed = prototype$directed,
      metadata = prototype$metadata
    )
  })
  names(out) <- names(first)
  relation_registry(out)
}

#' @export
print.relation_registry <- function(x, ...) {
  cat("<relation_registry>", length(x), "relations\n")
  if (length(x)) cat("  ", paste(names(x), collapse = ", "), "\n", sep = "")
  invisible(x)
}
