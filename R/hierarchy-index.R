.hierarchy_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_hierarchy", ...)
}

.validate_hierarchy_levels <- function(levels, entities_value) {
  if (!is.character(levels) || !length(levels) || anyNA(levels) ||
      any(!nzchar(levels))) {
    .hierarchy_abort("Hierarchy levels must contain at least one non-empty entity name.")
  }
  if (anyDuplicated(levels)) {
    .hierarchy_abort("Hierarchy levels must be unique.", levels = levels)
  }
  unknown <- setdiff(levels, entity_names(entities_value))
  if (length(unknown)) {
    .hierarchy_abort(
      sprintf("Unknown hierarchy level '%s'.", unknown[[1L]]),
      levels = unknown
    )
  }
  levels
}

.validate_hierarchy_relation_selection <- function(selection, levels) {
  if (is.null(selection)) return(NULL)
  if (!is.character(selection) || length(selection) != length(levels) ||
      is.null(names(selection)) || anyNA(selection) || any(!nzchar(selection)) ||
      anyNA(names(selection)) || any(!nzchar(names(selection))) ||
      anyDuplicated(names(selection)) || !setequal(names(selection), levels)) {
    .hierarchy_abort(
      "Explicit hierarchy relations must name every hierarchy level exactly once.",
      levels = levels
    )
  }
  if (anyDuplicated(selection)) {
    .hierarchy_abort("Each hierarchy edge must use a distinct key relation.")
  }
  selection
}

.hierarchy_edge_domains <- function(levels) {
  reversed <- rev(levels)
  parent_sources <- if (length(reversed) > 1L) {
    paste0("entity:", reversed[seq_len(length(reversed) - 1L)])
  } else {
    character()
  }
  sources <- c("observation", parent_sources)
  targets <- paste0("entity:", reversed)
  data.frame(
    level = reversed,
    source = sources,
    target = targets,
    stringsAsFactors = FALSE
  )
}

.select_hierarchy_relation <- function(registry, edge, selection) {
  candidates <- relation_names(registry)[vapply(registry, function(value) {
    inherits(value, "key_relation") &&
      identical(value$source, edge$source) &&
      identical(value$target, edge$target)
  }, logical(1))]

  if (!is.null(selection)) {
    selected_name <- unname(selection[[edge$level]])
    if (!selected_name %in% relation_names(registry)) {
      .hierarchy_abort(
        sprintf("Unknown hierarchy relation '%s'.", selected_name),
        relation = selected_name,
        level = edge$level
      )
    }
    selected <- registry[[selected_name]]
    if (!inherits(selected, "key_relation") ||
        !identical(selected$source, edge$source) ||
        !identical(selected$target, edge$target)) {
      .hierarchy_abort(
        sprintf(
          "Hierarchy relation '%s' does not connect %s to %s.",
          selected_name, edge$source, edge$target
        ),
        relation = selected_name,
        source = edge$source,
        target = edge$target
      )
    }
    return(selected_name)
  }

  if (!length(candidates)) {
    .hierarchy_abort(
      sprintf(
        "No key relation forms the strict containment path from %s to %s.",
        edge$source, edge$target
      ),
      source = edge$source,
      target = edge$target
    )
  }
  if (length(candidates) > 1L) {
    .hierarchy_abort(
      sprintf(
        "The hierarchy edge from %s to %s has multiple key relations; select one explicitly.",
        edge$source, edge$target
      ),
      source = edge$source,
      target = edge$target,
      relations = candidates
    )
  }
  candidates[[1L]]
}

.hierarchy_domain <- function(x, domain) {
  if (identical(domain, "observation")) {
    return(list(ids = observation_ids(x), data = observations(x)))
  }
  name <- sub("^entity:", "", domain)
  value <- entity(x, name)
  list(ids = entity_ids(value), data = entity_data(value))
}

.assert_hierarchy_index <- function(x) {
  required <- c(
    "observation_ids", "levels", "ids", "groups", "relations", "complete",
    "entity_digest", "relation_digest", "schema_version"
  )
  if (!inherits(x, "fmri_hierarchy_index") || !identical(names(x), required) ||
      !identical(x$schema_version, 1L)) {
    .hierarchy_abort("x is not a valid fmri_hierarchy_index.")
  }
  expected_names <- c(".obs_id", x$levels)
  valid_ids <- is.character(x$observation_ids) && !anyNA(x$observation_ids) &&
    !any(!nzchar(x$observation_ids)) && !anyDuplicated(x$observation_ids)
  valid_levels <- is.character(x$levels) && length(x$levels) > 0L &&
    !anyNA(x$levels) && !any(!nzchar(x$levels)) && !anyDuplicated(x$levels)
  valid_tables <- tibble::is_tibble(x$ids) && tibble::is_tibble(x$groups) &&
    identical(names(x$ids), expected_names) &&
    identical(names(x$groups), expected_names) &&
    identical(x$ids$.obs_id, x$observation_ids) &&
    identical(x$groups$.obs_id, x$observation_ids)
  valid_group_columns <- valid_levels && valid_tables && all(vapply(
    x$groups[x$levels],
    function(value) is.integer(value) && all(is.na(value) | value >= 1L),
    logical(1)
  ))
  valid_id_columns <- valid_levels && valid_tables && all(vapply(
    x$ids[x$levels],
    function(value) is.character(value) && all(is.na(value) | nzchar(value)),
    logical(1)
  ))
  valid_relations <- is.character(x$relations) &&
    identical(names(x$relations), x$levels) && !anyNA(x$relations) &&
    !any(!nzchar(x$relations)) && !anyDuplicated(x$relations)
  valid_complete <- valid_levels && valid_tables && is.logical(x$complete) &&
    !anyNA(x$complete) &&
    identical(x$complete, stats::complete.cases(x$ids[x$levels]))
  valid_digests <- is.character(x$entity_digest) &&
    length(x$entity_digest) == 1L && !is.na(x$entity_digest) &&
    grepl("^[0-9a-f]{64}$", x$entity_digest) &&
    is.character(x$relation_digest) && length(x$relation_digest) == 1L &&
    !is.na(x$relation_digest) &&
    grepl("^[0-9a-f]{64}$", x$relation_digest)
  if (!valid_ids || !valid_levels || !valid_tables || !valid_group_columns ||
      !valid_id_columns || !valid_relations || !valid_complete ||
      !valid_digests) {
    .hierarchy_abort("x is not a valid fmri_hierarchy_index.")
  }
  invisible(x)
}

#' Derive stable observation hierarchy indices
#'
#' A hierarchy index is an immutable, assay-free cache derived from validated
#' key relations. `levels` are supplied root-to-leaf. Each adjacent edge must
#' form a strict chain from observations to the deepest entity and then through
#' its parents. Crossed relations are therefore never mistaken for containment.
#'
#' Integer group codes use entity-registry order, so they remain stable when a
#' frame is filtered or reordered.
#'
#' @param x An `fmri_frame` or `fmri_view`.
#' @param levels Unique entity names in root-to-leaf order.
#' @param relations Optional named character vector mapping every level to the
#'   key-relation name used for its incoming edge. This is required when an edge
#'   is ambiguous.
#' @return An `fmri_hierarchy_index`.
#' @export
hierarchy_index <- function(x, levels, relations = NULL) {
  if (!inherits(x, "fmri_frame")) {
    .hierarchy_abort("x must be an fmri_frame or fmri_view.")
  }
  entity_values <- entities(x)
  levels <- .validate_hierarchy_levels(levels, entity_values)
  selection <- .validate_hierarchy_relation_selection(relations, levels)
  registry <- relations(x)
  edges <- .hierarchy_edge_domains(levels)
  selected_names <- vapply(seq_len(nrow(edges)), function(i) {
    .select_hierarchy_relation(registry, edges[i, , drop = FALSE], selection)
  }, character(1))

  current_ids <- observation_ids(x)
  ids_by_level <- stats::setNames(vector("list", length(levels)), levels)
  relation_by_level <- stats::setNames(rep(NA_character_, length(levels)), levels)
  for (i in seq_len(nrow(edges))) {
    edge <- edges[i, , drop = FALSE]
    relation_name <- selected_names[[i]]
    descriptor <- registry[[relation_name]]
    domain <- .hierarchy_domain(x, edge$source)
    positions <- match(current_ids, domain$ids)
    parent_ids <- rep(NA_character_, length(current_ids))
    present <- !is.na(positions)
    if (any(present)) {
      parent_ids[present] <- as.character(
        domain$data[[descriptor$key]][positions[present]]
      )
    }
    ids_by_level[[edge$level]] <- parent_ids
    relation_by_level[[edge$level]] <- relation_name
    current_ids <- parent_ids
  }

  id_table <- tibble::tibble(.obs_id = observation_ids(x))
  group_table <- tibble::tibble(.obs_id = observation_ids(x))
  for (level in levels) {
    id_table[[level]] <- ids_by_level[[level]]
    group_table[[level]] <- as.integer(match(
      ids_by_level[[level]],
      entity_ids(entity_values[[level]])
    ))
  }
  complete <- stats::complete.cases(id_table[levels])

  out <- structure(
    list(
      observation_ids = observation_ids(x),
      levels = levels,
      ids = id_table,
      groups = group_table,
      relations = relation_by_level,
      complete = complete,
      entity_digest = .canonical_digest(entity_values[levels]),
      relation_digest = .canonical_digest(
        unclass(registry[unname(relation_by_level)])
      ),
      schema_version = 1L
    ),
    class = "fmri_hierarchy_index"
  )
  if (.source_contains_runtime_state(out)) {
    .hierarchy_abort("Hierarchy indices cannot contain runtime state.")
  }
  out
}

#' Access derived hierarchy index data
#'
#' @param x An `fmri_hierarchy_index`.
#' @return `hierarchy_ids()` returns stable entity IDs; `hierarchy_groups()`
#'   returns stable integer grouping codes; `hierarchy_levels()` and
#'   `hierarchy_relations()` return named character vectors;
#'   `hierarchy_complete()` returns a logical vector.
#' @name hierarchy-accessors
NULL

#' @rdname hierarchy-accessors
#' @export
hierarchy_ids <- function(x) {
  .assert_hierarchy_index(x)
  x$ids
}

#' @rdname hierarchy-accessors
#' @export
hierarchy_groups <- function(x) {
  .assert_hierarchy_index(x)
  x$groups
}

#' @rdname hierarchy-accessors
#' @export
hierarchy_levels <- function(x) {
  .assert_hierarchy_index(x)
  x$levels
}

#' @rdname hierarchy-accessors
#' @export
hierarchy_relations <- function(x) {
  .assert_hierarchy_index(x)
  x$relations
}

#' @rdname hierarchy-accessors
#' @export
hierarchy_complete <- function(x) {
  .assert_hierarchy_index(x)
  x$complete
}

#' Compute the deterministic digest of a hierarchy index
#'
#' @param x An `fmri_hierarchy_index`.
#' @return A SHA-256 digest.
#' @export
hierarchy_digest <- function(x) {
  .assert_hierarchy_index(x)
  .canonical_digest(unclass(x))
}

#' @export
print.fmri_hierarchy_index <- function(x, ...) {
  .assert_hierarchy_index(x)
  cat("<fmri_hierarchy_index>", length(x$observation_ids), "observations\n")
  cat("  levels:", paste(x$levels, collapse = " > "), "\n")
  cat("  complete:", sum(x$complete), "/", length(x$complete), "\n")
  invisible(x)
}
