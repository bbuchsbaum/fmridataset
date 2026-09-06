.metadata_abort <- function(message, field = NULL, ...) {
  .frame_abort(
    message,
    "fmridataset_error_metadata",
    field = field,
    ...
  )
}

.table_abort <- function(message, field = NULL, ...) {
  .frame_abort(
    message,
    "fmridataset_error_table",
    field = field,
    ...
  )
}

.provenance_abort <- function(message, field = "provenance", ...) {
  .frame_abort(
    message,
    "fmridataset_error_provenance",
    field = field,
    ...
  )
}

.metadata_diagnostic_names <- c(
  "convergence", "converged", "diagnostics", "fitdiagnostics",
  "resultdiagnostics", "loglik", "loglikelihood"
)

.metadata_field_token <- function(x) {
  tolower(gsub("[^[:alnum:]]", "", x))
}

.normalize_alignment_domains <- function(domains) {
  if (is.null(domains)) return(integer())
  if (!is.numeric(domains) || is.null(names(domains)) ||
      anyNA(domains) || any(domains < 0) ||
      any(domains != as.integer(domains)) || any(!nzchar(names(domains)))) {
    .metadata_abort("Alignment domains must be named non-negative sizes.", "domains")
  }
  stats::setNames(as.integer(domains), names(domains))
}

.metadata_alignment_label <- function(n, domains) {
  if (n <= 1L || !length(domains)) return(NULL)
  matches <- names(domains)[domains == n]
  if (!length(matches)) NULL else matches[[1L]]
}

.normalize_unaligned_value <- function(value, path, domains) {
  if (.source_contains_runtime_state(value)) {
    .metadata_abort(
      sprintf("Metadata field '%s' cannot contain runtime state.", path),
      path
    )
  }
  if (is.data.frame(value) || (!is.null(dim(value)) && !inherits(value, "AsIs"))) {
    .metadata_abort(
      sprintf(
        "Metadata field '%s' is tabular or array-valued; use an axis block, assay, relation, or typed table.",
        path
      ),
      path
    )
  }
  if (is.list(value)) {
    if (is.object(value) && !inherits(value, "unaligned_record")) {
      .metadata_abort(
        sprintf(
          "Metadata field '%s' is a typed object; migrate it to an unaligned record, typed table, relation, or provenance record.",
          path
        ),
        path
      )
    }
    .normalize_unaligned_list(value, path, domains)
  } else {
    label <- .metadata_alignment_label(length(value), domains)
    if (!is.null(label)) {
      .metadata_abort(
        sprintf(
          "Metadata field '%s' appears %s-aligned; store aligned values on the declared axis, entity, block, assay, relation, table, or linked frame.",
          path, label
        ),
        path,
        alignment = label,
        size = length(value)
      )
    }
    value
  }
}

.normalize_unaligned_list <- function(value, path, domains) {
    if (!inherits(value, "unaligned_record") && length(value) &&
        (is.null(names(value)) || anyNA(names(value)) ||
         any(!nzchar(names(value))) || anyDuplicated(names(value)))) {
      .metadata_abort(
        sprintf("Metadata record '%s' must have unique non-empty field names.", path),
        path
      )
    }
    fields <- unclass(value)
    if (length(fields)) {
      tokens <- .metadata_field_token(names(fields))
      bad <- tokens %in% .metadata_diagnostic_names
      if (any(bad)) {
        field <- names(fields)[which(bad)[[1L]]]
        .metadata_abort(
          sprintf(
            "Metadata field '%s.%s' is a result diagnostic; store diagnostics as aligned assays, blocks, or typed tables.",
            path, field
          ),
          paste(path, field, sep = ".")
        )
      }
      fields <- lapply(names(fields), function(name) {
        child <- if (nzchar(path)) paste(path, name, sep = ".") else name
        .normalize_unaligned_value(fields[[name]], child, domains)
      })
      names(fields) <- names(unclass(value))
    }
    return(structure(fields, class = c("unaligned_record", "list")))
}

#' Construct a typed unaligned metadata record
#'
#' Container metadata is deliberately unaligned. Array- or table-valued data
#' belong in assays, axis blocks, relations, typed tables, or linked frames.
#' When `domains` are supplied, vector lengths matching an observation,
#' feature, or entity domain are rejected as probable hidden alignment.
#'
#' @param x A named serializable list.
#' @param domains Optional named observation, feature, or entity sizes used to
#'   detect hidden alignment.
#' @return An `unaligned_record`.
#' @export
unaligned_record <- function(x = list(), domains = NULL) {
  if (inherits(x, "unaligned_record")) x <- unclass(x)
  if (!is.list(x)) {
    .metadata_abort("Unaligned metadata must be a named list.", "metadata")
  }
  if (length(x) && (is.null(names(x)) || anyNA(names(x)) ||
                    any(!nzchar(names(x))) || anyDuplicated(names(x)))) {
    .metadata_abort(
      "Unaligned metadata must have unique non-empty field names.",
      "metadata"
    )
  }
  domains <- .normalize_alignment_domains(domains)
  if (length(x)) {
    tokens <- .metadata_field_token(names(x))
    bad <- tokens %in% .metadata_diagnostic_names
    if (any(bad)) {
      .metadata_abort(
        sprintf(
          "Metadata field '%s' is a result diagnostic; store diagnostics as aligned assays, blocks, or typed tables.",
          names(x)[which(bad)[[1L]]]
        ),
        names(x)[which(bad)[[1L]]]
      )
    }
  }
  .normalize_unaligned_value(x, "metadata", domains)
}

#' Validate a typed unaligned metadata record
#'
#' @param x An `unaligned_record`.
#' @param domains Optional named alignment-domain sizes.
#' @return `x`, invisibly.
#' @export
validate_unaligned_record <- function(x, domains = NULL) {
  if (!inherits(x, "unaligned_record")) {
    .metadata_abort("x is not an unaligned_record.", "metadata")
  }
  unaligned_record(x, domains = domains)
  invisible(x)
}

.container_alignment_domains <- function(observations = NULL, features = NULL,
                                         entities = NULL, prefix = NULL) {
  out <- integer()
  label <- function(x) if (is.null(prefix)) x else paste0(prefix, x)
  if (!is.null(observations) && length(observations) > 1L) {
    out[[label("observation")]] <- length(observations)
  }
  if (!is.null(features) && length(features) > 1L) {
    out[[label("feature")]] <- length(features)
  }
  if (!is.null(entities)) {
    for (name in entity_names(entities)) {
      if (length(entities[[name]]) > 1L) {
        out[[label(paste0("entity:", name))]] <- length(entities[[name]])
      }
    }
  }
  out
}

.frame_alignment_domains <- function(observations, features, entities) {
  .container_alignment_domains(observations, features, entities)
}

.collection_alignment_domains <- function(frames) {
  out <- integer()
  for (name in names(frames)) {
    value <- frames[[name]]
    current <- .container_alignment_domains(
      observation_axis(value), feature_axis(value), entities(value),
      prefix = paste0("frame:", name, ":")
    )
    out <- c(out, current)
  }
  out
}

.study_alignment_domains <- function(frames, entities) {
  out <- .container_alignment_domains(entities = entities)
  for (name in names(frames)) {
    members <- if (inherits(frames[[name]], "fmri_collection")) {
      collection_frames(frames[[name]])
    } else {
      stats::setNames(list(frames[[name]]), name)
    }
    for (member_name in names(members)) {
      prefix <- if (inherits(frames[[name]], "fmri_collection")) {
        paste0("representation:", name, ":member:", member_name, ":")
      } else {
        paste0("representation:", name, ":")
      }
      out <- c(out, .container_alignment_domains(
        observation_axis(members[[member_name]]),
        feature_axis(members[[member_name]]),
        prefix = prefix
      ))
    }
  }
  out
}

.normalize_container_metadata <- function(metadata, domains) {
  unaligned_record(metadata, domains = domains)
}

.validate_container_provenance <- function(provenance, owner) {
  if (is.null(provenance)) return(NULL)
  if (!inherits(provenance, "provenance_graph")) {
    .provenance_abort(sprintf(
      "%s provenance must be NULL or a provenance_graph; use as_provenance_graph() for explicit legacy migration.",
      owner
    ))
  }
  tryCatch(
    validate_provenance_graph(provenance),
    error = function(error) {
      .provenance_abort(
        paste0("Invalid provenance_graph: ", conditionMessage(error)),
        parent = error
      )
    }
  )
  provenance
}

.validate_scalar_table_data <- function(data, owner) {
  if (!is.data.frame(data)) {
    .table_abort(sprintf("%s data must be a data frame.", owner), "data")
  }
  data <- tibble::as_tibble(data)
  non_scalar <- vapply(data, function(value) {
    is.list(value) || !is.null(dim(value)) || length(value) != nrow(data)
  }, logical(1))
  if (any(non_scalar)) {
    .table_abort(
      sprintf("%s columns must contain scalar values.", owner),
      "data",
      columns = names(data)[non_scalar]
    )
  }
  data
}

#' Construct a typed auxiliary table
#'
#' @param data Scalar tabular data.
#' @param key Optional stable unique-key column.
#' @param role Stable table role such as `"files"`, `"contrasts"`, or
#'   `"transforms"`.
#' @param metadata Unaligned table-level metadata.
#' @return An `fmri_auxiliary_table`.
#' @export
auxiliary_table <- function(data, key = NULL, role = "auxiliary",
                            metadata = list()) {
  data <- .validate_scalar_table_data(data, "Auxiliary table")
  if (!is.null(key)) {
    if (!is.character(key) || length(key) != 1L || is.na(key) ||
        !nzchar(key) || !key %in% names(data)) {
      .table_abort("Auxiliary table key must name one scalar column.", "key")
    }
    ids <- as.character(data[[key]])
    if (anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
      .table_abort(
        "Auxiliary table keys must be unique, non-missing, and non-empty.",
        key
      )
    }
    data[[key]] <- ids
  }
  if (!is.character(role) || length(role) != 1L || is.na(role) || !nzchar(role)) {
    .table_abort("Auxiliary table role must be one non-empty string.", "role")
  }
  metadata <- unaligned_record(metadata)
  structure(
    list(
      data = data, key = key, role = role, metadata = metadata,
      schema_version = 1L
    ),
    class = "fmri_auxiliary_table"
  )
}

#' Validate and inspect a typed auxiliary table
#'
#' @param x An `fmri_auxiliary_table`.
#' @return `validate_auxiliary_table()` returns `x` invisibly; other functions
#'   return table data, key, or role.
#' @name auxiliary-table
NULL

#' @rdname auxiliary-table
#' @export
validate_auxiliary_table <- function(x) {
  required <- c("data", "key", "role", "metadata", "schema_version")
  if (!inherits(x, "fmri_auxiliary_table") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .table_abort("x is not a valid fmri_auxiliary_table.")
  }
  auxiliary_table(x$data, x$key, x$role, x$metadata)
  invisible(x)
}

#' @rdname auxiliary-table
#' @export
table_data <- function(x) {
  if (inherits(x, "fmri_event_table")) return(event_data(x))
  validate_auxiliary_table(x)
  x$data
}

#' @rdname auxiliary-table
#' @export
table_key <- function(x) {
  if (inherits(x, "fmri_event_table")) return(event_key(x))
  validate_auxiliary_table(x)
  x$key
}

#' @rdname auxiliary-table
#' @export
table_role <- function(x) {
  if (inherits(x, "fmri_event_table")) return("events")
  validate_auxiliary_table(x)
  x$role
}

.validate_table_registry <- function(tables, owner = "Frame") {
  if (!is.list(tables)) {
    .table_abort(sprintf("%s tables must be a named list.", owner), "tables")
  }
  if (!length(tables)) return(tables)
  ids <- names(tables)
  if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
    .table_abort(
      sprintf("%s tables require unique non-empty names.", owner),
      "tables"
    )
  }
  for (name in ids) {
    value <- tables[[name]]
    tryCatch(
      {
        if (inherits(value, "fmri_event_table")) validate_event_table(value)
        else if (inherits(value, "fmri_auxiliary_table")) validate_auxiliary_table(value)
        else .table_abort(
          sprintf(
            "%s table '%s' must be a typed table created by event_table() or auxiliary_table().",
            owner, name
          ),
          paste0("tables.", name)
        )
      },
      fmridataset_error_table = function(error) stop(error),
      error = function(error) {
        .table_abort(
          sprintf("Invalid %s table '%s': %s", tolower(owner), name,
                  conditionMessage(error)),
          paste0("tables.", name), parent = error
        )
      }
    )
  }
  tables
}
