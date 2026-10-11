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

# Container metadata is unaligned by definition. A vector whose length happens
# to equal an axis or entity size (a voxel size of c(2, 2, 2) on a
# three-observation frame) is not evidence of alignment, so field lengths are
# never compared with axis sizes; only tabular, array-valued, typed, and
# runtime values are refused.
.normalize_unaligned_value <- function(value, path) {
  .assert_no_runtime_state(
    value, .metadata_abort,
    sprintf("Metadata field '%s' cannot contain runtime state.", path),
    field = path
  )
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
    .normalize_unaligned_list(value, path)
  } else {
    value
  }
}

.normalize_unaligned_list <- function(value, path) {
  if (!inherits(value, "unaligned_record")) {
    .assert_unique_names(
      value, .metadata_abort,
      sprintf("Metadata record '%s' must have unique non-empty field names.", path),
      field = path
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
      .normalize_unaligned_value(fields[[name]], child)
    })
    names(fields) <- names(unclass(value))
  }
  return(structure(fields, class = c("unaligned_record", "list")))
}

#' Construct a typed unaligned metadata record
#'
#' Container metadata is deliberately unaligned. Array- or table-valued data
#' belong in assays, axis blocks, relations, typed tables, or linked frames.
#' A vector field is an unaligned value whatever its length; a length that
#' coincides with an axis or entity size is not treated as alignment.
#'
#' @param x A named serializable list.
#' @return An `unaligned_record`.
#' @examples
#' rec <- unaligned_record(list(task = "rest", tr = 2))
#' rec$task
#' @export
unaligned_record <- function(x = list()) {
  if (inherits(x, "unaligned_record")) x <- unclass(x)
  if (!is.list(x)) {
    .metadata_abort("Unaligned metadata must be a named list.", "metadata")
  }
  .assert_unique_names(
    x, .metadata_abort,
    "Unaligned metadata must have unique non-empty field names.",
    field = "metadata"
  )
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
  .normalize_unaligned_value(x, "metadata")
}

#' Validate a typed unaligned metadata record
#'
#' @param x An `unaligned_record`.
#' @return `x`, invisibly.
#' @examples
#' rec <- unaligned_record(list(task = "rest", tr = 2))
#' validate_unaligned_record(rec)
#' @export
validate_unaligned_record <- function(x) {
  if (!inherits(x, "unaligned_record")) {
    .metadata_abort("x is not an unaligned_record.", "metadata")
  }
  unaligned_record(x)
  invisible(x)
}

.normalize_container_metadata <- function(metadata) {
  unaligned_record(metadata)
}

.validate_container_provenance <- function(provenance, owner) {
  if (is.null(provenance)) {
    return(NULL)
  }
  if (!inherits(provenance, "provenance_graph")) {
    .provenance_abort(sprintf(
      "%s provenance must be NULL or a provenance_graph; build one with provenance_graph() and provenance_record().",
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
  .assert_scalar_columns(
    data, .table_abort, sprintf("%s columns must contain scalar values.", owner)
  )
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
#' @examples
#' at <- auxiliary_table(
#'   data.frame(contrast = c("A-B", "B-A"), stat = c(2.1, -2.1)),
#'   key = "contrast", role = "contrasts"
#' )
#' table_data(at)
#' @export
auxiliary_table <- function(data, key = NULL, role = "auxiliary",
                            metadata = list()) {
  data <- .validate_scalar_table_data(data, "Auxiliary table")
  if (!is.null(key)) {
    if (!.is_one_string(key) || !key %in% names(data)) {
      .table_abort("Auxiliary table key must name one scalar column.", "key")
    }
    data[[key]] <- .assert_stable_keys(
      as.character(data[[key]]), .table_abort,
      field = key,
      message = "Auxiliary table keys must be unique, non-missing, and non-empty."
    )
  }
  .assert_one_string(
    role, "role", .table_abort,
    message = "Auxiliary table role must be one non-empty string."
  )
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
#' @examples
#' at <- auxiliary_table(
#'   data.frame(contrast = c("A-B", "B-A"), stat = c(2.1, -2.1)),
#'   key = "contrast", role = "contrasts"
#' )
#' validate_auxiliary_table(at)
#' table_key(at)
#' table_role(at)
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
  if (inherits(x, "fmri_event_table")) {
    return(event_data(x))
  }
  validate_auxiliary_table(x)
  x$data
}

#' @rdname auxiliary-table
#' @export
table_key <- function(x) {
  if (inherits(x, "fmri_event_table")) {
    return(event_key(x))
  }
  validate_auxiliary_table(x)
  x$key
}

#' @rdname auxiliary-table
#' @export
table_role <- function(x) {
  if (inherits(x, "fmri_event_table")) {
    return("events")
  }
  validate_auxiliary_table(x)
  x$role
}

.validate_table_registry <- function(tables, owner = "Frame") {
  if (!is.list(tables)) {
    .table_abort(sprintf("%s tables must be a named list.", owner), "tables")
  }
  if (!length(tables)) {
    return(tables)
  }
  .assert_unique_names(
    tables, .table_abort,
    sprintf("%s tables require unique non-empty names.", owner),
    field = "tables"
  )
  for (name in names(tables)) {
    value <- tables[[name]]
    tryCatch(
      {
        if (inherits(value, "fmri_event_table")) {
          validate_event_table(value)
        } else if (inherits(value, "fmri_auxiliary_table")) {
          validate_auxiliary_table(value)
        } else {
          .table_abort(
            sprintf(
              "%s table '%s' must be a typed table created by event_table() or auxiliary_table().",
              owner, name
            ),
            paste0("tables.", name)
          )
        }
      },
      fmridataset_error_table = function(error) stop(error),
      error = function(error) {
        .table_abort(
          sprintf(
            "Invalid %s table '%s': %s", tolower(owner), name,
            conditionMessage(error)
          ),
          paste0("tables.", name),
          parent = error
        )
      }
    )
  }
  tables
}
