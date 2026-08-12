.feature_map_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_feature_map", ...)
}

.one_map_string <- function(x, field) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    .feature_map_abort(sprintf("%s must be one non-empty string.", field),
                       field = field)
  }
  x
}

.validate_serializable_list <- function(x, field) {
  if (!is.list(x) || .source_contains_runtime_state(x)) {
    .feature_map_abort(
      sprintf("%s must be a serializable list.", field),
      field = field
    )
  }
  x
}

#' Describe an explicit transformation between feature spaces
#'
#' A feature map owns a target-by-source linear operator and the complete
#' spatial identity of both axes. Equal dimensions are never treated as
#' evidence of spatial compatibility. Statistical execution plans and
#' covariance models remain the responsibility of packages such as
#' `fmrigds`.
#'
#' @param from Source `feature_space`.
#' @param to Target `feature_space`.
#' @param operator Target-by-source matrix, sparse `Matrix`, or serializable
#'   two-dimensional `array_source`.
#' @param map_type Stable map-family label.
#' @param traits Named serializable semantic traits.
#' @param provenance Serializable derivation metadata for the map itself.
#' @param metadata Additional serializable metadata.
#' @return A serializable `feature_map` descriptor.
#' @export
feature_map <- function(from, to, operator, map_type = "linear",
                        traits = list(linear = TRUE), provenance = list(),
                        metadata = list()) {
  if (!inherits(from, "feature_space") || !inherits(to, "feature_space")) {
    .feature_map_abort("from and to must both be feature spaces.",
                       field = "space")
  }
  map_type <- .one_map_string(map_type, "map_type")
  traits <- .validate_serializable_list(traits, "traits")
  if (length(traits) &&
      (is.null(names(traits)) || anyNA(names(traits)) ||
       any(!nzchar(names(traits))) || anyDuplicated(names(traits)))) {
    .feature_map_abort(
      "traits must be named with unique non-empty values.", field = "traits"
    )
  }
  provenance <- .validate_serializable_list(provenance, "provenance")
  metadata <- .validate_serializable_list(metadata, "metadata")
  operator <- tryCatch(
    .validate_linear_operator(
      operator, c(n_features(to), n_features(from)), "feature-map operator"
    ),
    error = function(error) {
      .feature_map_abort(
        conditionMessage(error), field = "operator",
        expected = c(n_features(to), n_features(from)),
        parent = error
      )
    }
  )
  out <- structure(
    list(
      from = from,
      to = to,
      operator = operator,
      map_type = map_type,
      traits = traits,
      provenance = provenance,
      metadata = metadata,
      schema_version = 1L
    ),
    class = "feature_map"
  )
  if (.source_contains_runtime_state(out)) {
    .feature_map_abort("Feature maps must be serializable.",
                       field = "runtime_state")
  }
  out
}

#' Validate and inspect feature maps
#'
#' @param x A `feature_map`.
#' @return `validate_feature_map()` returns `x` invisibly. The accessors return
#'   the source space, target space, linear operator, or deterministic digest.
#' @name feature-map-accessors
NULL

#' @rdname feature-map-accessors
#' @export
validate_feature_map <- function(x) {
  required <- c(
    "from", "to", "operator", "map_type", "traits", "provenance",
    "metadata", "schema_version"
  )
  if (!inherits(x, "feature_map") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .feature_map_abort("x is not a valid feature_map descriptor.")
  }
  feature_map(
    from = x$from, to = x$to, operator = x$operator,
    map_type = x$map_type, traits = x$traits,
    provenance = x$provenance, metadata = x$metadata
  )
  invisible(x)
}

#' @rdname feature-map-accessors
#' @export
feature_map_source_space <- function(x) {
  validate_feature_map(x)
  x$from
}

#' @rdname feature-map-accessors
#' @export
feature_map_target_space <- function(x) {
  validate_feature_map(x)
  x$to
}

#' @rdname feature-map-accessors
#' @export
feature_map_operator <- function(x) {
  validate_feature_map(x)
  x$operator
}

#' @rdname feature-map-accessors
#' @export
feature_map_digest <- function(x) {
  validate_feature_map(x)
  .canonical_digest(list(
    type = "feature_map",
    schema_version = x$schema_version,
    from = space_digest(x$from),
    from_ids = feature_ids(x$from),
    to = space_digest(x$to),
    to_ids = feature_ids(x$to),
    operator = .linear_operator_digest(x$operator),
    map_type = x$map_type,
    traits = x$traits,
    provenance = x$provenance
  ))
}

#' @export
print.feature_map <- function(x, ...) {
  validate_feature_map(x)
  cat("<feature_map>", n_features(x$from), "source features ->",
      n_features(x$to), "target features\n")
  cat("  type:", x$map_type, "\n")
  cat("  digest:", substr(feature_map_digest(x), 1L, 12L), "\n")
  invisible(x)
}

#' Derive the canonical map owned by a parent-linked target space
#'
#' Parcel spaces contribute their aggregation operator and basis spaces their
#' analysis operator. Other transformations require an explicit
#' `feature_map()`.
#'
#' @param target A parent-linked `parcel_space` or `basis_space`.
#' @return A `feature_map` from `parent_space(target)` to `target`.
#' @export
feature_map_from_target <- function(target) {
  if (inherits(target, "parcel_space")) {
    return(feature_map(
      from = parent_space(target), to = target,
      operator = parcel_aggregation(target), map_type = "parcel_aggregation",
      traits = list(
        linear = TRUE,
        aggregation = target$aggregation,
        preserves_constant = identical(target$aggregation, "mean")
      ),
      provenance = list(target_space = space_digest(target))
    ))
  }
  if (inherits(target, "basis_space")) {
    return(feature_map(
      from = parent_space(target), to = target,
      operator = basis_analysis(target), map_type = target$basis_type,
      traits = list(linear = TRUE, representational = TRUE),
      provenance = c(
        list(target_space = space_digest(target)),
        target$provenance
      )
    ))
  }
  .feature_map_abort(
    "target has no canonical parent-to-target feature map; supply map explicitly.",
    field = "target"
  )
}

.feature_map_operator_rows <- function(x, index) {
  operator <- x$operator
  if (inherits(operator, "array_source")) {
    return(source_read(
      operator,
      observations = index,
      features = seq_len(source_shape(operator)[2L])
    ))
  }
  operator[index, , drop = FALSE]
}

.operator_contributing_columns <- function(operator) {
  if (!nrow(operator) || !ncol(operator)) return(integer())
  if (methods::is(operator, "sparseMatrix")) {
    return(which(Matrix::colSums(operator != 0) > 0))
  }
  which(colSums(operator != 0) > 0)
}

#' Construct a lazy source transformed through a feature map
#'
#' @param source Observation-by-source-feature `array_source`.
#' @param map A compatible `feature_map`.
#' @param rule Transformation rule. `"linear"` maps ordinary values;
#'   `"independent_variance"` maps diagonal variances with squared weights.
#' @return A serializable `feature_mapped_source`.
#' @export
feature_mapped_source <- function(source, map,
                                  rule = c("linear", "independent_variance")) {
  source <- as_array_source(source)
  validate_array_source(source)
  validate_feature_map(map)
  rule <- match.arg(rule)
  if (source_shape(source)[2L] != n_features(map$from)) {
    .feature_map_abort(
      "Source columns do not match the feature-map source space.",
      field = "source_shape"
    )
  }
  target_chunks <- if (inherits(map$operator, "array_source")) {
    source_chunks(map$operator)[1L]
  } else {
    min(4096L, max(1L, n_features(map$to)))
  }
  dtype <- if (grepl("^complex", source_dtype(source))) {
    "complex128"
  } else {
    "float64"
  }
  out <- structure(
    list(
      source = source,
      map = map,
      rule = rule,
      shape = c(source_shape(source)[1L], n_features(map$to)),
      dtype = dtype,
      chunks = pmin(
        c(source_chunks(source)[1L], target_chunks),
        pmax(1L, c(source_shape(source)[1L], n_features(map$to)))
      ),
      schema_version = 1L
    ),
    class = c("feature_mapped_source", "array_source")
  )
  validate_array_source(out)
  out
}

#' @export
source_shape.feature_mapped_source <- function(x, ...) as.integer(x$shape)
#' @export
source_dtype.feature_mapped_source <- function(x, ...) x$dtype
#' @export
source_chunks.feature_mapped_source <- function(x, ...) as.integer(x$chunks)
#' @export
source_capabilities.feature_mapped_source <- function(x, ...) {
  c("row_slice", "column_slice", "block_slice", "serializable")
}
#' @export
source_fingerprint.feature_mapped_source <- function(x, ...) {
  .canonical_digest(list(
    type = "feature_mapped_source",
    schema_version = x$schema_version,
    source = source_fingerprint(x$source),
    map = feature_map_digest(x$map),
    rule = x$rule
  ))
}
#' @export
source_open.feature_mapped_source <- function(x, ...) {
  structure(
    list(source = x),
    class = c("feature_mapped_source_handle", "array_source_handle")
  )
}
#' @export
source_read.feature_mapped_source <- function(x, observations = NULL,
                                               features = NULL, ...) {
  observations <- .normalize_source_index(observations, x$shape[1L])
  features <- .normalize_source_index(features, x$shape[2L])
  if (!length(observations) || !length(features)) {
    return(matrix(
      vector(mode = if (grepl("^complex", x$dtype)) "complex" else "numeric",
             length = length(observations) * length(features)),
      nrow = length(observations), ncol = length(features)
    ))
  }
  operator <- .feature_map_operator_rows(x$map, features)
  contributing <- .operator_contributing_columns(operator)
  if (!length(contributing)) {
    return(matrix(0, nrow = length(observations), ncol = length(features)))
  }
  values <- source_read(
    x$source, observations = observations, features = contributing, ...
  )
  operator <- operator[, contributing, drop = FALSE]
  if (identical(x$rule, "independent_variance")) operator <- operator^2
  unname(as.matrix(values %*% Matrix::t(operator)))
}
#' @export
source_read_native.feature_mapped_source <- function(x, observations = NULL, ...) {
  .frame_abort(
    "feature_mapped_source has no native spatial read path.",
    "fmridataset_error_backend_io", operation = "native_read"
  )
}
#' @export
source_close.feature_mapped_source <- function(x, ...) invisible(TRUE)

#' Create a content-addressed provenance record
#'
#' @param operation Stable operation name.
#' @param parents IDs of direct parent records.
#' @param inputs,parameters,outputs,software,metadata Serializable record data.
#' @return A `provenance_record`.
#' @export
provenance_record <- function(operation, parents = character(), inputs = list(),
                              parameters = list(), outputs = list(),
                              software = list(package = "fmridataset"),
                              metadata = list()) {
  operation <- .one_map_string(operation, "operation")
  if (!is.character(parents) || anyNA(parents) || any(!nzchar(parents)) ||
      anyDuplicated(parents)) {
    .feature_map_abort("parents must contain unique non-empty record IDs.",
                       field = "parents")
  }
  fields <- list(
    operation = operation,
    parents = parents,
    inputs = .validate_serializable_list(inputs, "inputs"),
    parameters = .validate_serializable_list(parameters, "parameters"),
    outputs = .validate_serializable_list(outputs, "outputs"),
    software = .validate_serializable_list(software, "software"),
    metadata = .validate_serializable_list(metadata, "metadata"),
    schema_version = 1L
  )
  structure(c(list(id = .canonical_digest(fields)), fields),
            class = "provenance_record")
}

.validate_provenance_record_shape <- function(x) {
  required <- c(
    "id", "operation", "parents", "inputs", "parameters", "outputs",
    "software", "metadata", "schema_version"
  )
  if (!inherits(x, "provenance_record") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .feature_map_abort("Invalid provenance_record descriptor.",
                       field = "record")
  }
  invisible(x)
}

.provenance_is_acyclic <- function(records) {
  if (!length(records)) return(TRUE)
  state <- setNames(integer(length(records)), names(records))
  visit <- function(id) {
    if (state[[id]] == 1L) return(FALSE)
    if (state[[id]] == 2L) return(TRUE)
    state[[id]] <<- 1L
    parents <- records[[id]]$parents
    for (parent in parents[parents %in% names(records)]) {
      if (!visit(parent)) return(FALSE)
    }
    state[[id]] <<- 2L
    TRUE
  }
  all(vapply(names(records), visit, logical(1)))
}

#' Construct and inspect an immutable provenance graph
#'
#' @param ... `provenance_record` objects, or one list of records.
#' @param x A `provenance_graph`.
#' @param records One or more records appended to `x`.
#' @return A validated `provenance_graph`, its records, tips, or digest.
#' @name provenance-graph
NULL

#' @rdname provenance-graph
#' @export
provenance_graph <- function(...) {
  values <- list(...)
  if (length(values) == 1L && inherits(values[[1L]], "provenance_graph")) {
    validate_provenance_graph(values[[1L]])
    return(values[[1L]])
  }
  if (length(values) == 1L && is.list(values[[1L]]) &&
      !inherits(values[[1L]], "provenance_record")) {
    values <- values[[1L]]
  }
  if (length(values)) {
    invisible(lapply(values, .validate_provenance_record_shape))
    ids <- vapply(values, `[[`, character(1), "id")
    if (anyDuplicated(ids)) {
      .feature_map_abort("Provenance record IDs must be unique.", field = "id")
    }
    names(values) <- ids
    unknown <- setdiff(unique(unlist(lapply(values, `[[`, "parents"))), ids)
    if (length(unknown)) {
      .feature_map_abort("Provenance parents must exist in the graph.",
                         field = "parents", unknown = unknown)
    }
    if (!.provenance_is_acyclic(values)) {
      .feature_map_abort("Provenance graph must be acyclic.", field = "parents")
    }
    for (value in values) {
      expected <- provenance_record(
        value$operation, value$parents, value$inputs, value$parameters,
        value$outputs, value$software, value$metadata
      )$id
      if (!identical(value$id, expected)) {
        .feature_map_abort("Provenance record ID does not match its content.",
                           field = "id")
      }
    }
  }
  structure(list(records = values, schema_version = 1L),
            class = "provenance_graph")
}

#' @rdname provenance-graph
#' @export
validate_provenance_graph <- function(x) {
  required <- c("records", "schema_version")
  if (!inherits(x, "provenance_graph") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .feature_map_abort("x is not a valid provenance_graph.")
  }
  provenance_graph(x$records)
  invisible(x)
}

#' @rdname provenance-graph
#' @export
provenance_records <- function(x) {
  validate_provenance_graph(x)
  x$records
}

#' @rdname provenance-graph
#' @export
provenance_tips <- function(x) {
  records <- provenance_records(x)
  if (!length(records)) return(character())
  parents <- unique(unlist(lapply(records, `[[`, "parents")))
  setdiff(names(records), parents)
}

#' @rdname provenance-graph
#' @export
provenance_digest <- function(x) {
  validate_provenance_graph(x)
  .canonical_digest(list(
    type = "provenance_graph", schema_version = x$schema_version,
    records = x$records
  ))
}

#' @rdname provenance-graph
#' @export
append_provenance <- function(x, records) {
  validate_provenance_graph(x)
  additions <- if (inherits(records, "provenance_record")) {
    list(records)
  } else {
    records
  }
  if (!is.list(additions)) {
    .feature_map_abort("records must contain provenance records.")
  }
  provenance_graph(c(x$records, additions))
}

#' @export
print.provenance_graph <- function(x, ...) {
  records <- provenance_records(x)
  cat("<provenance_graph>", length(records), "records\n")
  if (length(records)) {
    cat("  tips:", paste(provenance_tips(x), collapse = ", "), "\n")
  }
  invisible(x)
}

.as_provenance_graph <- function(x) {
  if (is.null(x)) return(provenance_graph())
  if (inherits(x, "provenance_graph")) return(provenance_graph(x))
  provenance_graph(provenance_record(
    "legacy_provenance", inputs = list(value = x),
    metadata = list(migrated = TRUE)
  ))
}

.relations_without_feature_domain <- function(x) {
  keep <- vapply(x, function(value) {
    if (inherits(value, "entity_feature_validity")) return(FALSE)
    if (inherits(value, "key_relation")) {
      return(!identical(value$source, "feature"))
    }
    !"feature" %in% c(value$from, value$to)
  }, logical(1))
  relation_registry(unclass(x)[keep])
}

.mapped_aligned_assay <- function(x, source) {
  structure(
    list(
      source = source,
      role = x$role,
      units = x$units,
      metadata = x$metadata
    ),
    class = "aligned_assay"
  )
}

#' Lazily transform a frame into a new feature domain
#'
#' @param x An `fmri_frame` or view.
#' @param target Optional parent-linked target space from which a canonical map
#'   can be derived.
#' @param map Optional explicit `feature_map`.
#' @param assay_rules Named rules for every assay: `"linear"` or
#'   `"independent_variance"`. Unnamed scalar rules are recycled.
#' @return A new linked-domain `fmri_frame` whose assays remain lazy.
#' @export
map_features <- function(x, target = NULL, map = NULL,
                         assay_rules = "linear") {
  if (!inherits(x, "fmri_frame")) {
    .feature_map_abort("x must be an fmri_frame or view.", field = "x")
  }
  if (is.null(map)) {
    if (is.null(target)) {
      .feature_map_abort("Supply target or map.", field = "map")
    }
    map <- feature_map_from_target(target)
  } else {
    validate_feature_map(map)
    if (!is.null(target)) assert_compatible_space(target, map$to)
  }
  assert_compatible_space(space(x), map$from)

  assay_names <- names(assays(x))
  if (length(assay_rules) == 1L && is.null(names(assay_rules))) {
    assay_rules <- rep(assay_rules, length(assay_names))
    names(assay_rules) <- assay_names
  }
  allowed <- c("linear", "independent_variance")
  if (!is.character(assay_rules) || is.null(names(assay_rules)) ||
      !identical(sort(names(assay_rules)), sort(assay_names)) ||
      anyNA(assay_rules) || any(!assay_rules %in% allowed)) {
    .feature_map_abort(
      "assay_rules must name every assay with a supported transformation rule.",
      field = "assay_rules", allowed = allowed
    )
  }
  assay_rules <- assay_rules[assay_names]
  mapped_assays <- lapply(assay_names, function(name) {
    descriptor <- assay(x, name)
    .mapped_aligned_assay(
      descriptor,
      feature_mapped_source(
        .frame_assay_source(x, name), map, rule = assay_rules[[name]]
      )
    )
  })
  names(mapped_assays) <- assay_names

  graph <- .as_provenance_graph(x$provenance %||% x$base$provenance)
  record <- provenance_record(
    "map_features",
    parents = provenance_tips(graph),
    inputs = list(
      source_space = space_digest(space(x)),
      feature_map = feature_map_digest(map),
      assays = lapply(assay_names, function(name) {
        source_fingerprint(.frame_assay_source(x, name))
      })
    ),
    parameters = list(assay_rules = as.list(assay_rules)),
    outputs = list(target_space = space_digest(map$to))
  )
  graph <- append_provenance(graph, record)

  fmri_frame(
    assays = mapped_assays,
    observations = observation_axis(x),
    features = feature_axis(feature_data(map$to), space = map$to),
    entities = entities(x),
    relations = .relations_without_feature_domain(relations(x)),
    tables = x$tables %||% x$base$tables,
    active_assay = active_assay(x),
    metadata = x$metadata %||% x$base$metadata,
    provenance = graph
  )
}
