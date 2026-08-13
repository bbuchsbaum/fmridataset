.fds_schema <- list(
  id = "org.fmridataset.fds/v1",
  version = 1L,
  object_types = "fmri_frame"
)

.fds_schema_abort <- function(message, field = NULL, ...) {
  .frame_abort(
    message,
    "fmridataset_error_schema",
    field = field,
    ...
  )
}

#' FDS logical schema identity
#'
#' FDS version 1 is the backend-neutral semantic contract for persisted
#' `fmri_frame` objects. Physical codecs may add locations, chunks, compression,
#' and checksums outside this manifest, but cannot change its field meanings.
#'
#' @return `fds_schema()` returns the immutable schema identity;
#'   `fds_schema_version()` returns its integer major version.
#' @export
fds_schema <- function() .fds_schema

#' @rdname fds_schema
#' @export
fds_schema_version <- function() .fds_schema$version

.fds_array_descriptor <- function(key, axes, data) {
  if (inherits(data, "array_source")) {
    shape <- source_shape(data)
    dtype <- source_dtype(data)
  } else {
    shape <- dim(data)
    if (is.null(shape) || length(shape) < 2L) {
      .fds_schema_abort(
        sprintf("Array '%s' must have at least two dimensions.", key),
        paste0("arrays.", key)
      )
    }
    dtype <- if (inherits(data, "Matrix")) "float64" else .source_dtype_from_data(data)
  }
  if (length(axes) != length(shape)) {
    .fds_schema_abort(
      sprintf("Array '%s' must name each logical axis.", key),
      paste0("arrays.", key, ".axes")
    )
  }
  list(
    key = key,
    axes = axes,
    shape = as.integer(shape),
    dtype = dtype
  )
}

.fds_block_manifests <- function(blocks, axis, prefix = paste0("axis/", axis)) {
  if (!length(blocks)) {
    return(list())
  }
  out <- lapply(names(blocks), function(name) {
    block <- blocks[[name]]
    key <- paste0(prefix, "/blocks/", name)
    list(
      name = name,
      array = key,
      components = block$components,
      role = block$role,
      units = block$units,
      metadata = block$metadata
    )
  })
  names(out) <- names(blocks)
  out
}

.fds_entity_manifest <- function(x, name) {
  list(
    name = name,
    key = entity_key(x),
    ids = entity_ids(x),
    data = entity_data(x),
    blocks = .fds_block_manifests(
      entity_blocks(x),
      paste0("entity:", name),
      prefix = paste0("entities/", name)
    ),
    entity_type = x$entity_type,
    metadata = x$metadata
  )
}

.fds_axis_manifest <- function(x, axis, include_space = FALSE) {
  value <- list(
    ids = axis_ids(x),
    id_column = x$id_col,
    data = x$data,
    blocks = .fds_block_manifests(x$blocks, axis),
    metadata = x$metadata
  )
  if (include_space) value$space <- x$space
  value
}

#' Construct and validate an FDS v1 frame manifest
#'
#' The manifest owns semantic alignment but deliberately excludes physical
#' source descriptors. Storage packages bind assay names to physical array
#' locations separately and reconstruct frames with `frame_from_fds_manifest()`.
#'
#' @param x An `fmri_frame`.
#' @return A serializable backend-neutral manifest.
#' @export
fds_frame_manifest <- function(x) {
  if (!inherits(x, "fmri_frame")) {
    .fds_schema_abort("x must be an fmri_frame.", "object_type")
  }
  observation <- observation_axis(x)
  feature <- feature_axis(x)
  observation_digest <- .axis_digest(observation)
  feature_digest <- .axis_digest(feature)
  assay_manifest <- lapply(assays(x), function(value) {
    list(
      name = value$name,
      array = paste0("assays/", value$name),
      dtype = value$dtype,
      shape = as.integer(dim(x)),
      observation_digest = observation_digest,
      feature_digest = feature_digest,
      role = value$role,
      units = value$units,
      metadata = value$metadata
    )
  })
  arrays <- lapply(names(assays(x)), function(name) {
    .fds_array_descriptor(
      paste0("assays/", name),
      c("observation", "feature"),
      .frame_assay_source(x, name)
    )
  })
  names(arrays) <- paste0("assays/", names(assays(x)))
  for (axis_name in c("observation", "feature")) {
    axis_value <- if (axis_name == "observation") observation else feature
    for (block_name in names(axis_value$blocks)) {
      key <- paste0("axis/", axis_name, "/blocks/", block_name)
      block_shape <- if (inherits(axis_value$blocks[[block_name]]$data, "array_source")) {
        source_shape(axis_value$blocks[[block_name]]$data)
      } else {
        dim(axis_value$blocks[[block_name]]$data)
      }
      extra_axes <- if (length(block_shape) > 2L) {
        paste0("dimension:", key, ":", seq.int(3L, length(block_shape)))
      } else {
        character()
      }
      arrays[[key]] <- .fds_array_descriptor(
        key,
        c(axis_name, paste0("component:", key), extra_axes),
        axis_value$blocks[[block_name]]$data
      )
    }
  }
  entity_values <- entities(x)
  for (entity_name in names(entity_values)) {
    entity_value <- entity_values[[entity_name]]
    for (block_name in names(entity_blocks(entity_value))) {
      key <- paste0("entities/", entity_name, "/blocks/", block_name)
      block_data <- axis_block_data(entity_blocks(entity_value)[[block_name]])
      block_shape <- if (inherits(block_data, "array_source")) {
        source_shape(block_data)
      } else {
        dim(block_data)
      }
      extra_axes <- if (length(block_shape) > 2L) {
        paste0("dimension:", key, ":", seq.int(3L, length(block_shape)))
      } else {
        character()
      }
      arrays[[key]] <- .fds_array_descriptor(
        key,
        c(paste0("entity:", entity_name), paste0("component:", key), extra_axes),
        block_data
      )
    }
  }
  entity_manifest <- lapply(names(entity_values), function(name) {
    .fds_entity_manifest(entity_values[[name]], name)
  })
  names(entity_manifest) <- names(entity_values)
  manifest <- list(
    schema = fds_schema(),
    object_type = "fmri_frame",
    shape = as.integer(dim(x)),
    axes = list(
      observation = .fds_axis_manifest(observation, "observation"),
      feature = .fds_axis_manifest(feature, "feature", include_space = TRUE)
    ),
    arrays = arrays,
    assays = assay_manifest,
    entities = entity_manifest,
    relations = relations(x),
    tables = x$tables,
    active_assay = active_assay(x),
    metadata = x$metadata,
    provenance = x$provenance,
    extensions = list()
  )
  validate_fds_manifest(manifest)
  manifest
}

.validate_manifest_entities <- function(values, arrays) {
  if (!is.list(values)) {
    .fds_schema_abort("Manifest entities must be a named list.", "entities")
  }
  if (!length(values)) {
    return(invisible(TRUE))
  }
  names_value <- names(values)
  if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
    anyDuplicated(names_value)) {
    .fds_schema_abort("Manifest entities must have unique, non-empty names.", "entities")
  }
  required <- c("name", "key", "ids", "data", "blocks", "entity_type", "metadata")
  for (name in names_value) {
    value <- values[[name]]
    field <- paste0("entities.", name)
    if (!is.list(value) || !all(required %in% names(value)) ||
      !identical(value$name, name)) {
      .fds_schema_abort(
        sprintf("Entity '%s' is missing required fields.", name),
        field
      )
    }
    .validate_manifest_ids(value$ids, length(value$ids), paste0("entity:", name))
    if (!is.character(value$key) || length(value$key) != 1L ||
      is.na(value$key) || !nzchar(value$key) ||
      !is.data.frame(value$data) || nrow(value$data) != length(value$ids) ||
      !value$key %in% names(value$data) ||
      !identical(as.character(value$data[[value$key]]), value$ids)) {
      .fds_schema_abort(
        sprintf("Entity '%s' has invalid or misaligned scalar keys.", name),
        paste0(field, ".key")
      )
    }
    if (!is.null(value$entity_type) &&
      (!is.character(value$entity_type) || length(value$entity_type) != 1L ||
        is.na(value$entity_type) || !nzchar(value$entity_type))) {
      .fds_schema_abort("Entity type must be NULL or one non-empty string.", paste0(field, ".entity_type"))
    }
    if (!is.list(value$blocks) ||
      (length(value$blocks) &&
        (is.null(names(value$blocks)) || any(!nzchar(names(value$blocks))) ||
          anyDuplicated(names(value$blocks))))) {
      .fds_schema_abort("Entity blocks must be a uniquely named list.", paste0(field, ".blocks"))
    }
    for (block_name in names(value$blocks)) {
      block <- value$blocks[[block_name]]
      block_required <- c("name", "array", "components", "role", "units", "metadata")
      if (!is.list(block) || !all(block_required %in% names(block)) ||
        !identical(block$name, block_name) || !is.character(block$array) ||
        length(block$array) != 1L || !block$array %in% names(arrays)) {
        .fds_schema_abort(
          sprintf("Entity block '%s.%s' has an invalid array reference.", name, block_name),
          paste0(field, ".blocks.", block_name)
        )
      }
      array <- arrays[[block$array]]
      expected_axis <- paste0("entity:", name)
      if (!identical(array$axes[[1L]], expected_axis) ||
        array$shape[[1L]] != length(value$ids) ||
        !is.data.frame(block$components) ||
        nrow(block$components) != array$shape[[2L]] ||
        !".component_id" %in% names(block$components)) {
        .fds_schema_abort(
          sprintf("Entity block '%s.%s' is not aligned with its keys or components.", name, block_name),
          paste0(field, ".blocks.", block_name)
        )
      }
    }
  }
  invisible(TRUE)
}

.manifest_entity_registry <- function(values) {
  out <- lapply(values, function(value) {
    entity_frame(
      data = value$data,
      key = value$key,
      blocks = list(),
      entity_type = value$entity_type,
      metadata = value$metadata
    )
  })
  names(out) <- names(values)
  entity_registry(out)
}

.validate_manifest_relations <- function(manifest) {
  values <- manifest$relations
  if (is.list(values) && !length(values) && !inherits(values, "relation_registry")) {
    values <- relation_registry()
  }
  if (!inherits(values, "relation_registry")) {
    .fds_schema_abort("Manifest relations must be a relation_registry.", "relations")
  }
  tryCatch(
    {
      observation <- manifest$axes$observation
      feature <- manifest$axes$feature
      resolved <- .resolve_relation_registry(
        values,
        axis_frame(
          observation$data,
          id = observation$ids,
          axis = "observation",
          id_col = observation$id_column,
          metadata = observation$metadata
        ),
        feature_axis(
          feature$data,
          space = feature$space,
          metadata = feature$metadata
        ),
        .manifest_entity_registry(manifest$entities)
      )
      if (!identical(resolved, values)) {
        .fds_schema_abort("Manifest relations must use normalized domain names.", "relations")
      }
    },
    fmridataset_error_schema = function(error) stop(error),
    error = function(error) {
      .fds_schema_abort(
        paste0("Manifest relation validation failed: ", conditionMessage(error)),
        "relations"
      )
    }
  )
  invisible(TRUE)
}

.validate_manifest_ids <- function(ids, expected_n, axis) {
  if (!is.character(ids) || length(ids) != expected_n || anyNA(ids) ||
    any(!nzchar(ids)) || anyDuplicated(ids)) {
    .fds_schema_abort(
      sprintf("%s axis IDs must be unique, non-empty strings matching the axis length.", axis),
      paste0("axes.", axis, ".ids")
    )
  }
}

.validate_manifest_axis <- function(value, expected_n, axis, arrays, require_space = FALSE) {
  required <- c("ids", "id_column", "data", "blocks", "metadata")
  if (!is.list(value) || !all(required %in% names(value))) {
    .fds_schema_abort(
      sprintf("The %s axis is missing required fields.", axis),
      paste0("axes.", axis)
    )
  }
  .validate_manifest_ids(value$ids, expected_n, axis)
  if (!is.character(value$id_column) || length(value$id_column) != 1L ||
    is.na(value$id_column) || !nzchar(value$id_column)) {
    .fds_schema_abort("Axis id_column must be one non-empty string.", paste0("axes.", axis, ".id_column"))
  }
  if (!is.data.frame(value$data) || nrow(value$data) != expected_n ||
    !value$id_column %in% names(value$data) ||
    !identical(as.character(value$data[[value$id_column]]), value$ids)) {
    .fds_schema_abort(
      sprintf("The %s scalar data are not aligned with their IDs.", axis),
      paste0("axes.", axis, ".data")
    )
  }
  if (!is.list(value$blocks) ||
    (length(value$blocks) && (is.null(names(value$blocks)) || any(!nzchar(names(value$blocks))) || anyDuplicated(names(value$blocks))))) {
    .fds_schema_abort("Axis blocks must be a uniquely named list.", paste0("axes.", axis, ".blocks"))
  }
  for (name in names(value$blocks)) {
    block <- value$blocks[[name]]
    block_required <- c("name", "array", "components", "role", "units", "metadata")
    if (!is.list(block) || !all(block_required %in% names(block)) ||
      !identical(block$name, name) || !is.character(block$array) ||
      length(block$array) != 1L || !block$array %in% names(arrays)) {
      .fds_schema_abort(
        sprintf("Axis block '%s' has an invalid array reference.", name),
        paste0("axes.", axis, ".blocks.", name)
      )
    }
    array <- arrays[[block$array]]
    if (!identical(array$axes[[1L]], axis) || array$shape[[1L]] != expected_n ||
      !is.data.frame(block$components) || nrow(block$components) != array$shape[[2L]] ||
      !".component_id" %in% names(block$components)) {
      .fds_schema_abort(
        sprintf("Axis block '%s' is not aligned with the %s axis or components.", name, axis),
        paste0("axes.", axis, ".blocks.", name)
      )
    }
  }
  if (require_space) {
    if (!inherits(value$space, "feature_space") ||
      !identical(feature_ids(value$space), value$ids)) {
      .fds_schema_abort(
        "The feature space IDs do not exactly match the feature axis IDs.",
        "axes.feature.space"
      )
    }
  }
  invisible(TRUE)
}

#' @param manifest An FDS manifest.
#' @rdname fds_frame_manifest
#' @export
validate_fds_manifest <- function(manifest) {
  required <- c(
    "schema", "object_type", "shape", "axes", "arrays", "assays", "entities",
    "relations", "tables", "active_assay", "metadata", "provenance", "extensions"
  )
  if (!is.list(manifest) || !all(required %in% names(manifest))) {
    .fds_schema_abort("The FDS manifest is missing required fields.", "manifest")
  }
  if (!is.list(manifest$schema) ||
    !identical(manifest$schema$id, .fds_schema$id) ||
    !identical(manifest$schema$version, .fds_schema$version)) {
    .fds_schema_abort(
      "Unsupported FDS schema identity or version.",
      "schema",
      supported = .fds_schema
    )
  }
  if (!identical(manifest$object_type, "fmri_frame")) {
    .fds_schema_abort("FDS v1 currently supports object_type fmri_frame.", "object_type")
  }
  if (inherits(manifest$provenance, "provenance_graph")) {
    tryCatch(
      validate_provenance_graph(manifest$provenance),
      error = function(error) {
        .fds_schema_abort(
          paste0("Invalid provenance graph: ", conditionMessage(error)),
          "provenance"
        )
      }
    )
  }
  shape <- manifest$shape
  if (!is.numeric(shape) || length(shape) != 2L || anyNA(shape) ||
    any(shape < 0) || any(shape != as.integer(shape))) {
    .fds_schema_abort("Frame shape must contain two non-negative integers.", "shape")
  }
  shape <- as.integer(shape)
  array_names <- names(manifest$arrays)
  if (!is.list(manifest$arrays) || !length(manifest$arrays) ||
    is.null(array_names) || any(!nzchar(array_names)) || anyDuplicated(array_names)) {
    .fds_schema_abort("Frame arrays must be a non-empty uniquely named registry.", "arrays")
  }
  for (name in array_names) {
    value <- manifest$arrays[[name]]
    if (!is.list(value) || !all(c("key", "axes", "shape", "dtype") %in% names(value)) ||
      !identical(value$key, name) || !is.character(value$axes) ||
      !is.numeric(value$shape) || length(value$shape) < 2L ||
      length(value$axes) != length(value$shape) || anyNA(value$shape) ||
      any(value$shape < 0) || any(value$shape != as.integer(value$shape)) ||
      !is.character(value$dtype) || length(value$dtype) != 1L ||
      !value$dtype %in% .supported_source_dtypes) {
      .fds_schema_abort(sprintf("Array '%s' has an invalid declaration.", name), paste0("arrays.", name))
    }
  }
  if (!is.list(manifest$axes) || !all(c("observation", "feature") %in% names(manifest$axes))) {
    .fds_schema_abort("Frame manifests require observation and feature axes.", "axes")
  }
  if (!identical(
    c(length(manifest$axes$observation$ids), length(manifest$axes$feature$ids)),
    shape
  )) {
    .fds_schema_abort("Frame shape does not match the declared axis lengths.", "shape")
  }
  .validate_manifest_axis(manifest$axes$observation, shape[[1L]], "observation", manifest$arrays)
  .validate_manifest_axis(manifest$axes$feature, shape[[2L]], "feature", manifest$arrays, require_space = TRUE)
  .validate_manifest_entities(manifest$entities, manifest$arrays)
  .validate_manifest_relations(manifest)

  assay_names <- names(manifest$assays)
  if (!is.list(manifest$assays) || !length(manifest$assays) ||
    is.null(assay_names) || any(!nzchar(assay_names)) || anyDuplicated(assay_names)) {
    .fds_schema_abort("Frame assays must be a non-empty uniquely named list.", "assays")
  }
  observation_digest <- .canonical_digest(manifest$axes$observation$ids)
  feature_digest <- .canonical_digest(manifest$axes$feature$ids)
  assay_required <- c(
    "name", "array", "dtype", "shape", "observation_digest", "feature_digest",
    "role", "units", "metadata"
  )
  for (name in assay_names) {
    value <- manifest$assays[[name]]
    if (!is.list(value) || !all(assay_required %in% names(value)) ||
      !identical(value$name, name) || !identical(value$array, paste0("assays/", name)) ||
      !value$array %in% names(manifest$arrays)) {
      .fds_schema_abort(sprintf("Assay '%s' has an invalid descriptor.", name), paste0("assays.", name))
    }
    if (!identical(as.integer(value$shape), shape)) {
      .fds_schema_abort(sprintf("Assay '%s' shape does not match the frame shape.", name), paste0("assays.", name, ".shape"))
    }
    array <- manifest$arrays[[value$array]]
    if (!identical(array$axes, c("observation", "feature")) ||
      !identical(as.integer(array$shape), shape) ||
      !identical(array$dtype, value$dtype)) {
      .fds_schema_abort(sprintf("Assay '%s' array declaration is inconsistent.", name), paste0("assays.", name, ".array"))
    }
    if (!is.character(value$dtype) || length(value$dtype) != 1L ||
      is.na(value$dtype) || !value$dtype %in% .supported_source_dtypes) {
      .fds_schema_abort(sprintf("Assay '%s' has an unsupported dtype.", name), paste0("assays.", name, ".dtype"))
    }
    if (!identical(value$observation_digest, observation_digest) ||
      !identical(value$feature_digest, feature_digest)) {
      .fds_schema_abort(sprintf("Assay '%s' axis digest does not match the manifest axes.", name), paste0("assays.", name, ".digest"))
    }
    if (any(c("source", "uri", "dataset", "chunks") %in% names(value))) {
      .fds_schema_abort(sprintf("Assay '%s' contains physical source fields.", name), paste0("assays.", name))
    }
  }
  if (!is.character(manifest$active_assay) || length(manifest$active_assay) != 1L ||
    !manifest$active_assay %in% assay_names) {
    .fds_schema_abort("active_assay must name one manifest assay.", "active_assay")
  }
  if (.source_contains_runtime_state(manifest)) {
    .fds_schema_abort(
      "FDS manifests cannot contain runtime functions, environments, or external pointers.",
      "runtime_state"
    )
  }
  invisible(manifest)
}

#' Extract physical array bindings from a frame
#'
#' Storage codecs use this helper to pair each source-free FDS array declaration
#' with its current physical or in-memory `array_source`.
#'
#' @param x An `fmri_frame`.
#' @return A named list of `array_source` descriptors keyed exactly like the
#'   manifest `arrays` registry.
#' @export
fds_frame_bindings <- function(x) {
  manifest <- fds_frame_manifest(x)
  out <- lapply(names(assays(x)), function(name) .frame_assay_source(x, name))
  names(out) <- paste0("assays/", names(assays(x)))
  for (axis_name in c("observation", "feature")) {
    axis_value <- if (axis_name == "observation") observation_axis(x) else feature_axis(x)
    for (block_name in names(axis_value$blocks)) {
      key <- paste0("axis/", axis_name, "/blocks/", block_name)
      data <- axis_value$blocks[[block_name]]$data
      out[[key]] <- if (inherits(data, "array_source") || length(dim(data)) != 2L) {
        data
      } else {
        tryCatch(as_array_source(data), error = function(error) data)
      }
    }
  }
  entity_values <- entities(x)
  for (entity_name in names(entity_values)) {
    entity_value <- entity_values[[entity_name]]
    for (block_name in names(entity_blocks(entity_value))) {
      key <- paste0("entities/", entity_name, "/blocks/", block_name)
      data <- axis_block_data(entity_blocks(entity_value)[[block_name]])
      out[[key]] <- if (inherits(data, "array_source") || length(dim(data)) != 2L) {
        data
      } else {
        tryCatch(as_array_source(data), error = function(error) data)
      }
    }
  }
  out <- out[names(manifest$arrays)]
  out
}

#' Compute a canonical FDS manifest digest
#'
#' @param manifest A valid FDS manifest.
#' @return A stable hexadecimal digest over semantic manifest content.
#' @export
fds_manifest_digest <- function(manifest) {
  validate_fds_manifest(manifest)
  .canonical_digest(manifest)
}

#' Reconstruct a frame from an FDS manifest and physical sources
#'
#' @param manifest A valid FDS v1 frame manifest.
#' @param bindings Named physical array payloads or `array_source` descriptors,
#'   one per manifest array declaration.
#' @return An `fmri_frame` whose semantic state comes from `manifest` and whose
#'   lazy arrays come from `bindings`.
#' @export
frame_from_fds_manifest <- function(manifest, bindings) {
  validate_fds_manifest(manifest)
  expected <- names(manifest$arrays)
  if (!is.list(bindings) || is.null(names(bindings)) || anyDuplicated(names(bindings)) ||
    !setequal(names(bindings), expected)) {
    .fds_schema_abort("Physical binding names must exactly match manifest arrays.", "bindings")
  }
  bindings <- bindings[expected]
  bindings <- lapply(expected, function(key) {
    value <- bindings[[key]]
    descriptor <- manifest$arrays[[key]]
    if (inherits(value, "array_source")) {
      validate_array_source(value)
      actual_shape <- source_shape(value)
      actual_dtype <- source_dtype(value)
    } else {
      actual_shape <- dim(value)
      actual_dtype <- if (inherits(value, "Matrix")) "float64" else .source_dtype_from_data(value)
    }
    if (is.null(actual_shape) || !identical(as.integer(actual_shape), as.integer(descriptor$shape))) {
      .fds_schema_abort(sprintf("Physical binding '%s' shape does not match its array.", key), paste0("bindings.", key, ".shape"))
    }
    if (!identical(actual_dtype, descriptor$dtype)) {
      .fds_schema_abort(sprintf("Physical binding '%s' dtype does not match its array.", key), paste0("bindings.", key, ".dtype"))
    }
    value
  })
  names(bindings) <- expected
  annotated <- lapply(names(manifest$assays), function(name) {
    descriptor <- manifest$assays[[name]]
    source <- as_array_source(bindings[[descriptor$array]])
    structure(
      list(
        name = name,
        source = source,
        role = descriptor$role,
        units = descriptor$units,
        metadata = descriptor$metadata
      ),
      class = "aligned_assay"
    )
  })
  names(annotated) <- names(manifest$assays)
  observation <- manifest$axes$observation
  feature <- manifest$axes$feature
  rebuild_blocks <- function(axis_manifest) {
    out <- lapply(axis_manifest$blocks, function(block) {
      axis_block(
        bindings[[block$array]],
        components = block$components,
        role = block$role,
        units = block$units,
        metadata = block$metadata
      )
    })
    names(out) <- names(axis_manifest$blocks)
    out
  }
  rebuild_entities <- function(entity_manifests) {
    out <- lapply(entity_manifests, function(value) {
      blocks <- lapply(value$blocks, function(block) {
        axis_block(
          bindings[[block$array]],
          components = block$components,
          role = block$role,
          units = block$units,
          metadata = block$metadata
        )
      })
      names(blocks) <- names(value$blocks)
      entity_frame(
        data = value$data,
        key = value$key,
        blocks = blocks,
        entity_type = value$entity_type,
        metadata = value$metadata
      )
    })
    names(out) <- names(entity_manifests)
    entity_registry(out)
  }
  fmri_frame(
    assays = annotated,
    observations = axis_frame(
      observation$data,
      blocks = rebuild_blocks(observation),
      id = observation$ids,
      axis = "observation",
      id_col = observation$id_column,
      metadata = observation$metadata
    ),
    features = feature_axis(
      feature$data,
      space = feature$space,
      blocks = rebuild_blocks(feature),
      metadata = feature$metadata
    ),
    entities = rebuild_entities(manifest$entities),
    relations = manifest$relations,
    tables = manifest$tables,
    active_assay = manifest$active_assay,
    metadata = manifest$metadata,
    provenance = manifest$provenance
  )
}
