.fds_study_schema <- list(
  id = "org.fmridataset.fds-study/v1",
  version = 1L
)

.fds_study_representation_manifest <- function(value) {
  if (inherits(value, "fmri_collection")) {
    members <- lapply(collection_frames(value), fds_frame_manifest)
    return(list(
      type = "fmri_collection",
      members = members,
      metadata = value$metadata,
      provenance = value$provenance
    ))
  }
  list(type = "fmri_frame", manifest = fds_frame_manifest(value))
}

.fds_study_entity_arrays <- function(registry) {
  arrays <- list()
  for (entity_name in entity_names(registry)) {
    value <- registry[[entity_name]]
    for (block_name in names(entity_blocks(value))) {
      key <- paste0("entities/", entity_name, "/blocks/", block_name)
      data <- axis_block_data(entity_blocks(value)[[block_name]])
      shape <- if (inherits(data, "array_source")) source_shape(data) else dim(data)
      extra_axes <- if (length(shape) > 2L) {
        paste0("dimension:", key, ":", seq.int(3L, length(shape)))
      } else {
        character()
      }
      arrays[[key]] <- .fds_array_descriptor(
        key,
        c(paste0("entity:", entity_name), paste0("component:", key), extra_axes),
        data
      )
    }
  }
  arrays
}

#' Construct and validate an FDS v1 study manifest
#'
#' Study manifests retain shared entities, typed links, relational tables, and
#' the semantic manifests of every frame or collection member. Numerical
#' sources remain separate bindings so physical storage packages do not own or
#' reinterpret study semantics.
#'
#' @param x An `fmri_study` or filtered study view.
#' @param manifest An FDS study manifest.
#' @return `fds_study_manifest()` returns a serializable source-free manifest;
#'   `validate_fds_study_manifest()` returns `manifest` invisibly.
#' @export
fds_study_manifest <- function(x) {
  validate_fmri_study(x)
  frame_values <- fds_study_representations(x)
  entity_values <- entities(x)
  arrays <- .fds_study_entity_arrays(entity_values)
  entity_manifests <- lapply(entity_names(entity_values), function(name) {
    .fds_entity_manifest(entity_values[[name]], name)
  })
  names(entity_manifests) <- entity_names(entity_values)
  base <- .study_base(x)
  manifest <- list(
    schema = .fds_study_schema,
    object_type = "fmri_study",
    representations = lapply(frame_values, .fds_study_representation_manifest),
    arrays = arrays,
    entities = entity_manifests,
    links = study_links(x),
    tables = study_tables(x),
    metadata = base$metadata,
    provenance = base$provenance,
    extensions = list()
  )
  validate_fds_study_manifest(manifest)
  manifest
}

#' Extract canonical study representations for persistence
#'
#' Filtered study views are compacted against their visible shared entities so
#' the persisted object is a self-contained study rather than a view retaining
#' references to filtered-out registry rows.
#'
#' @param x An `fmri_study` or filtered study view.
#' @return Named frames and collections matching `fds_study_manifest(x)`.
#' @export
fds_study_representations <- function(x) {
  validate_fmri_study(x)
  values <- study_frames(x, contextual = FALSE)
  if (!inherits(x, "fmri_study_view")) return(values)
  shared <- entities(x)
  lapply(values, .contextualize_study_frame, shared = shared)
}

.validate_study_array_declarations <- function(arrays) {
  if (!is.list(arrays)) {
    .fds_schema_abort("Study arrays must be a named registry.", "arrays")
  }
  if (!length(arrays)) return(invisible(TRUE))
  array_names <- names(arrays)
  if (is.null(array_names) || anyNA(array_names) || any(!nzchar(array_names)) ||
      anyDuplicated(array_names)) {
    .fds_schema_abort("Study arrays must have unique non-empty names.", "arrays")
  }
  for (name in array_names) {
    value <- arrays[[name]]
    if (!is.list(value) ||
        !all(c("key", "axes", "shape", "dtype") %in% names(value)) ||
        !identical(value$key, name) || !is.character(value$axes) ||
        !is.numeric(value$shape) || length(value$shape) < 2L ||
        length(value$axes) != length(value$shape) || anyNA(value$shape) ||
        any(value$shape < 0) || any(value$shape != as.integer(value$shape)) ||
        !is.character(value$dtype) || length(value$dtype) != 1L ||
        !value$dtype %in% .supported_source_dtypes) {
      .fds_schema_abort(
        sprintf("Study array '%s' has an invalid declaration.", name),
        paste0("arrays.", name)
      )
    }
  }
  invisible(TRUE)
}

.validate_study_representation_manifest <- function(value, name) {
  if (!is.list(value) || !is.character(value$type) || length(value$type) != 1L) {
    .fds_schema_abort("Study representation has no valid type.", paste0("representations.", name))
  }
  if (identical(value$type, "fmri_frame")) {
    if (!identical(names(value), c("type", "manifest"))) {
      .fds_schema_abort("Frame representation has invalid fields.", paste0("representations.", name))
    }
    validate_fds_manifest(value$manifest)
    return(invisible(TRUE))
  }
  if (!identical(value$type, "fmri_collection") ||
      !identical(names(value), c("type", "members", "metadata", "provenance")) ||
      !is.list(value$members) || !length(value$members)) {
    .fds_schema_abort("Study representation type is unsupported or invalid.", paste0("representations.", name))
  }
  member_names <- names(value$members)
  if (is.null(member_names) || anyNA(member_names) || any(!nzchar(member_names)) ||
      anyDuplicated(member_names)) {
    .fds_schema_abort("Collection members require unique stable names.", paste0("representations.", name, ".members"))
  }
  for (member_name in member_names) validate_fds_manifest(value$members[[member_name]])
  tryCatch(
    .validate_container_provenance(value$provenance, "FDS collection representation"),
    error = function(error) {
      .fds_schema_abort(conditionMessage(error), paste0("representations.", name, ".provenance"))
    }
  )
  member_domains <- integer()
  for (member_name in member_names) {
    member <- value$members[[member_name]]
    sizes <- c(
      observation = length(member$axes$observation$ids),
      feature = length(member$axes$feature$ids)
    )
    sizes <- sizes[sizes > 1L]
    if (length(sizes)) {
      names(sizes) <- paste0("member:", member_name, ":", names(sizes))
      member_domains <- c(member_domains, sizes)
    }
  }
  tryCatch(
    validate_unaligned_record(value$metadata, member_domains),
    error = function(error) {
      .fds_schema_abort(conditionMessage(error), paste0("representations.", name, ".metadata"))
    }
  )
  invisible(TRUE)
}

.study_manifest_axis_ids <- function(representation, axis, endpoint) {
  if (identical(representation$type, "fmri_collection")) {
    .fds_schema_abort(
      sprintf("Mapped link endpoint '%s' is a collection and has no single %s axis.", endpoint, axis),
      "links"
    )
  }
  representation$manifest$axes[[axis]]$ids
}

.validate_study_manifest_links <- function(links, representations) {
  if (!is.list(links)) .fds_schema_abort("Study links must be a named list.", "links")
  if (!length(links)) return(invisible(TRUE))
  link_names <- names(links)
  if (is.null(link_names) || anyNA(link_names) || any(!nzchar(link_names)) ||
      anyDuplicated(link_names)) {
    .fds_schema_abort("Study links require unique non-empty names.", "links")
  }
  for (name in link_names) {
    value <- tryCatch(
      {
        .validate_frame_link(links[[name]], name)
        links[[name]]
      },
      error = function(error) {
        .fds_schema_abort(
          paste0("Invalid study link: ", conditionMessage(error)),
          paste0("links.", name)
        )
      }
    )
    if (!all(c(value$from, value$to) %in% names(representations))) {
      .fds_schema_abort(
        sprintf("Study link '%s' has an unknown endpoint.", name),
        paste0("links.", name)
      )
    }
    if (!is.null(value$map)) {
      from_ids <- .study_manifest_axis_ids(
        representations[[value$from]], value$from_axis, value$from
      )
      to_ids <- .study_manifest_axis_ids(
        representations[[value$to]], value$to_axis, value$to
      )
      if (any(!value$map$.from_id %in% from_ids) ||
          any(!value$map$.to_id %in% to_ids)) {
        .fds_schema_abort(
          sprintf("Study link '%s' map contains unknown axis IDs.", name),
          paste0("links.", name, ".map")
        )
      }
    }
    typed_map <- value$metadata$feature_map
    if (!is.null(typed_map)) {
      tryCatch(
        {
          validate_feature_map(typed_map)
          if (!identical(value$type, "mapped_from") ||
              !identical(value$from_axis, "feature") ||
              !identical(value$to_axis, "feature")) {
            .fds_schema_abort(
              sprintf("Study link '%s' uses a feature_map outside a feature mapped_from link.", name),
              paste0("links.", name, ".metadata.feature_map")
            )
          }
          from_representation <- representations[[value$from]]
          to_representation <- representations[[value$to]]
          if (identical(from_representation$type, "fmri_collection") ||
              identical(to_representation$type, "fmri_collection")) {
            .fds_schema_abort(
              sprintf("Study link '%s' feature_map endpoints must be single frames.", name),
              paste0("links.", name, ".metadata.feature_map")
            )
          }
          assert_compatible_space(
            feature_map_source_space(typed_map),
            to_representation$manifest$axes$feature$space
          )
          assert_compatible_space(
            feature_map_target_space(typed_map),
            from_representation$manifest$axes$feature$space
          )
        },
        error = function(error) {
          .fds_schema_abort(
            paste0("Invalid study feature map: ", conditionMessage(error)),
            paste0("links.", name, ".metadata.feature_map")
          )
        }
      )
    }
  }
  invisible(TRUE)
}

#' @rdname fds_study_manifest
#' @export
validate_fds_study_manifest <- function(manifest) {
  required <- c(
    "schema", "object_type", "representations", "arrays", "entities",
    "links", "tables", "metadata", "provenance", "extensions"
  )
  if (!is.list(manifest) || !identical(names(manifest), required)) {
    .fds_schema_abort("The FDS study manifest is missing required fields.", "manifest")
  }
  if (!identical(manifest$schema, .fds_study_schema)) {
    .fds_schema_abort("Unsupported FDS study schema identity or version.", "schema")
  }
  if (!identical(manifest$object_type, "fmri_study")) {
    .fds_schema_abort("FDS study manifests require object_type fmri_study.", "object_type")
  }
  tryCatch(
    .validate_container_provenance(manifest$provenance, "FDS study manifest"),
    error = function(error) {
      .fds_schema_abort(conditionMessage(error), "provenance")
    }
  )
  representations <- manifest$representations
  representation_names <- names(representations)
  if (!is.list(representations) || !length(representations) ||
      is.null(representation_names) || anyNA(representation_names) ||
      any(!nzchar(representation_names)) || anyDuplicated(representation_names)) {
    .fds_schema_abort("Study representations require unique stable names.", "representations")
  }
  for (name in representation_names) {
    .validate_study_representation_manifest(representations[[name]], name)
  }
  .validate_study_array_declarations(manifest$arrays)
  .validate_manifest_entities(manifest$entities, manifest$arrays)
  .validate_study_manifest_links(manifest$links, representations)
  tryCatch(
    .validate_study_tables(manifest$tables, .manifest_entity_registry(manifest$entities)),
    error = function(error) {
      .fds_schema_abort(
        paste0("Study table validation failed: ", conditionMessage(error)),
        "tables"
      )
    }
  )
  metadata_domains <- integer()
  for (name in names(manifest$entities)) {
    size <- length(manifest$entities[[name]]$ids)
    if (size > 1L) metadata_domains[[paste0("entity:", name)]] <- size
  }
  for (name in representation_names) {
    representation <- representations[[name]]
    if (identical(representation$type, "fmri_frame")) {
      observation_size <- length(representation$manifest$axes$observation$ids)
      feature_size <- length(representation$manifest$axes$feature$ids)
      if (observation_size > 1L) {
        metadata_domains[[paste0("representation:", name, ":observation")]] <-
          observation_size
      }
      if (feature_size > 1L) {
        metadata_domains[[paste0("representation:", name, ":feature")]] <-
          feature_size
      }
    } else {
      for (member_name in names(representation$members)) {
        member <- representation$members[[member_name]]
        observation_size <- length(member$axes$observation$ids)
        feature_size <- length(member$axes$feature$ids)
        if (observation_size > 1L) {
          metadata_domains[[paste0(
            "representation:", name, ":member:", member_name, ":observation"
          )]] <- observation_size
        }
        if (feature_size > 1L) {
          metadata_domains[[paste0(
            "representation:", name, ":member:", member_name, ":feature"
          )]] <- feature_size
        }
      }
    }
  }
  tryCatch(
    validate_unaligned_record(manifest$metadata, metadata_domains),
    error = function(error) {
      .fds_schema_abort(conditionMessage(error), "metadata")
    }
  )
  if (.source_contains_runtime_state(manifest)) {
    .fds_schema_abort(
      "FDS study manifests cannot contain runtime functions, environments, or external pointers.",
      "runtime_state"
    )
  }
  invisible(manifest)
}

#' Extract shared study-level physical bindings
#'
#' @param x An `fmri_study` or filtered study view.
#' @return A named list of shared entity-block payloads. Representation arrays
#'   remain owned by their individual frame bindings.
#' @export
fds_study_bindings <- function(x) {
  manifest <- fds_study_manifest(x)
  registry <- entities(x)
  out <- list()
  for (entity_name in entity_names(registry)) {
    value <- registry[[entity_name]]
    for (block_name in names(entity_blocks(value))) {
      key <- paste0("entities/", entity_name, "/blocks/", block_name)
      data <- axis_block_data(entity_blocks(value)[[block_name]])
      out[[key]] <- if (inherits(data, "array_source") || length(dim(data)) != 2L) {
        data
      } else {
        tryCatch(as_array_source(data), error = function(error) data)
      }
    }
  }
  out[names(manifest$arrays)]
}

.validate_study_binding <- function(value, descriptor, key) {
  if (inherits(value, "array_source")) {
    validate_array_source(value)
    actual_shape <- source_shape(value)
    actual_dtype <- source_dtype(value)
  } else {
    actual_shape <- dim(value)
    actual_dtype <- if (inherits(value, "Matrix")) "float64" else .source_dtype_from_data(value)
  }
  if (is.null(actual_shape) ||
      !identical(as.integer(actual_shape), as.integer(descriptor$shape))) {
    .fds_schema_abort(
      sprintf("Physical study binding '%s' shape does not match its array.", key),
      paste0("bindings.", key, ".shape")
    )
  }
  if (!identical(actual_dtype, descriptor$dtype)) {
    .fds_schema_abort(
      sprintf("Physical study binding '%s' dtype does not match its array.", key),
      paste0("bindings.", key, ".dtype")
    )
  }
  value
}

.study_entities_from_manifest <- function(manifest, bindings) {
  out <- lapply(manifest$entities, function(value) {
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
  names(out) <- names(manifest$entities)
  entity_registry(out)
}

.validate_bound_representations <- function(manifest, representations) {
  expected <- names(manifest$representations)
  if (!is.list(representations) || is.null(names(representations)) ||
      anyDuplicated(names(representations)) || !setequal(names(representations), expected)) {
    .fds_schema_abort(
      "Physical representation names must exactly match the study manifest.",
      "representations"
    )
  }
  representations <- representations[expected]
  for (name in expected) {
    descriptor <- manifest$representations[[name]]
    value <- representations[[name]]
    if (identical(descriptor$type, "fmri_frame")) {
      if (!inherits(value, "fmri_frame") ||
          !identical(fds_frame_manifest(value), descriptor$manifest)) {
        .fds_schema_abort(
          sprintf("Frame representation '%s' does not match its manifest.", name),
          paste0("representations.", name)
        )
      }
      next
    }
    if (!inherits(value, "fmri_collection") ||
        !identical(names(collection_frames(value)), names(descriptor$members)) ||
        !identical(value$metadata, descriptor$metadata) ||
        !identical(value$provenance, descriptor$provenance)) {
      .fds_schema_abort(
        sprintf("Collection representation '%s' does not match its manifest.", name),
        paste0("representations.", name)
      )
    }
    for (member_name in names(descriptor$members)) {
      if (!identical(
        fds_frame_manifest(collection_frame(value, member_name)),
        descriptor$members[[member_name]]
      )) {
        .fds_schema_abort(
          sprintf("Collection member '%s.%s' does not match its manifest.", name, member_name),
          paste0("representations.", name, ".members.", member_name)
        )
      }
    }
  }
  representations
}

#' Reconstruct a study from semantic and physical components
#'
#' @param manifest A valid FDS v1 study manifest.
#' @param representations Named lazy frames or collections matching the
#'   representation manifests.
#' @param bindings Named physical bindings for shared study arrays.
#' @return An `fmri_study`.
#' @export
study_from_fds_manifest <- function(manifest, representations, bindings = list()) {
  validate_fds_study_manifest(manifest)
  representations <- .validate_bound_representations(manifest, representations)
  expected <- names(manifest$arrays)
  if (!is.list(bindings) ||
      (length(expected) && (is.null(names(bindings)) || anyDuplicated(names(bindings)))) ||
      !setequal(names(bindings), expected)) {
    .fds_schema_abort(
      "Physical study binding names must exactly match manifest arrays.",
      "bindings"
    )
  }
  bindings <- bindings[expected]
  if (length(expected)) {
    bindings <- lapply(expected, function(key) {
      .validate_study_binding(bindings[[key]], manifest$arrays[[key]], key)
    })
    names(bindings) <- expected
  }
  fmri_study(
    frames = representations,
    entities = .study_entities_from_manifest(manifest, bindings),
    links = manifest$links,
    tables = manifest$tables,
    metadata = manifest$metadata,
    provenance = manifest$provenance
  )
}

#' Compute a canonical FDS study-manifest digest
#'
#' @param manifest A valid FDS study manifest.
#' @return A stable hexadecimal digest over source-free study semantics.
#' @export
fds_study_manifest_digest <- function(manifest) {
  validate_fds_study_manifest(manifest)
  .canonical_digest(manifest)
}
