.collection_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_collection", ...)
}

.assert_collection_semantics <- function(reference, candidate, frame_id) {
  report <- compare_frame_schema(candidate, reference, mode = "collection")
  if (!report$compatible) .collection_abort(
    sprintf("Frame '%s' has an incompatible schema at '%s'.", frame_id, report$path),
    frame = frame_id, field = report$path,
    expected = report$expected, actual = report$actual
  )
  invisible(TRUE)
}

.collection_frame_descriptor <- function(frame) {
  list(
    signature = frame_schema(frame),
    observation_ids = observation_ids(frame),
    feature_ids = feature_ids(frame),
    observation_data = observations(frame),
    feature_data = features(frame),
    assay_sources = lapply(names(assays(frame)), function(name) {
      source_fingerprint(.frame_assay_source(frame, name))
    }),
    space_digest = space_digest(space(frame)),
    entity_digest = entity_registry_digest(frame),
    relation_digest = relation_registry_digest(frame)
  )
}

#' Construct a collection of semantically equivalent fMRI frames
#'
#' A collection keeps frames separate when they share an observational and
#' assay contract but cannot share a feature axis, as with participant-native
#' volume or surface spaces. Equal feature dimensions or IDs are not required;
#' feature-space type and annotation semantics are validated explicitly.
#'
#' @param frames A non-empty named list of `fmri_frame` objects or lazy views.
#' @param metadata Serializable collection metadata.
#' @param provenance Serializable provenance records.
#' @return An `fmri_collection`.
#' @export
fmri_collection <- function(frames, metadata = list(), provenance = NULL) {
  if (!is.list(frames) || !length(frames)) {
    .collection_abort("frames must be a non-empty named list.", field = "frames")
  }
  ids <- names(frames)
  if (is.null(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
    .collection_abort(
      "Collection frame names must be unique, non-missing stable IDs.",
      field = "names"
    )
  }
  valid <- vapply(frames, inherits, logical(1), "fmri_frame")
  if (!all(valid)) {
    .collection_abort(
      "Every collection member must be an fmri_frame or fmri_view.",
      frames = ids[!valid]
    )
  }
  if (length(frames) > 1L) {
    for (i in 2:length(frames)) {
      .assert_collection_semantics(frames[[1L]], frames[[i]], ids[[i]])
    }
  }
  if (inherits(provenance, "provenance_graph")) {
    validate_provenance_graph(provenance)
  }
  out <- structure(
    list(
      frames = frames,
      metadata = metadata,
      provenance = provenance,
      schema_version = 1L
    ),
    class = "fmri_collection"
  )
  if (.source_contains_runtime_state(out)) {
    .collection_abort(
      "Collections cannot contain runtime functions, environments, or external pointers."
    )
  }
  out
}

#' Validate an fMRI collection
#'
#' @param x An `fmri_collection`.
#' @return `x`, invisibly, or a structured collection error.
#' @export
validate_fmri_collection <- function(x) {
  required <- c("frames", "metadata", "provenance", "schema_version")
  if (!inherits(x, "fmri_collection") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L)) {
    .collection_abort("x is not a valid fmri_collection.")
  }
  fmri_collection(x$frames, metadata = x$metadata, provenance = x$provenance)
  invisible(x)
}

#' Access frames in an fMRI collection
#'
#' @param x An `fmri_collection`.
#' @param id One stable frame ID.
#' @return The named frame list, one frame, or the stable frame IDs.
#' @name collection-accessors
NULL

#' @rdname collection-accessors
#' @export
collection_frames <- function(x) {
  validate_fmri_collection(x)
  x$frames
}

#' @rdname collection-accessors
#' @export
collection_frame <- function(x, id) {
  validate_fmri_collection(x)
  if (!is.character(id) || length(id) != 1L || is.na(id) ||
      !id %in% names(x$frames)) {
    label <- if (length(id)) as.character(id)[[1L]] else ""
    .collection_abort(sprintf("Unknown collection frame '%s'.", label), frame = id)
  }
  x$frames[[id]]
}

#' @rdname collection-accessors
#' @export
collection_ids <- function(x) {
  validate_fmri_collection(x)
  names(x$frames)
}

#' @export
length.fmri_collection <- function(x) length(x$frames)

#' @export
names.fmri_collection <- function(x) names(x$frames)

.normalize_collection_selector <- function(i, ids) {
  if (is.character(i)) {
    if (anyNA(i) || any(!i %in% ids) || anyDuplicated(i)) {
      .collection_abort("Collection frame selector contains unknown or duplicate IDs.")
    }
    return(match(i, ids))
  }
  if (is.logical(i)) {
    if (length(i) != length(ids) || anyNA(i)) {
      .collection_abort("Logical collection selectors must match collection length.")
    }
    return(which(i))
  }
  if (!is.numeric(i) || anyNA(i) || any(i != as.integer(i))) {
    .collection_abort("Collection selectors must contain frame IDs or integer positions.")
  }
  i <- as.integer(i)
  if (any(i < 1L | i > length(ids)) || anyDuplicated(i)) {
    .collection_abort("Collection selector is out of bounds or duplicated.")
  }
  i
}

#' @export
`[.fmri_collection` <- function(x, i, ...) {
  validate_fmri_collection(x)
  if (missing(i)) return(x)
  i <- .normalize_collection_selector(i, names(x$frames))
  if (!length(i)) {
    .collection_abort("An fmri_collection cannot be empty after subsetting.")
  }
  fmri_collection(
    x$frames[i],
    metadata = x$metadata,
    provenance = x$provenance
  )
}

#' @export
`[[.fmri_collection` <- function(x, i, ...) {
  validate_fmri_collection(x)
  if (is.character(i)) return(collection_frame(x, i))
  position <- .normalize_collection_selector(i, names(x$frames))
  if (length(position) != 1L) {
    .collection_abort("Double-bracket collection selection requires one frame.")
  }
  x$frames[[position]]
}

#' Summarize collection feature spaces
#'
#' @param x An `fmri_collection`.
#' @return `collection_space_data()` returns one metadata row per frame;
#'   `collection_common_space()` returns whether every feature space is exactly
#'   compatible with the first.
#' @name collection-spaces
NULL

#' @rdname collection-spaces
#' @export
collection_space_data <- function(x) {
  validate_fmri_collection(x)
  tibble::tibble(
    .frame_id = names(x$frames),
    n_observation = as.integer(vapply(x$frames, nrow, integer(1))),
    n_feature = as.integer(vapply(x$frames, ncol, integer(1))),
    space_type = unname(vapply(
      x$frames, function(frame) class(space(frame))[[1L]], character(1)
    )),
    space_digest = unname(vapply(
      x$frames, function(frame) space_digest(space(frame)), character(1)
    ))
  )
}

#' @rdname collection-spaces
#' @export
collection_common_space <- function(x) {
  validate_fmri_collection(x)
  if (length(x$frames) == 1L) return(TRUE)
  reference <- space(x$frames[[1L]])
  all(vapply(x$frames[-1L], function(frame) {
    isTRUE(compatible_space(reference, space(frame))$compatible)
  }, logical(1)))
}

#' Compute a deterministic collection digest
#'
#' @param x An `fmri_collection`.
#' @return A SHA-256 digest computed without reading numerical arrays.
#' @export
collection_digest <- function(x) {
  validate_fmri_collection(x)
  .canonical_digest(list(
    schema_version = x$schema_version,
    frames = lapply(x$frames, .collection_frame_descriptor),
    metadata = x$metadata,
    provenance = x$provenance
  ))
}

#' @export
print.fmri_collection <- function(x, ...) {
  validate_fmri_collection(x)
  spaces <- if (collection_common_space(x)) "common" else "heterogeneous"
  cat("<fmri_collection>", length(x), "frames\n")
  cat("  feature spaces:", spaces, "\n")
  cat("  assays:", paste(names(assays(x$frames[[1L]])), collapse = ", "), "\n")
  invisible(x)
}
