.validity_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_validity", ...)
}

.logical_mask_matrix <- function(x, n_feature) {
  if (methods::is(x, "Matrix")) x <- as.matrix(x)
  if (is.vector(x) && is.logical(x)) x <- matrix(x, nrow = 1L)
  if (!is.matrix(x) || !is.logical(x) || ncol(x) != n_feature || anyNA(x)) {
    .validity_abort(
      "masks must be a non-missing logical matrix with one column per feature.",
      field = "masks", expected_features = n_feature
    )
  }
  x
}

.pack_mask_rows <- function(x) {
  bytes <- as.integer(ceiling(ncol(x) / 8))
  packed <- lapply(seq_len(nrow(x)), function(i) {
    if (!bytes) return(raw())
    padded <- c(as.raw(x[i, ]), raw(bytes * 8L - ncol(x)))
    packBits(padded, type = "raw")
  })
  list(bits = as.raw(unlist(packed, use.names = FALSE)), bytes = bytes)
}

.unpack_mask_rows <- function(bits, n_mask, n_feature, bytes) {
  if (!n_mask) return(matrix(logical(), nrow = 0L, ncol = n_feature))
  out <- matrix(FALSE, nrow = n_mask, ncol = n_feature)
  for (i in seq_len(n_mask)) {
    at <- (i - 1L) * bytes + seq_len(bytes)
    out[i, ] <- as.logical(rawToBits(bits[at])[seq_len(n_feature)])
  }
  out
}

#' Construct a deduplicated, bit-packed bank of feature masks
#'
#' @param masks Logical mask-by-feature matrix. Duplicate rows are stored once.
#' @param space Exact feature space addressed by mask columns.
#' @param metadata Serializable metadata.
#' @return A serializable `mask_bank`.
#' @export
mask_bank <- function(masks, space, metadata = list()) {
  if (!inherits(space, "feature_space")) {
    .validity_abort("space must be a feature_space.", field = "space")
  }
  masks <- .logical_mask_matrix(masks, n_features(space))
  if (!nrow(masks)) {
    .validity_abort("masks must contain at least one row.", field = "masks")
  }
  keys <- vapply(seq_len(nrow(masks)), function(i) {
    row <- masks[i, ]
    bytes <- as.integer(ceiling(length(row) / 8))
    if (!bytes) {
      return(digest::digest(raw(), algo = "sha256", serialize = TRUE))
    }
    padded <- c(as.raw(row), raw(bytes * 8L - length(row)))
    digest::digest(packBits(padded, type = "raw"),
                   algo = "sha256", serialize = TRUE)
  }, character(1))
  unique_at <- match(unique(keys), keys)
  unique_masks <- masks[unique_at, , drop = FALSE]
  assignment <- match(keys, keys[unique_at])
  packed <- .pack_mask_rows(unique_masks)
  mask_ids <- paste0("mask-", substr(keys[unique_at], 1L, 16L))
  out <- structure(
    list(
      space = space,
      mask_ids = mask_ids,
      bits = packed$bits,
      n_features = as.integer(n_features(space)),
      bytes_per_mask = packed$bytes,
      assignment = as.integer(assignment),
      metadata = metadata,
      schema_version = 1L
    ),
    class = "mask_bank"
  )
  validate_mask_bank(out)
  out
}

#' Validate and inspect a mask bank
#'
#' @param x A `mask_bank` or validity descriptor.
#' @param mask Optional mask ID or integer position.
#' @return The validated bank, number of masks, unpacked logical masks, or
#'   deterministic digest.
#' @name mask-bank-accessors
NULL

#' @rdname mask-bank-accessors
#' @export
validate_mask_bank <- function(x) {
  required <- c(
    "space", "mask_ids", "bits", "n_features", "bytes_per_mask",
    "assignment", "metadata", "schema_version"
  )
  if (!inherits(x, "mask_bank") || !identical(names(unclass(x)), required) ||
      !identical(x$schema_version, 1L) ||
      !inherits(x$space, "feature_space") ||
      !identical(x$n_features, as.integer(n_features(x$space))) ||
      !identical(x$bytes_per_mask, as.integer(ceiling(x$n_features / 8))) ||
      !is.raw(x$bits) ||
      length(x$bits) != length(x$mask_ids) * x$bytes_per_mask ||
      anyNA(x$mask_ids) || any(!nzchar(x$mask_ids)) ||
      anyDuplicated(x$mask_ids) ||
      !is.integer(x$assignment) ||
      any(x$assignment < 1L | x$assignment > length(x$mask_ids)) ||
      !is.list(x$metadata) || .source_contains_runtime_state(x)) {
    .validity_abort("x is not a valid mask_bank descriptor.")
  }
  invisible(x)
}

#' @rdname mask-bank-accessors
#' @export
n_masks <- function(x) {
  x <- validity_mask_bank(x)
  length(x$mask_ids)
}

#' @rdname mask-bank-accessors
#' @export
mask_values <- function(x, mask = NULL) {
  x <- validity_mask_bank(x)
  values <- .unpack_mask_rows(
    x$bits, length(x$mask_ids), x$n_features, x$bytes_per_mask
  )
  if (is.null(mask)) return(values)
  index <- if (is.character(mask)) match(mask, x$mask_ids) else as.integer(mask)
  if (anyNA(index) || any(index < 1L | index > nrow(values))) {
    .validity_abort("Unknown mask selector.", field = "mask")
  }
  values[index, , drop = FALSE]
}

#' @rdname mask-bank-accessors
#' @export
mask_bank_digest <- function(x) {
  x <- validity_mask_bank(x)
  .canonical_digest(list(
    type = "mask_bank", schema_version = x$schema_version,
    space = space_digest(x$space), feature_ids = feature_ids(x$space),
    mask_ids = x$mask_ids, bits = x$bits, metadata = x$metadata
  ))
}

#' Describe compressed entity-by-feature validity
#'
#' @param entity Entity registry name.
#' @param entity_ids Stable entity IDs aligned to mask rows.
#' @param masks Logical entity-by-feature matrix or a `mask_bank` whose original
#'   row assignments have the same length as `entity_ids`.
#' @param space Exact feature space addressed by validity columns.
#' @param metadata Serializable relation metadata.
#' @return An `entity_feature_validity` relation descriptor.
#' @export
entity_feature_validity <- function(entity, entity_ids, masks, space,
                                    metadata = list()) {
  entity <- .one_relation_string(entity, "entity")
  entity <- sub("^entity:", "", entity)
  entity_ids <- .validate_stable_ids(as.character(entity_ids), "validity entity")
  if (inherits(masks, "mask_bank")) {
    bank <- masks
    validate_mask_bank(bank)
    assert_compatible_space(bank$space, space)
  } else {
    bank <- mask_bank(masks, space)
  }
  if (length(bank$assignment) != length(entity_ids)) {
    .validity_abort("Validity masks require one assignment per entity.",
                    field = "assignment")
  }
  out <- structure(
    list(
      type = "entity_feature_validity",
      entity = entity,
      entity_ids = entity_ids,
      mask_id = bank$mask_ids[bank$assignment],
      bank = bank,
      metadata = metadata,
      schema_version = 1L
    ),
    class = c("entity_feature_validity", "fmri_relation")
  )
  validate_entity_feature_validity(out)
  out
}

#' Validate and inspect entity-feature validity
#'
#' @param x An `entity_feature_validity`, frame, or view.
#' @param name Relation name when `x` is a frame or view.
#' @return The validated descriptor, entity name/IDs, mask bank, feature space,
#'   or expanded entity-by-feature logical matrix.
#' @name feature-validity-accessors
NULL

#' @rdname feature-validity-accessors
#' @export
validate_entity_feature_validity <- function(x) {
  required <- c(
    "type", "entity", "entity_ids", "mask_id", "bank", "metadata",
    "schema_version"
  )
  if (!inherits(x, "entity_feature_validity") ||
      !identical(names(unclass(x)), required) ||
      !identical(x$type, "entity_feature_validity") ||
      !identical(x$schema_version, 1L)) {
    .validity_abort("x is not a valid entity_feature_validity descriptor.")
  }
  .one_relation_string(x$entity, "entity")
  .validate_stable_ids(x$entity_ids, "validity entity")
  validate_mask_bank(x$bank)
  if (length(x$mask_id) != length(x$entity_ids) || anyNA(x$mask_id) ||
      any(!x$mask_id %in% x$bank$mask_ids) || !is.list(x$metadata) ||
      .source_contains_runtime_state(x)) {
    .validity_abort("Validity assignments are malformed.", field = "mask_id")
  }
  invisible(x)
}

.validity_relation <- function(x, name = NULL) {
  if (inherits(x, "entity_feature_validity")) return(x)
  if (!inherits(x, "fmri_frame")) {
    .validity_abort("x must be a validity descriptor, frame, or view.")
  }
  candidates <- relation_names(x)[vapply(
    relations(x), inherits, logical(1), "entity_feature_validity"
  )]
  if (is.null(name)) {
    if (length(candidates) != 1L) {
      .validity_abort("name is required unless the frame has one validity relation.")
    }
    name <- candidates[[1L]]
  }
  value <- relation(x, name)
  if (!inherits(value, "entity_feature_validity")) {
    .validity_abort(sprintf("Relation '%s' is not entity-feature validity.", name))
  }
  value
}

#' @rdname feature-validity-accessors
#' @export
validity_entity <- function(x, name = NULL) .validity_relation(x, name)$entity
#' @rdname feature-validity-accessors
#' @export
validity_entity_ids <- function(x, name = NULL) .validity_relation(x, name)$entity_ids
#' @rdname feature-validity-accessors
#' @export
validity_mask_bank <- function(x, name = NULL) {
  if (inherits(x, "mask_bank")) {
    validate_mask_bank(x)
    return(x)
  }
  .validity_relation(x, name)$bank
}
#' @rdname feature-validity-accessors
#' @export
validity_space <- function(x, name = NULL) validity_mask_bank(x, name)$space
#' @rdname feature-validity-accessors
#' @export
validity_matrix <- function(x, name = NULL) {
  value <- .validity_relation(x, name)
  bank <- value$bank
  masks <- mask_values(bank)
  masks[match(value$mask_id, bank$mask_ids), , drop = FALSE]
}

.restrict_entity_feature_validity <- function(x, feature_ids_value) {
  index <- match(feature_ids_value, feature_ids(x$bank$space))
  if (anyNA(index)) .validity_abort("Validity restriction contains unknown features.")
  entity_feature_validity(
    entity = x$entity,
    entity_ids = x$entity_ids,
    masks = validity_matrix(x)[, index, drop = FALSE],
    space = restrict_space(x$bank$space, index),
    metadata = x$metadata
  )
}

#' Resolve validity onto frame observations
#'
#' @param x An `fmri_frame` or view.
#' @param name Validity relation name.
#' @return Observation-by-feature logical validity matrix.
#' @export
observation_validity <- function(x, name = NULL) {
  value <- .validity_relation(x, name)
  maps <- .resolve_entity_maps(x)
  mapped <- maps[[paste0("entity:", value$entity)]]
  if (is.null(mapped)) {
    .validity_abort(
      sprintf("Observations have no key-relation path to entity '%s'.", value$entity),
      field = "entity"
    )
  }
  positions <- match(mapped, value$entity_ids)
  if (anyNA(positions)) {
    .validity_abort("Observation entity validity is incomplete.", field = "entity")
  }
  validity_matrix(value)[positions, , drop = FALSE]
}

#' Summarize feature coverage without imposing an analysis policy
#'
#' @param x An `fmri_frame` or view.
#' @param name Validity relation name.
#' @param domain Weight unique entities or frame observations.
#' @return Named fraction-valid vector over frame features.
#' @export
validity_coverage <- function(x, name = NULL,
                              domain = c("entity", "observation")) {
  domain <- match.arg(domain)
  values <- if (identical(domain, "entity")) {
    validity_matrix(x, name)
  } else {
    observation_validity(x, name)
  }
  out <- colMeans(values)
  names(out) <- feature_ids(x)
  out
}

#' Lazily mask invalid observation-feature cells with missing values
#'
#' @param source Observation-by-feature array source.
#' @param observation_mask_id One mask-bank ID per source row.
#' @param bank Compatible `mask_bank`.
#' @return A serializable `validity_masked_source`.
#' @export
validity_masked_source <- function(source, observation_mask_id, bank) {
  source <- as_array_source(source)
  validate_array_source(source)
  validate_mask_bank(bank)
  if (source_shape(source)[2L] != bank$n_features ||
      length(observation_mask_id) != source_shape(source)[1L] ||
      anyNA(observation_mask_id) ||
      any(!observation_mask_id %in% bank$mask_ids)) {
    .validity_abort("Source, mask assignments, and bank are not aligned.")
  }
  out <- structure(
    list(
      source = source,
      observation_mask_id = as.character(observation_mask_id),
      bank = bank,
      shape = source_shape(source),
      chunks = source_chunks(source),
      schema_version = 1L
    ),
    class = c("validity_masked_source", "array_source")
  )
  validate_array_source(out)
  out
}

#' @export
source_shape.validity_masked_source <- function(x, ...) as.integer(x$shape)
#' @export
source_dtype.validity_masked_source <- function(x, ...) {
  if (grepl("^complex", source_dtype(x$source))) "complex128" else "float64"
}
#' @export
source_chunks.validity_masked_source <- function(x, ...) as.integer(x$chunks)
#' @export
source_capabilities.validity_masked_source <- function(x, ...) {
  c("row_slice", "column_slice", "block_slice", "serializable")
}
#' @export
source_fingerprint.validity_masked_source <- function(x, ...) {
  .canonical_digest(list(
    type = "validity_masked_source", source = source_fingerprint(x$source),
    assignments = x$observation_mask_id, bank = mask_bank_digest(x$bank),
    schema_version = x$schema_version
  ))
}
#' @export
source_open.validity_masked_source <- function(x, ...) {
  structure(list(source = x),
            class = c("validity_masked_source_handle", "array_source_handle"))
}
#' @export
source_read.validity_masked_source <- function(x, observations = NULL,
                                                features = NULL, ...) {
  observations <- .normalize_source_index(observations, x$shape[1L])
  features <- .normalize_source_index(features, x$shape[2L])
  values <- source_read(x$source, observations, features, ...)
  if (!length(observations) || !length(features)) return(values)
  masks <- mask_values(x$bank)[
    match(x$observation_mask_id[observations], x$bank$mask_ids),
    features, drop = FALSE
  ]
  values[!masks] <- NA
  values
}
#' @export
source_read_native.validity_masked_source <- function(x, observations = NULL, ...) {
  .frame_abort(
    "validity_masked_source has no native spatial read path.",
    "fmridataset_error_backend_io", operation = "native_read"
  )
}
#' @export
source_close.validity_masked_source <- function(x, ...) invisible(TRUE)

#' Apply one validity relation lazily to frame assays
#'
#' @param x An `fmri_frame` or view.
#' @param name Validity relation name.
#' @param assays Assay names to mask. Defaults to all assays.
#' @return A new frame with lazy `NA` masking and derivation provenance.
#' @export
apply_feature_validity <- function(x, name = NULL, assays = NULL) {
  value <- .validity_relation(x, name)
  if (is.null(assays)) assays <- names(fmridataset::assays(x))
  if (!is.character(assays) || anyNA(assays) || anyDuplicated(assays) ||
      any(!assays %in% names(fmridataset::assays(x)))) {
    .validity_abort("assays contains unknown or duplicate names.", field = "assays")
  }
  observed <- observation_validity(x, name)
  bank <- mask_bank(observed, space(x))
  mask_ids <- bank$mask_ids[bank$assignment]
  values <- lapply(names(fmridataset::assays(x)), function(assay_name) {
    descriptor <- assay(x, assay_name)
    source <- .frame_assay_source(x, assay_name)
    if (assay_name %in% assays) {
      source <- validity_masked_source(source, mask_ids, bank)
    }
    .mapped_aligned_assay(descriptor, source)
  })
  names(values) <- names(fmridataset::assays(x))
  graph <- .as_provenance_graph(x$provenance %||% x$base$provenance)
  graph <- append_provenance(graph, provenance_record(
    "apply_feature_validity", parents = provenance_tips(graph),
    inputs = list(
      relation = name %||% relation_names(x)[vapply(
        relations(x), inherits, logical(1), "entity_feature_validity"
      )],
      validity = mask_bank_digest(value$bank)
    ),
    parameters = list(assays = assays),
    outputs = list(space = space_digest(space(x)))
  ))
  fmri_frame(
    assays = values, observations = observation_axis(x),
    features = feature_axis(x), entities = entities(x), relations = relations(x),
    tables = x$tables %||% x$base$tables, active_assay = active_assay(x),
    metadata = x$metadata %||% x$base$metadata, provenance = graph
  )
}
