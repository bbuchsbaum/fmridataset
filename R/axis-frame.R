.frame_abort <- function(message, class, ...) {
  stop(fmridataset_error(message, class = class, ...))
}

.new_uuid <- function(prefix, n = 1L) {
  paste0("ephemeral-", prefix, "-", uuid::UUIDgenerate(n = as.integer(n)))
}

.id_policy <- function(policy, namespace = NULL, keys = character()) {
  structure(
    list(
      policy = policy,
      namespace = namespace,
      keys = as.character(keys),
      durable = !identical(policy, "ephemeral"),
      schema_version = 1L
    ),
    class = c("fmri_id_policy", "list")
  )
}

.normalize_id_policy <- function(id_policy, id, data, axis, id_col,
                                 id_keys, id_namespace) {
  id_policy <- match.arg(id_policy, c("require", "deterministic", "ephemeral"))
  prefix <- .axis_id_prefix(axis)
  if (identical(id_policy, "require")) {
    if (is.null(id) && !id_col %in% names(data)) {
      .identity_abort(
        sprintf(
          "%s ID policy 'require' requires supplied IDs in `id` or `%s`.",
          axis, id_col
        ),
        field = id_col, policy = id_policy
      )
    }
    value <- id %||% data[[id_col]]
    return(list(ids = value, descriptor = .id_policy("require")))
  }
  if (identical(id_policy, "ephemeral")) {
    if (!is.null(id) || id_col %in% names(data)) {
      .identity_abort(
        "Ephemeral ID policy generates its own visibly marked IDs; do not supply IDs.",
        field = id_col, policy = id_policy
      )
    }
    return(list(
      ids = .new_uuid(prefix, nrow(data)),
      descriptor = .id_policy("ephemeral")
    ))
  }
  if (!.is_one_string(id_namespace)) {
    .identity_abort(
      "Deterministic ID policy requires one non-empty `id_namespace`.",
      field = "id_namespace", policy = id_policy
    )
  }
  if (!is.character(id_keys) || !length(id_keys) || anyNA(id_keys) ||
      any(!nzchar(id_keys)) || anyDuplicated(id_keys) ||
      !all(id_keys %in% names(data))) {
    .identity_abort(
      "Deterministic ID policy requires unique `id_keys` present in axis data.",
      field = "id_keys", policy = id_policy
    )
  }
  key_data <- as.data.frame(data[id_keys], stringsAsFactors = FALSE)
  if (any(vapply(key_data, is.list, logical(1))) || anyNA(key_data)) {
    .identity_abort(
      "Deterministic ID keys must be scalar and non-missing.",
      field = "id_keys", policy = id_policy
    )
  }
  signatures <- vapply(seq_len(nrow(key_data)), function(i) {
    canonical_sha256(list(
      namespace = id_namespace,
      axis = axis,
      keys = id_keys,
      values = lapply(key_data[i, , drop = FALSE], function(value) value[[1L]])
    ))
  }, character(1))
  generated <- paste0(prefix, "-", signatures)
  if (anyDuplicated(generated)) {
    .identity_abort(
      "Deterministic ID keys must uniquely identify every axis element.",
      field = "id_keys", policy = id_policy
    )
  }
  supplied <- id %||% if (id_col %in% names(data)) data[[id_col]] else NULL
  if (!is.null(supplied) && !identical(as.character(supplied), generated)) {
    .identity_abort(
      "Supplied IDs do not match IDs reconstructed from deterministic keys.",
      field = id_col, policy = id_policy
    )
  }
  list(
    ids = generated,
    descriptor = .id_policy("deterministic", id_namespace, id_keys)
  )
}

.validate_stable_ids <- function(ids, what = "axis") {
  .assert_stable_keys(ids, .alignment_abort, what = what)
}

.axis_id_column <- function(axis) {
  switch(axis,
    observation = ".obs_id",
    feature = ".feature_id",
    entity = ".entity_id",
    component = ".component_id",
    paste0(".", axis, "_id")
  )
}

.axis_id_prefix <- function(axis) {
  switch(axis,
    observation = "obs",
    feature = "feature",
    entity = "entity",
    component = "component",
    axis
  )
}

#' Construct an axis-aligned multivariate block
#'
#' A block is two-dimensional: rows are the elements of the owning axis and
#' columns are named components. Arrays with more than two dimensions are
#' rejected because a trailing axis without typed metadata would be an
#' anonymous semantic dimension; represent higher-order structure as named
#' components, as several blocks, or as an assay.
#'
#' @param data A matrix, two-dimensional lazy array, or serializable array
#'   source. Its first dimension is aligned with the owning axis and its second
#'   dimension indexes `components`.
#' @param components Component metadata. The `.component_id` column is
#'   generated when absent.
#' @param role Semantic role such as `"continuous"`, `"confound"`, or
#'   `"embedding"`.
#' @param units Optional units applying to the block as a whole.
#' @param metadata Additional serializable metadata.
#' @return An `axis_block`.
#' @export
axis_block <- function(data, components = NULL, role = "continuous",
                       units = NULL, metadata = list()) {
  .assert_block_shape(data)
  .assert_no_runtime_state(
    metadata, .alignment_abort,
    "Axis block metadata cannot contain runtime functions, environments, or external pointers.",
    field = "metadata"
  )
  n_component <- .block_shape(data)[[2L]]
  if (is.null(components)) {
    components <- data.frame(
      .component_id = sprintf("component-%06d", seq_len(n_component)),
      stringsAsFactors = FALSE
    )
  }
  components <- tibble::as_tibble(components)
  if (nrow(components) != n_component) {
    .frame_abort(
      "Component metadata must have one row per second-axis component.",
      "fmridataset_error_alignment",
      expected = n_component,
      actual = nrow(components)
    )
  }
  if (!".component_id" %in% names(components)) {
    components$.component_id <- sprintf("component-%06d", seq_len(n_component))
  }
  components$.component_id <- .validate_stable_ids(
    components$.component_id,
    "component"
  )
  structure(
    list(
      data = data,
      components = components,
      role = as.character(role)[1L],
      units = units,
      metadata = metadata
    ),
    class = "axis_block"
  )
}

#' @param x An `axis_block`.
#' @rdname axis_block
#' @export
axis_block_data <- function(x) x$data

#' @rdname axis_block
#' @export
block_components <- function(x) x$components

#' @rdname axis_block
#' @export
block_component_ids <- function(x) x$components$.component_id

#' Construct an annotated axis
#'
#' @param data A data frame with one row per axis element.
#' @param blocks Named `axis_block` objects aligned on their first dimension.
#' @param id Optional stable IDs.
#' @param axis Axis role. Observation is the public default.
#' @param id_col Name of the ID column.
#' @param metadata Additional serializable metadata.
#' @param id_policy ID policy. `"require"` accepts only supplied durable IDs;
#'   `"deterministic"` derives durable IDs from `id_keys` and `id_namespace`;
#'   `"ephemeral"` creates visibly marked session-only IDs that cannot be
#'   persisted or used for certified semantic identity.
#' @param id_keys Columns that uniquely identify rows under deterministic policy.
#' @param id_namespace Stable namespace under deterministic policy.
#' @return An `axis_frame`.
#' @export
axis_frame <- function(data, blocks = list(), id = NULL,
                       axis = c("observation", "feature", "entity", "component"),
                       id_col = NULL, metadata = list(),
                       id_policy = c("require", "deterministic", "ephemeral"),
                       id_keys = NULL, id_namespace = NULL) {
  axis <- match.arg(axis)
  data <- tibble::as_tibble(data)
  id_col <- id_col %||% .axis_id_column(axis)

  policy_value <- .normalize_id_policy(
    id_policy, id, data, axis, id_col, id_keys, id_namespace
  )
  id <- policy_value$ids
  id <- .validate_stable_ids(as.character(id), axis)
  if (length(id) != nrow(data)) {
    .frame_abort(
      sprintf("%s IDs must have one value per metadata row.", axis),
      "fmridataset_error_alignment"
    )
  }
  data[[id_col]] <- id
  data <- data[c(id_col, setdiff(names(data), id_col))]

  .assert_aligned_blocks(blocks, nrow(data), what = "Axis")
  .assert_no_runtime_state(
    metadata, .alignment_abort,
    "Axis metadata cannot contain runtime functions, environments, or external pointers.",
    field = "metadata"
  )

  structure(
    list(
      data = data,
      blocks = blocks,
      id_col = id_col,
      axis = axis,
      id_policy = policy_value$descriptor,
      metadata = metadata,
      # Axis frames are immutable values, so the digest of their IDs is
      # computed once here. Every consumer that keys on the axis (assay
      # descriptors, views, manifests, explain()) reads it back instead of
      # re-hashing an axis as long as a feature space on each access.
      id_digest = .axis_id_digest(id)
    ),
    class = "axis_frame"
  )
}

.axis_id_digest <- function(ids) {
  list(n = length(ids), sha256 = .canonical_digest(ids))
}

#' @param x An `axis_frame`.
#' @rdname axis_frame
#' @export
axis_data <- function(x) x$data

#' @rdname axis_frame
#' @export
axis_blocks <- function(x) x$blocks

#' @rdname axis_frame
#' @export
axis_ids <- function(x) UseMethod("axis_ids")

#' @export
axis_ids.axis_frame <- function(x) x$data[[x$id_col]]

#' Inspect axis ID durability
#'
#' @param x An axis, feature space, frame, or view.
#' @return `axis_id_policy()` returns the versioned ID-policy descriptor;
#'   `ids_are_durable()` returns one logical value.
#' @name id-policy
NULL

#' @rdname id-policy
#' @export
axis_id_policy <- function(x) UseMethod("axis_id_policy")

#' @export
axis_id_policy.axis_frame <- function(x) {
  x$id_policy %||% .id_policy("require")
}

#' @export
axis_id_policy.feature_space <- function(x) .id_policy("require")

#' @rdname id-policy
#' @export
ids_are_durable <- function(x) UseMethod("ids_are_durable")

#' @export
ids_are_durable.axis_frame <- function(x) isTRUE(axis_id_policy(x)$durable)

#' @export
ids_are_durable.feature_space <- function(x) isTRUE(axis_id_policy(x)$durable)

#' @export
length.axis_frame <- function(x) nrow(x$data)

#' @export
`[.axis_frame` <- function(x, i, ...) {
  n <- nrow(x$data)
  selection <- if (missing(i)) {
    .selection_all(n)
  } else {
    .normalize_selection(i, n, ids = x$data[[x$id_col]], axis = x$axis %||% "axis")
  }
  .subset_axis_frame(x, selection)
}

# Subset an axis frame by an already-normalized selection.
.subset_axis_frame <- function(x, selection) {
  rows <- .subset_keyed_rows(x$data, x$blocks, .selection_expand(selection))
  data <- rows$data
  out <- axis_frame(
    data,
    blocks = rows$blocks,
    id = data[[x$id_col]],
    axis = x$axis,
    id_col = x$id_col,
    metadata = x$metadata,
    id_policy = "require"
  )
  out$id_policy <- axis_id_policy(x)
  out
}

#' Construct a spatial feature axis
#'
#' @param data Feature metadata or an `fmri_frame` when used as an accessor.
#' @param space A `FeatureSpace`.
#' @param blocks Feature-aligned blocks.
#' @param metadata Additional metadata.
#' @param ... Additional arguments for methods.
#' @return A feature `axis_frame` carrying its space.
#' @export
feature_axis <- function(data, space = NULL, blocks = list(), metadata = list(), ...) {
  UseMethod("feature_axis")
}

#' @export
feature_axis.default <- function(data, space = NULL, blocks = list(), metadata = list(), ...) {
  if (is.null(space) || !inherits(space, "feature_space")) {
    .frame_abort("A feature axis requires a FeatureSpace.", "fmridataset_error_space_mismatch")
  }
  data <- tibble::as_tibble(data)
  ids <- feature_ids(space)
  if (nrow(data) != length(ids)) {
    .frame_abort(
      "Feature metadata must have one row per feature-space element.",
      "fmridataset_error_alignment"
    )
  }
  out <- axis_frame(
    data,
    blocks = blocks,
    id = if (".feature_id" %in% names(data)) data$.feature_id else ids,
    axis = "feature",
    id_col = ".feature_id",
    metadata = metadata,
    id_policy = "require"
  )
  if (!identical(axis_ids(out), ids)) {
    .frame_abort(
      "Feature metadata IDs must exactly match the FeatureSpace IDs.",
      "fmridataset_error_alignment"
    )
  }
  out$space <- space
  out$id_policy <- axis_id_policy(space)
  class(out) <- c("spatial_axis_frame", class(out))
  out
}

#' @export
feature_axis.fmri_frame <- function(data, space = NULL, blocks = list(), metadata = list(), ...) data$features
