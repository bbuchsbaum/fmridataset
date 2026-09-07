# Frame-level selectors resolve through the package-wide normalization law
# (R/axis-selection.R). Frames own stable IDs, so character selectors are
# resolved here and the stored form is always positional.
.frame_axis_selection <- function(index, n, ids, axis) {
  .normalize_selection(index, n, ids = ids, axis = axis)
}

# Expanded positions for callers that need a vector, such as spatial reads.
.normalize_frame_selector <- function(index, ids, axis) {
  .selection_expand(.frame_axis_selection(index, length(ids), ids, axis))
}

.new_fmri_view <- function(base, observation, feature) {
  structure(
    list(base = base, observation = observation, feature = feature),
    class = c("fmri_view", "fmri_frame")
  )
}

# Subset an axis frame by a selection without re-validating it and without
# touching a select-all axis.
.select_axis_frame <- function(axis, selection) {
  if (.selection_is_all(selection)) axis else .subset_axis_frame(axis, selection)
}

#' @export
`[.fmri_frame` <- function(x, i, j, ..., drop = FALSE) {
  if (!identical(drop, FALSE)) {
    .frame_abort("fmri_frame slicing is always non-dropping.", "fmridataset_error_alignment")
  }
  i <- if (missing(i)) {
    .selection_all(nrow(x))
  } else {
    .frame_axis_selection(i, nrow(x), observation_ids(x), "observation")
  }
  j <- if (missing(j)) {
    .selection_all(ncol(x))
  } else {
    .frame_axis_selection(j, ncol(x), feature_ids(x), "feature")
  }
  .new_fmri_view(x, i, j)
}

#' @export
`[.fmri_view` <- function(x, i, j, ..., drop = FALSE) {
  if (!identical(drop, FALSE)) {
    .frame_abort("fmri_frame slicing is always non-dropping.", "fmridataset_error_alignment")
  }
  # A view of a view composes into one view over the base frame; the composed
  # selection stays compact when both levels are select-all or ranges.
  i <- if (missing(i)) {
    x$observation
  } else {
    .selection_compose(
      .frame_axis_selection(i, nrow(x), observation_ids(x), "observation"),
      x$observation
    )
  }
  j <- if (missing(j)) {
    x$feature
  } else {
    .selection_compose(
      .frame_axis_selection(j, ncol(x), feature_ids(x), "feature"),
      x$feature
    )
  }
  .new_fmri_view(x$base, i, j)
}

#' @rdname frame-accessors
#' @export
dim.fmri_view <- function(x) {
  c(.selection_length(x$observation), .selection_length(x$feature))
}
#' @rdname frame-accessors
#' @export
nrow.fmri_view <- function(x) .selection_length(x$observation)
#' @rdname frame-accessors
#' @export
ncol.fmri_view <- function(x) .selection_length(x$feature)
#' @export
assays.fmri_view <- function(x, ...) {
  observation_digest <- .axis_digest(observation_axis(x))
  feature_digest <- .axis_digest(feature_axis(x))
  out <- lapply(assays(x$base), function(value) {
    value$source <- source_view(
      value$source,
      observations = x$observation,
      features = x$feature
    )
    value$observation_digest <- observation_digest
    value$feature_digest <- feature_digest
    value
  })
  class(out) <- c("aligned_assay_set", "list")
  out
}
#' @export
assay.fmri_view <- function(x, name = active_assay(x), ...) {
  value <- assays(x)[[name]]
  if (is.null(value)) {
    .frame_abort(sprintf("Unknown assay '%s'.", name), "fmridataset_error_alignment")
  }
  value
}
#' @export
active_assay.fmri_view <- function(x, ...) active_assay(x$base)
#' @export
observation_axis.fmri_view <- function(x, ...) {
  .select_axis_frame(observation_axis(x$base), x$observation)
}
#' @export
feature_axis.fmri_view <- function(data, space = NULL, blocks = list(), metadata = list(), ...) {
  out <- .select_axis_frame(feature_axis(data$base), data$feature)
  out$space <- space(data)
  class(out) <- c("spatial_axis_frame", setdiff(class(out), "spatial_axis_frame"))
  out
}
#' @export
observations.fmri_view <- function(x, resolve = FALSE, ...) {
  resolve <- .validate_resolve_flag(resolve)
  if (resolve) .resolved_observation_data(x) else axis_data(observation_axis(x))
}
#' @export
entities.fmri_view <- function(x, ...) entities(x$base)
#' @export
entity.fmri_view <- function(x, name, ...) entity(entities(x), name)
#' @export
relations.fmri_view <- function(x, ...) {
  .restrict_relation_registry(
    relations(x$base),
    observation_ids = observation_ids(x),
    feature_ids = feature_ids(x)
  )
}
#' @export
relation.fmri_view <- function(x, name, ...) relation(relations(x), name)
#' @export
features.fmri_view <- function(x, ...) axis_data(feature_axis(x))
#' @export
observation_ids.fmri_view <- function(x, ...) {
  .selection_subset(observation_ids(x$base), x$observation)
}
#' @export
feature_ids.fmri_view <- function(x, ...) {
  .selection_subset(feature_ids(x$base), x$feature)
}
#' @export
obs_blocks.fmri_view <- function(x, resolve = FALSE, ...) {
  resolve <- .validate_resolve_flag(resolve)
  if (resolve) .resolved_observation_blocks(x) else axis_blocks(observation_axis(x))
}
#' @export
feature_blocks.fmri_view <- function(x, ...) axis_blocks(feature_axis(x))
#' @export
space.fmri_view <- function(x, ...) {
  # Restricting a space to its complete feature axis, in order, is an identity,
  # so a select-all view shares the base space instead of rebuilding it.
  if (.selection_is_all(x$feature)) {
    return(space(x$base))
  }
  restrict_space(space(x$base), .selection_expand(x$feature))
}
#' @export
print.fmri_view <- function(x, ...) {
  cat("<fmri_view>", nrow(x), "observations x", ncol(x), "features\n")
  cat("  base:", nrow(x$base), "x", ncol(x$base), "\n")
  cat("  assays:", paste(names(assays(x)), collapse = ", "), "\n")
  invisible(x)
}

#' Filter frame observations using scalar metadata
#'
#' @param x An `fmri_frame` or view.
#' @param predicate A metadata expression returning one logical value per
#'   observation.
#' @param resolve Whether the predicate may use namespaced entity metadata.
#' @return An `fmri_view`.
#' @examples
#' sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
#'   observations = data.frame(
#'     .obs_id = sprintf("vol-%d", 1:4),
#'     run_id = rep(c("run-1", "run-2"), each = 2)
#'   ),
#'   space = sp
#' )
#' filter_obs(frame, run_id == "run-1")
#' @export
filter_obs <- function(x, predicate, resolve = TRUE) {
  resolve <- .validate_resolve_flag(resolve)
  keep <- rlang::eval_tidy(
    rlang::enquo(predicate),
    data = observations(x, resolve = resolve)
  )
  if (!is.logical(keep) || length(keep) != nrow(x) || anyNA(keep)) {
    .frame_abort("Observation predicate must return non-missing logical values.", "fmridataset_error_alignment")
  }
  x[which(keep), ]
}

#' Select frame features using feature metadata
#'
#' @param x An `fmri_frame` or view.
#' @param predicate A metadata expression returning one logical value per
#'   feature.
#' @return An `fmri_view`.
#' @examples
#' sp <- volume_space(dim = c(2, 2, 2), affine = diag(4))
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(rnorm(4 * n_features(sp)), nrow = 4)),
#'   observations = data.frame(.obs_id = sprintf("vol-%d", 1:4)),
#'   space = sp
#' )
#' select_features(frame, i == 1)
#' @export
select_features <- function(x, predicate) {
  keep <- rlang::eval_tidy(rlang::enquo(predicate), data = features(x))
  if (!is.logical(keep) || length(keep) != ncol(x) || anyNA(keep)) {
    .frame_abort("Feature predicate must return non-missing logical values.", "fmridataset_error_alignment")
  }
  x[, which(keep)]
}
