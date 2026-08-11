.normalize_frame_selector <- function(index, ids, axis) {
  n <- length(ids)
  if (missing(index) || is.null(index)) return(seq_len(n))
  if (is.character(index)) {
    if (anyNA(index) || anyDuplicated(index)) {
      .frame_abort(sprintf("%s ID selectors must be unique and non-missing.", axis), "fmridataset_error_alignment")
    }
    out <- match(index, ids)
    if (anyNA(out)) {
      .frame_abort(sprintf("Unknown %s ID in selector.", axis), "fmridataset_error_alignment")
    }
    return(as.integer(out))
  }
  if (is.logical(index)) {
    if (length(index) != n || anyNA(index)) {
      .frame_abort(sprintf("Logical %s selectors must match the axis and contain no NA.", axis), "fmridataset_error_alignment")
    }
    return(which(index))
  }
  if (!is.numeric(index)) {
    .frame_abort(sprintf("Unsupported %s selector type.", axis), "fmridataset_error_alignment")
  }
  if (anyNA(index) || any(index != as.integer(index))) {
    .frame_abort(sprintf("%s positions must be non-missing integers.", axis), "fmridataset_error_alignment")
  }
  index <- as.integer(index)
  if (any(index < 0L)) {
    if (any(index > 0L)) {
      .frame_abort("Positive and negative selectors cannot be mixed.", "fmridataset_error_alignment")
    }
    index <- setdiff(seq_len(n), abs(index[index < 0L]))
  } else {
    index <- index[index != 0L]
  }
  if (any(index < 1L | index > n) || anyDuplicated(index)) {
    .frame_abort(sprintf("%s selector is out of bounds or duplicated.", axis), "fmridataset_error_alignment")
  }
  index
}

.new_fmri_view <- function(base, observation_index, feature_index) {
  structure(
    list(
      base = base,
      observation_index = as.integer(observation_index),
      feature_index = as.integer(feature_index)
    ),
    class = c("fmri_view", "fmri_frame", "fmri_dataset")
  )
}

#' @export
`[.fmri_frame` <- function(x, i, j, ..., drop = FALSE) {
  if (!identical(drop, FALSE)) {
    .frame_abort("fmri_frame slicing is always non-dropping.", "fmridataset_error_alignment")
  }
  i <- if (missing(i)) seq_len(nrow(x)) else .normalize_frame_selector(i, observation_ids(x), "observation")
  j <- if (missing(j)) seq_len(ncol(x)) else .normalize_frame_selector(j, feature_ids(x), "feature")
  .new_fmri_view(x, i, j)
}

#' @export
`[.fmri_view` <- function(x, i, j, ..., drop = FALSE) {
  if (!identical(drop, FALSE)) {
    .frame_abort("fmri_frame slicing is always non-dropping.", "fmridataset_error_alignment")
  }
  i <- if (missing(i)) seq_along(x$observation_index) else .normalize_frame_selector(i, observation_ids(x), "observation")
  j <- if (missing(j)) seq_along(x$feature_index) else .normalize_frame_selector(j, feature_ids(x), "feature")
  .new_fmri_view(x$base, x$observation_index[i], x$feature_index[j])
}

#' @rdname frame-accessors
#' @export
dim.fmri_view <- function(x) c(length(x$observation_index), length(x$feature_index))
#' @rdname frame-accessors
#' @export
nrow.fmri_view <- function(x) length(x$observation_index)
#' @rdname frame-accessors
#' @export
ncol.fmri_view <- function(x) length(x$feature_index)
#' @export
assays.fmri_view <- function(x, ...) assays(x$base)
#' @export
assay.fmri_view <- function(x, name = active_assay(x), ...) assay(x$base, name)
#' @export
active_assay.fmri_view <- function(x, ...) active_assay(x$base)
#' @export
observation_axis.fmri_view <- function(x, ...) observation_axis(x$base)[x$observation_index]
#' @export
feature_axis.fmri_view <- function(data, space = NULL, blocks = list(), metadata = list(), ...) {
  out <- feature_axis(data$base)[data$feature_index]
  out$space <- restrict_space(space(data$base), data$feature_index)
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
observation_ids.fmri_view <- function(x, ...) observation_ids(x$base)[x$observation_index]
#' @export
feature_ids.fmri_view <- function(x, ...) feature_ids(x$base)[x$feature_index]
#' @export
obs_blocks.fmri_view <- function(x, resolve = FALSE, ...) {
  resolve <- .validate_resolve_flag(resolve)
  if (resolve) .resolved_observation_blocks(x) else axis_blocks(observation_axis(x))
}
#' @export
feature_blocks.fmri_view <- function(x, ...) axis_blocks(feature_axis(x))
#' @export
space.fmri_view <- function(x, ...) restrict_space(space(x$base), x$feature_index)
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
#' @export
select_features <- function(x, predicate) {
  keep <- rlang::eval_tidy(rlang::enquo(predicate), data = features(x))
  if (!is.logical(keep) || length(keep) != ncol(x) || anyNA(keep)) {
    .frame_abort("Feature predicate must return non-missing logical values.", "fmridataset_error_alignment")
  }
  x[, which(keep)]
}
