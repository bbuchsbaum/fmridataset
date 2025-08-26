#' Helpers for fmri_series spatial and temporal selection
#'
#' These functions convert user-facing selectors into numeric indices used by
#' the fmri_series implementation. They are not exported to users directly.
#'
#' @name fmri_series_resolvers
NULL

#' Resolve Spatial Selector
#'
#' @param dataset An `fmri_dataset` object.
#' @param selector Spatial selector or `NULL` for all voxels. Supported types are
#'   integer indices, coordinate matrices with three columns, and logical or ROI
#'   volumes.
#' @return Integer vector of voxel indices within the dataset mask.
#' @keywords internal
#' @export
resolve_selector <- function(dataset, selector) {
  # Handle series_selector objects
  if (inherits(selector, "series_selector")) {
    return(resolve_indices(selector, dataset))
  }

  # Legacy selector handling for backward compatibility
  if (is.null(selector)) {
    mask_vec <- backend_get_mask(dataset$backend)
    return(seq_len(sum(mask_vec)))
  }

  # Handle 3-column coordinate matrices BEFORE numeric check
  if (is.matrix(selector) && ncol(selector) == 3) {
    dims <- backend_get_dims(dataset$backend)$spatial
    # Convert coordinates to linear indices in the full volume
    full_vol_indices <- selector[, 1] + (selector[, 2] - 1) * dims[1] + (selector[, 3] - 1) * dims[1] * dims[2]

    # Get the indices of voxels that are inside the mask
    mask_vec <- backend_get_mask(dataset$backend)
    mask_indices <- which(mask_vec)

    # Map from full volume indices to masked data indices
    final_indices <- match(full_vol_indices, mask_indices)

    # Remove any coordinates that fall outside the mask
    final_indices <- final_indices[!is.na(final_indices)]

    return(as.integer(final_indices))
  }

  if (is.numeric(selector)) {
    return(as.integer(selector))
  }

  # Handle general arrays, ROI volumes, and logical arrays as masks
  if (is.array(selector) || inherits(selector, "ROIVol") || inherits(selector, "LogicalNeuroVol")) {
    ind <- which(as.logical(as.vector(selector)))
    return(as.integer(ind))
  }

  stop_fmridataset(
    fmridataset_error_config,
    message = "Unsupported selector type",
    parameter = "selector",
    value = class(selector)[1]
  )
}

#' Resolve Timepoint Selection
#'
#' @param dataset An `fmri_dataset` object.
#' @param timepoints Integer or logical vector of timepoints, or `NULL` for all.
#' @return Integer vector of timepoint indices.
#' @keywords internal
#' @export
resolve_timepoints <- function(dataset, timepoints) {
  n_time <- backend_get_dims(dataset$backend)$time

  if (is.null(timepoints)) {
    return(seq_len(n_time))
  }

  if (is.logical(timepoints)) {
    if (length(timepoints) != n_time) {
      stop_fmridataset(
        fmridataset_error_config,
        message = "Logical timepoints length must equal number of timepoints",
        parameter = "timepoints",
        value = length(timepoints)
      )
    }
    return(which(timepoints))
  }

  if (is.numeric(timepoints)) {
    return(as.integer(timepoints))
  }

  stop_fmridataset(
    fmridataset_error_config,
    message = "Unsupported timepoints type",
    parameter = "timepoints",
    value = class(timepoints)[1]
  )
}

#' Helper returning all timepoints for a dataset
#'
#' @param dataset An `fmri_dataset` object.
#' @return Integer vector of all valid timepoint indices.
#' @keywords internal
#' @export
all_timepoints <- function(dataset) {
  n_time <- backend_get_dims(dataset$backend)$time
  seq_len(n_time)
}
