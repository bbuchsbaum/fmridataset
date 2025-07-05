#' Query fMRI Time Series
#'
#' Core interface for retrieving voxel time series from fMRI datasets.
#'
#' @param dataset An `fmri_dataset` object.
#' @param selector Spatial selector or `NULL` for all voxels.
#' @param timepoints Optional temporal subset or `NULL` for all.
#' @param output Return type - "FmriSeries" (default) or "DelayedMatrix".
#' @param event_window Reserved for future use.
#' @param ... Additional arguments passed to methods.
#'
#' @return Either an `FmriSeries` object or a `DelayedMatrix`.
#' @importFrom methods new
#' @export
fmri_series <- function(dataset, selector = NULL, timepoints = NULL,
                        output = c("FmriSeries", "DelayedMatrix"),
                        event_window = NULL, ...) {
  UseMethod("fmri_series")
}

#' @export
fmri_series.fmri_dataset <- function(dataset, selector = NULL, timepoints = NULL,
                                     output = c("FmriSeries", "DelayedMatrix"),
                                     event_window = NULL, ...) {
  output <- match.arg(output)

  voxel_ind <- resolve_selector(dataset, selector)
  time_ind <- resolve_timepoints(dataset, timepoints)

  da <- as_delayed_array(dataset$backend)
  da <- da[time_ind, voxel_ind, drop = FALSE]

  if (output == "DelayedMatrix") {
    return(da)
  }

  voxel_info <- S4Vectors::DataFrame(voxel = voxel_ind)
  temporal_info <- build_temporal_info_lazy(dataset, time_ind)

  new("FmriSeries",
      da,
      voxel_info = voxel_info,
      temporal_info = temporal_info,
      selection_info = list(selector = selector, timepoints = timepoints),
      dataset_info = list(backend_type = class(dataset$backend)[1]))
}

#' @export
fmri_series.fmri_study_dataset <- function(dataset, selector = NULL, timepoints = NULL,
                                           output = c("FmriSeries", "DelayedMatrix"),
                                           event_window = NULL, ...) {
  output <- match.arg(output)

  voxel_ind <- resolve_selector(dataset, selector)
  time_ind <- resolve_timepoints(dataset, timepoints)

  da <- as_delayed_array(dataset$backend)
  da <- da[time_ind, voxel_ind, drop = FALSE]

  if (output == "DelayedMatrix") {
    return(da)
  }

  voxel_info <- S4Vectors::DataFrame(voxel = voxel_ind)
  temporal_info <- build_temporal_info_lazy(dataset, time_ind)

  new("FmriSeries",
      da,
      voxel_info = voxel_info,
      temporal_info = temporal_info,
      selection_info = list(selector = selector, timepoints = timepoints),
      dataset_info = list(backend_type = class(dataset$backend)[1]))
}
