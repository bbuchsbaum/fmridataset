#' Dataset Methods for fmridataset
#'
#' This file implements methods for dataset objects that delegate to
#' their internal sampling_frame objects for temporal information.
#'
#' All dataset subclasses (matrix_dataset, fmri_mem_dataset,
#' fmri_file_dataset, fmri_study_dataset) inherit from fmri_dataset,
#' so the fmri_dataset methods are dispatched automatically via S3
#' inheritance. Only fmri_dataset-level methods are needed.
#'
#' @name dataset_methods
#' @keywords internal
NULL

#' @rdname get_TR
#' @method get_TR fmri_dataset
#' @export
get_TR.fmri_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths fmri_dataset
#' @export
get_run_lengths.fmri_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname n_runs
#' @method n_runs fmri_dataset
#' @export
n_runs.fmri_dataset <- function(x, ...) {
  n_runs(x$sampling_frame, ...)
}

#' @rdname n_timepoints
#' @method n_timepoints fmri_dataset
#' @export
n_timepoints.fmri_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens fmri_dataset
#' @export
blocklens.fmri_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids fmri_dataset
#' @export
blockids.fmri_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration fmri_dataset
#' @export
get_run_duration.fmri_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration fmri_dataset
#' @export
get_total_duration.fmri_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples fmri_dataset
#' @export
samples.fmri_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

#' Get subject IDs
#' @rdname subject_ids
#' @method subject_ids fmri_study_dataset
#' @export
subject_ids.fmri_study_dataset <- function(x, ...) {
  x$subject_ids
}
