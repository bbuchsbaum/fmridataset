#' Dataset Methods for fmridataset
#'
#' This file implements methods for dataset objects that delegate to
#' their internal sampling_frame objects for temporal information.
#'
#' @name dataset_methods
#' @keywords internal
NULL

# Define dataset classes that have sampling_frame delegation
.dataset_classes <- c("matrix_dataset", "fmri_dataset", "fmri_mem_dataset", "fmri_file_dataset")

# Define methods that delegate to sampling_frame
.delegated_methods <- c(
  "get_TR", "get_run_lengths", "n_runs", "n_timepoints",
  "blocklens", "blockids", "get_run_duration",
  "get_total_duration", "samples"
)

# Note: Dynamic registration removed - methods are explicitly defined below

# Export the methods for documentation
#' @rdname get_TR
#' @method get_TR matrix_dataset
#' @export
get_TR.matrix_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_TR
#' @method get_TR fmri_dataset
#' @export
get_TR.fmri_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_TR
#' @method get_TR fmri_mem_dataset
#' @export
get_TR.fmri_mem_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_TR
#' @method get_TR fmri_file_dataset
#' @export
get_TR.fmri_file_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths matrix_dataset
#' @export
get_run_lengths.matrix_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths fmri_dataset
#' @export
get_run_lengths.fmri_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths fmri_mem_dataset
#' @export
get_run_lengths.fmri_mem_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths fmri_file_dataset
#' @export
get_run_lengths.fmri_file_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname n_runs
#' @method n_runs matrix_dataset
#' @export
n_runs.matrix_dataset <- function(x, ...) {
  n_runs(x$sampling_frame, ...)
}

#' @rdname n_runs
#' @method n_runs fmri_dataset
#' @export
n_runs.fmri_dataset <- function(x, ...) {
  n_runs(x$sampling_frame, ...)
}

#' @rdname n_runs
#' @method n_runs fmri_mem_dataset
#' @export
n_runs.fmri_mem_dataset <- function(x, ...) {
  n_runs(x$sampling_frame, ...)
}

#' @rdname n_runs
#' @method n_runs fmri_file_dataset
#' @export
n_runs.fmri_file_dataset <- function(x, ...) {
  n_runs(x$sampling_frame, ...)
}

#' @rdname n_timepoints
#' @method n_timepoints matrix_dataset
#' @export
n_timepoints.matrix_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname n_timepoints
#' @method n_timepoints fmri_dataset
#' @export
n_timepoints.fmri_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname n_timepoints
#' @method n_timepoints fmri_mem_dataset
#' @export
n_timepoints.fmri_mem_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname n_timepoints
#' @method n_timepoints fmri_file_dataset
#' @export
n_timepoints.fmri_file_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens matrix_dataset
#' @export
blocklens.matrix_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens fmri_dataset
#' @export
blocklens.fmri_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens fmri_mem_dataset
#' @export
blocklens.fmri_mem_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens fmri_file_dataset
#' @export
blocklens.fmri_file_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids matrix_dataset
#' @export
blockids.matrix_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids fmri_dataset
#' @export
blockids.fmri_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids fmri_mem_dataset
#' @export
blockids.fmri_mem_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids fmri_file_dataset
#' @export
blockids.fmri_file_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration matrix_dataset
#' @export
get_run_duration.matrix_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration fmri_dataset
#' @export
get_run_duration.fmri_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration fmri_mem_dataset
#' @export
get_run_duration.fmri_mem_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration fmri_file_dataset
#' @export
get_run_duration.fmri_file_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration matrix_dataset
#' @export
get_total_duration.matrix_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration fmri_dataset
#' @export
get_total_duration.fmri_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration fmri_mem_dataset
#' @export
get_total_duration.fmri_mem_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration fmri_file_dataset
#' @export
get_total_duration.fmri_file_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples matrix_dataset
#' @export
samples.matrix_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples fmri_dataset
#' @export
samples.fmri_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples fmri_mem_dataset
#' @export
samples.fmri_mem_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples fmri_file_dataset
#' @export
samples.fmri_file_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

# Special case: fmri_study_dataset has subject_ids
#' @rdname n_runs
#' @method n_runs fmri_study_dataset
#' @export
n_runs.fmri_study_dataset <- function(x, ...) {
  x$n_runs
}

#' @rdname n_timepoints
#' @method n_timepoints fmri_study_dataset
#' @export
n_timepoints.fmri_study_dataset <- function(x, ...) {
  n_timepoints(x$sampling_frame, ...)
}

#' @rdname blocklens
#' @method blocklens fmri_study_dataset
#' @export
blocklens.fmri_study_dataset <- function(x, ...) {
  blocklens(x$sampling_frame, ...)
}

#' @rdname blockids
#' @method blockids fmri_study_dataset
#' @export
blockids.fmri_study_dataset <- function(x, ...) {
  blockids(x$sampling_frame, ...)
}

#' @rdname get_TR
#' @method get_TR fmri_study_dataset
#' @export
get_TR.fmri_study_dataset <- function(x, ...) {
  get_TR(x$sampling_frame, ...)
}

#' @rdname get_run_lengths
#' @method get_run_lengths fmri_study_dataset
#' @export
get_run_lengths.fmri_study_dataset <- function(x, ...) {
  get_run_lengths(x$sampling_frame, ...)
}

#' @rdname get_run_duration
#' @method get_run_duration fmri_study_dataset
#' @export
get_run_duration.fmri_study_dataset <- function(x, ...) {
  get_run_duration(x$sampling_frame, ...)
}

#' @rdname get_total_duration
#' @method get_total_duration fmri_study_dataset
#' @export
get_total_duration.fmri_study_dataset <- function(x, ...) {
  get_total_duration(x$sampling_frame, ...)
}

#' @rdname samples
#' @method samples fmri_study_dataset
#' @export
samples.fmri_study_dataset <- function(x, ...) {
  samples(x$sampling_frame, ...)
}

#' Get subject IDs
#' @rdname subject_ids
#' @method subject_ids fmri_study_dataset
#' @export
subject_ids.fmri_study_dataset <- function(x, ...) {
  x$subject_ids
}
