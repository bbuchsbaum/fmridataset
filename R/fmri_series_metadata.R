#' Temporal metadata builders for fmri_series
#'
#' Internal helpers used to construct the `temporal_info` component of
#' `fmri_series` objects. These functions return data.frame
#' objects describing each selected timepoint. They are not exported
#' for users.
#'
#' @keywords internal
build_temporal_info_lazy <- function(dataset, time_indices) {
  UseMethod("build_temporal_info_lazy")
}

#' @export
build_temporal_info_lazy.fmri_dataset <- function(dataset, time_indices) {
  run_ids <- fmrihrf::blockids(dataset$sampling_frame)
  data.frame(
    run_id = run_ids[time_indices],
    timepoint = time_indices
  )
}

#' @export
build_temporal_info_lazy.fmri_study_dataset <- function(dataset, time_indices) {
  run_lengths <- dataset$sampling_frame$blocklens
  run_ids <- fmrihrf::blockids(dataset$sampling_frame)
  backend_times <- vapply(
    dataset$backend$backends,
    function(b) backend_get_dims(b)$time, numeric(1)
  )
  subj_ids <- dataset$subject_ids

  run_subject <- character(length(run_lengths))
  subj_idx <- 1
  acc <- 0
  for (i in seq_along(run_lengths)) {
    run_subject[i] <- subj_ids[subj_idx]
    acc <- acc + run_lengths[i]
    if (acc == backend_times[subj_idx]) {
      subj_idx <- subj_idx + 1
      acc <- 0
    } else if (acc > backend_times[subj_idx]) {
      stop_fmridataset(
        fmridataset_error_config,
        "run lengths inconsistent with backend dimensions"
      )
    }
  }

  row_subject <- rep(run_subject, run_lengths)
  data.frame(
    subject_id = row_subject[time_indices],
    run_id = run_ids[time_indices],
    timepoint = time_indices
  )
}
