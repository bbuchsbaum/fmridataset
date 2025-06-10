#' Convert fmri_study_dataset to a tibble or DelayedMatrix
#'
#' Primary data access method for study-level datasets. By default this
#' returns a lazy `DelayedMatrix` with row-wise metadata attached. When
#' `materialise = TRUE`, the data matrix is materialised and returned as
#' a tibble with metadata columns prepended.
#'
#' @param x An `fmri_study_dataset` object
#' @param materialise Logical; return a materialised tibble? Default `FALSE`.
#' @param ... Additional arguments (unused)
#'
#' @return Either a `DelayedMatrix` with metadata attributes or a tibble
#'   when `materialise = TRUE`.
#' @export
#' @importFrom tibble as_tibble
as_tibble.fmri_study_dataset <- function(x, materialise = FALSE, ...) {
  mat <- backend_get_data(x$backend)

  run_lengths <- x$sampling_frame$blocklens
  run_ids <- fmrihrf::blockids(x$sampling_frame)
  backend_times <- vapply(x$backend$backends,
                          function(b) backend_get_dims(b)$time, numeric(1))
  subj_ids <- x$subject_ids

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
  rowData <- data.frame(
    subject_id = row_subject,
    run_id = run_ids,
    timepoint = seq_len(length(run_ids))
  )

  if (materialise) {
    tb <- tibble::as_tibble(cbind(rowData, as.matrix(mat)))
    return(tb)
  }

  if (nrow(mat) > 100000) {
    attr(mat, "AltExp") <- rowData
  } else {
    mat <- with_rowData(mat, rowData)
  }
  mat
}
