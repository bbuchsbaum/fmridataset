#' @export
get_data.fmri_study_dataset <- function(x, ...) {
  backend_get_data(x$backend, ...)
}

#' @export
get_data_matrix.fmri_study_dataset <- function(x, subject_id = NULL, ...) {
  if (!is.null(subject_id)) {
    # Return data for specific subject(s)
    if (is.character(subject_id)) {
      idx <- match(subject_id, x$subject_ids)
      if (any(is.na(idx))) {
        stop("Subject ID(s) not found: ", paste(subject_id[is.na(idx)], collapse = ", "))
      }
    } else if (is.numeric(subject_id)) {
      idx <- subject_id
    } else {
      stop("subject_id must be character or numeric")
    }

    # Get data from specific backend(s)
    if (length(idx) == 1) {
      backend_get_data(x$backend$backends[[idx]], ...)
    } else {
      # Combine data from multiple subjects
      data_list <- lapply(idx, function(i) backend_get_data(x$backend$backends[[i]], ...))
      do.call(rbind, data_list)
    }
  } else {
    # Return all data
    backend_get_data(x$backend, ...)
  }
}

#' Convert fmri_study_dataset to a tibble or lazy matrix
#'
#' Primary data access method for study-level datasets. By default this
#' returns a lazy matrix (typically a `delarr` object) with row-wise
#' metadata attached. When
#' `materialise = TRUE`, the data matrix is materialised and returned as
#' a tibble with metadata columns prepended.
#'
#' @param x An `fmri_study_dataset` object
#' @param materialise Logical; return a materialised tibble? Default `FALSE`.
#' @param ... Additional arguments (unused)
#'
#' @return Either a lazy matrix with metadata attributes or a tibble
#'   when `materialise = TRUE`.
#' @export
#' @importFrom tibble as_tibble
as_tibble.fmri_study_dataset <- function(x, materialise = FALSE, ...) {
  run_lengths <- x$sampling_frame$blocklens
  run_ids <- fmrihrf::blockids(x$sampling_frame)
  backend_times <- vapply(
    x$backend$backends,
    function(b) backend_get_dims(b)$time, numeric(1)
  )
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
    subject_mats <- lapply(x$backend$backends, function(backend) {
      subj_mat <- backend_get_data(backend)
      if (!is.matrix(subj_mat)) {
        subj_mat <- as.matrix(subj_mat)
      }
      subj_mat
    })
    combined <- do.call(rbind, subject_mats)
    tb <- tibble::as_tibble(cbind(rowData, combined))
    return(tb)
  }

  mat <- backend_get_data(x$backend)
  if (nrow(mat) > 100000) {
    attr(mat, "AltExp") <- rowData
  } else {
    mat <- with_rowData(mat, rowData)
  }
  mat
}
