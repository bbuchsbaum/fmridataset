#' Create an fmri_group (one row per subject)
#'
#' @param subjects A `data.frame` (or tibble) with one row per subject where one
#'   column contains per-subject `fmri_dataset` objects stored as a list column.
#' @param id Character scalar giving the name of the subject identifier column.
#' @param dataset_col Character scalar naming the list column that stores the
#'   per-subject dataset handles.
#' @param space Optional character string describing the nominal common space
#'   for all subjects (e.g., "MNI152NLin2009cAsym").
#' @param mask_strategy One of "subject_specific", "intersect", or "union"
#'   describing how masks should be handled when combining subjects. This is a
#'   declarative flag only; no resampling is performed by the constructor.
#'
#' @return An object of class `fmri_group` that wraps the input table.
#' @export
fmri_group <- function(subjects,
                       id,
                       dataset_col = "dataset",
                       space = NULL,
                       mask_strategy = c("subject_specific", "intersect", "union")) {
  stopifnot(is.data.frame(subjects))
  stopifnot(is.character(id), length(id) == 1L, !is.na(id), nzchar(id))
  stopifnot(id %in% names(subjects))
  stopifnot(is.character(dataset_col), length(dataset_col) == 1L, !is.na(dataset_col), nzchar(dataset_col))
  stopifnot(dataset_col %in% names(subjects))

  mask_strategy <- match.arg(mask_strategy)

  dataset_column <- subjects[[dataset_col]]
  if (!is.list(dataset_column)) {
    stop("`", dataset_col, "` must be a list-column of per-subject fmri_dataset objects.", call. = FALSE)
  }

  x <- list(
    subjects = subjects,
    id = id,
    dataset_col = dataset_col,
    space = space,
    mask_strategy = mask_strategy
  )
  class(x) <- c("fmri_group", "fmridataset_group")
  validate_fmri_group(x)
  x
}

#' Validate an fmri_group object
#'
#' @param x Object to validate.
#'
#' @return The validated `fmri_group` (invisibly).
#' @export
validate_fmri_group <- function(x) {
  stopifnot(inherits(x, "fmri_group"))

  s <- x$subjects
  dataset_col <- x$dataset_col

  nlen <- vapply(s[[dataset_col]], length, integer(1))
  if (length(nlen) && any(nlen != 1L)) {
    bad <- which(nlen != 1L)[1]
    stop("Row ", bad, " of `", dataset_col, "` does not contain a length-1 entry.", call. = FALSE)
  }

  if (length(s[[dataset_col]]) && any(vapply(s[[dataset_col]], is.null, logical(1)))) {
    warning("At least one subject has NULL in `", dataset_col, "`.", call. = FALSE)
  }

  invisible(x)
}

#' Access the subjects tibble stored inside an fmri_group
#'
#' @param x An `fmri_group`.
#'
#' @return The underlying `data.frame` with one row per subject.
#' @export
subjects <- function(x) {
  stopifnot(inherits(x, "fmri_group"))
  x$subjects
}

#' @rdname subjects
#' @param value A replacement table containing the dataset column used by the
#'   group.
#' @return An updated `fmri_group`.
#' @export
`subjects<-` <- function(x, value) {
  stopifnot(inherits(x, "fmri_group"))
  stopifnot(is.data.frame(value))

  if (!x$dataset_col %in% names(value)) {
    stop("Replacement `subjects` must have column `", x$dataset_col, "`.", call. = FALSE)
  }

  x$subjects <- value
  validate_fmri_group(x)
  x
}

#' Coerce a data frame into an fmri_group
#'
#' @inheritParams fmri_group
#'
#' @return An `fmri_group` object.
#' @export
as_fmri_group <- function(subjects,
                          id,
                          dataset_col = "dataset",
                          space = NULL,
                          mask_strategy = c("subject_specific", "intersect", "union")) {
  fmri_group(
    subjects = subjects,
    id = id,
    dataset_col = dataset_col,
    space = space,
    mask_strategy = match.arg(mask_strategy)
  )
}

#' @export
print.fmri_group <- function(x, ...) {
  stopifnot(inherits(x, "fmri_group"))

  s <- x$subjects
  dataset_col <- x$dataset_col
  n_subjects <- nrow(s)
  attrs <- setdiff(names(s), dataset_col)

  cat("<fmri_group>\n", sep = "")
  cat("  subjects       : ", n_subjects, "\n", sep = "")
  cat("  id column      : ", x$id, "\n", sep = "")
  cat("  dataset column : ", dataset_col, "\n", sep = "")
  if (!is.null(x$space)) {
    cat("  space          : ", x$space, "\n", sep = "")
  }
  cat("  mask strategy  : ", x$mask_strategy, "\n", sep = "")
  if (length(attrs)) {
    cat("  subject attrs  : ", paste(attrs, collapse = ", "), "\n", sep = "")
  } else {
    cat("  subject attrs  : (none)\n", sep = "")
  }
  invisible(x)
}

#' Number of subjects in a group
#'
#' @param gd An `fmri_group`.
#'
#' @return Integer number of subjects.
#' @export
n_subjects <- function(gd) {
  stopifnot(inherits(gd, "fmri_group"))
  nrow(gd$subjects)
}
