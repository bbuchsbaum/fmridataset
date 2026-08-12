#' Iterate subjects one-by-one (streaming)
#'
#' @param gd An `fmri_group`.
#' @param order_by Optional character scalar giving the column used to order
#'   iteration. If supplied, subjects are iterated in ascending order of this
#'   column (with `NA` values placed last).
#'
#' @return A list with a single element `next` that yields a one-row
#'   `data.frame` for each subject when called repeatedly. The dataset column is
#'   automatically flattened to the underlying `fmri_dataset` object.
#' @export
iter_subjects <- function(gd, order_by = NULL) {
  stopifnot(inherits(gd, "fmri_group"))

  df <- gd$subjects
  dataset_col <- gd$dataset_col

  if (!is.null(order_by)) {
    stopifnot(is.character(order_by), length(order_by) == 1L, !is.na(order_by))
    if (!order_by %in% names(df)) {
      stop("`order_by` must be the name of a column in `subjects(gd)`.", call. = FALSE)
    }
    ord <- order(df[[order_by]], na.last = TRUE)
    df <- df[ord, , drop = FALSE]
  }

  i <- 0L
  n <- nrow(df)

  list(`next` = function() {
    i <<- i + 1L
    if (i > n) {
      return(NULL)
    }

    row <- df[i, , drop = FALSE]
    row[[dataset_col]] <- row[[dataset_col]][[1]]
    row
  })
}
