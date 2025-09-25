#' Stream subjects with optional ordering
#'
#' @param gd An `fmri_group`.
#' @param prefetch Number of subjects to prefetch. Currently only `1L` is
#'   supported; higher values are accepted for future compatibility but do not
#'   change behaviour.
#' @param order_by Optional column name used to order subjects.
#'
#' @return An iterator identical to `iter_subjects()`.
#' @export
stream_subjects <- function(gd, prefetch = 1L, order_by = NULL) {
  stopifnot(inherits(gd, "fmri_group"))
  stopifnot(length(prefetch) == 1L)
  prefetch <- as.integer(prefetch)
  if (prefetch != 1L) {
    warning("Prefetch values other than 1 are not yet implemented; falling back to sequential iteration.", call. = FALSE)
  }
  iter_subjects(gd, order_by = order_by)
}

#' Reduce over subjects in a single pass
#'
#' @param gd An `fmri_group`.
#' @param .map Function applied to each subject row. Should return an object that
#'   can be combined by `.reduce`.
#' @param .reduce Binary function combining the accumulator and the mapped value.
#' @param .init Initial accumulator value.
#' @param order_by Optional ordering column.
#' @param on_error Error handling policy: "stop", "warn", or "skip".
#' @param ... Additional arguments passed to `.map`.
#'
#' @return The reduced value after visiting all subjects.
#' @export
group_reduce <- function(gd, .map, .reduce, .init, order_by = NULL,
                          on_error = c("stop", "warn", "skip"), ...) {
  stopifnot(inherits(gd, "fmri_group"))
  stopifnot(is.function(.map))
  stopifnot(is.function(.reduce))

  on_error <- match.arg(on_error)
  acc <- .init
  iterator <- iter_subjects(gd, order_by = order_by)
  skip_sentinel <- structure(list(), class = "group_reduce_skip")

  repeat {
    row <- iterator[["next"]]()
    if (is.null(row)) {
      break
    }

    value <- tryCatch(
      .map(row, ...),
      error = function(e) {
        if (on_error == "stop") {
          stop(e)
        }
        if (on_error == "warn") {
          warning("group_reduce(): ", conditionMessage(e), call. = FALSE)
        }
        skip_sentinel
      }
    )

    if (identical(value, skip_sentinel)) {
      next
    }

    acc <- .reduce(acc, value)
  }

  acc
}
