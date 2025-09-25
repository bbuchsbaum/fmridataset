#' Map a function over subjects in an fmri_group
#'
#' @param gd An `fmri_group`.
#' @param .f A function with signature `function(row, ...)` where `row` is a
#'   one-row `data.frame` corresponding to a single subject.
#' @param ... Additional arguments passed through to `.f`.
#' @param out Either "list" (default) or "bind_rows" describing how to collect
#'   outputs.
#' @param order_by Optional column name used to define iteration order.
#' @param on_error One of "stop", "warn", or "skip" describing how to handle
#'   errors raised by `.f`.
#'
#' @return Either a list (for `out = "list"`) or a bound table (for
#'   `out = "bind_rows"`).
#' @export
group_map <- function(gd, .f, ..., out = c("list", "bind_rows"),
                      order_by = NULL,
                      on_error = c("stop", "warn", "skip")) {
  stopifnot(inherits(gd, "fmri_group"))
  stopifnot(is.function(.f))

  out <- match.arg(out)
  on_error <- match.arg(on_error)

  iterator <- iter_subjects(gd, order_by = order_by)
  results <- list()
  index <- 0L
  skip_sentinel <- structure(list(), class = "group_map_skip")

  repeat {
    row <- iterator[["next"]]()
    if (is.null(row)) {
      break
    }

    subject_id <- tryCatch(
      as.character(row[[gd$id]][1]),
      error = function(e) NA_character_
    )

    value <- tryCatch(
      .f(row, ...),
      error = function(e) {
        if (on_error == "stop") {
          stop(e)
        }
        if (on_error == "warn") {
          warning("group_map(): ", conditionMessage(e), call. = FALSE)
        }
        skip_sentinel
      }
    )

    if (identical(value, skip_sentinel)) {
      next
    }

    if (is.null(value)) {
      next
    }

    index <- index + 1L
    if (!is.na(subject_id)) {
      results[[subject_id]] <- value
    } else {
      results[[index]] <- value
    }
  }

  if (identical(out, "list")) {
    return(results)
  }

  if (!length(results)) {
    if (isTRUE(requireNamespace("tibble", quietly = TRUE))) {
      return(tibble::tibble())
    }
    return(data.frame())
  }

  if (isTRUE(requireNamespace("dplyr", quietly = TRUE))) {
    return(dplyr::bind_rows(results))
  }

  all_df <- all(vapply(results, is.data.frame, logical(1)))
  if (all_df) {
    bound <- do.call(rbind, results)
    rownames(bound) <- NULL
    return(bound)
  }

  stop("`out = \"bind_rows\"` requires `dplyr` or data.frame outputs with matching columns.", call. = FALSE)
}
