#' Filter subjects in an fmri_group
#'
#' Expressions are evaluated in the context of `subjects(gd)` and may refer to
#' its columns directly. Multiple expressions are combined with logical AND.
#'
#' @param gd An `fmri_group`.
#' @param ... Logical expressions used to filter rows.
#'
#' @return An updated `fmri_group` containing only the rows that satisfy all
#'   predicates.
#' @export
filter_subjects <- function(gd, ...) {
  stopifnot(inherits(gd, "fmri_group"))
  dots <- as.list(substitute(list(...)))[-1L]

  if (!length(dots)) {
    return(gd)
  }

  df <- gd$subjects
  if (!nrow(df)) {
    return(gd)
  }

  mask <- rep(TRUE, nrow(df))
  for (expr in dots) {
    res <- eval(expr, df, parent.frame())
    if (!is.logical(res)) {
      stop("Filter expressions must evaluate to logical vectors.", call. = FALSE)
    }
    if (length(res) == 1L) {
      res <- rep(res, nrow(df))
    }
    if (length(res) != nrow(df)) {
      stop("Logical expressions must be length 1 or `nrow(subjects)`.", call. = FALSE)
    }
    mask <- mask & res
  }

  subjects(gd) <- df[mask, , drop = FALSE]
  gd
}

#' Mutate subject-level attributes
#'
#' Adds or modifies columns on the underlying subjects table. Expressions are
#' evaluated sequentially so newly created columns are available to later
#' expressions.
#'
#' @inheritParams filter_subjects
#'
#' @return An updated `fmri_group` with modified subject attributes.
#' @export
mutate_subjects <- function(gd, ...) {
  stopifnot(inherits(gd, "fmri_group"))
  dots <- as.list(substitute(list(...)))[-1L]

  if (!length(dots)) {
    return(gd)
  }

  df <- gd$subjects
  n <- nrow(df)

  for (i in seq_along(dots)) {
    name <- names(dots)[i]
    if (is.null(name) || name == "") {
      stop("All mutate_subjects() arguments must be named.", call. = FALSE)
    }
    value <- eval(dots[[i]], df, parent.frame())
    if (length(value) == 1L) {
      value <- rep(value, n)
    }
    if (length(value) != n) {
      stop("Column `", name, "` must be length 1 or match the number of subjects.", call. = FALSE)
    }
    df[[name]] <- value
  }

  subjects(gd) <- df
  gd
}

#' Left join additional subject metadata
#'
#' @param gd An `fmri_group`.
#' @param y A data frame containing additional subject-level columns.
#' @param by Character vector of join keys (defaults to the group id column).
#' @param ... Additional arguments passed to `dplyr::left_join()` when
#'   available.
#'
#' @return An `fmri_group` with metadata from `y` attached.
#' @export
left_join_subjects <- function(gd, y, by = NULL, ...) {
  stopifnot(inherits(gd, "fmri_group"))
  stopifnot(is.data.frame(y))

  df <- gd$subjects
  if (is.null(by)) {
    by <- intersect(names(df), names(y))
    if (!(gd$id %in% by)) {
      by <- gd$id
    }
  }
  stopifnot(length(by) >= 1L)

  if (isTRUE(requireNamespace("dplyr", quietly = TRUE))) {
    joined <- dplyr::left_join(df, y, by = by, ...)
  } else {
    if (length(by) != 1L) {
      stop("`dplyr` is required for joins on multiple columns.", call. = FALSE)
    }
    key <- by
    lookup <- match(df[[key]], y[[key]])
    extra_cols <- setdiff(names(y), key)
    extra <- y[lookup, extra_cols, drop = FALSE]
    joined <- cbind(df, extra)
  }

  subjects(gd) <- joined
  gd
}

#' Sample subjects from an fmri_group
#'
#' @param gd An `fmri_group`.
#' @param n Number of subjects to sample. When `strata` is supplied and `n` has
#'   length 1, the same number is drawn from each stratum. Provide a named vector
#'   to request different counts per stratum.
#' @param replace Logical indicating whether to sample with replacement.
#' @param strata Optional column name used to stratify the sampling.
#'
#' @return A sampled `fmri_group`.
#' @export
sample_subjects <- function(gd, n, replace = FALSE, strata = NULL) {
  stopifnot(inherits(gd, "fmri_group"))
  stopifnot(!missing(n))
  stopifnot(is.numeric(n))

  df <- gd$subjects
  total <- nrow(df)

  if (total == 0L) {
    return(gd)
  }

  if (is.null(strata)) {
    n <- as.integer(n)
    if (length(n) != 1L) {
      stop("For unstratified sampling `n` must be a single integer.", call. = FALSE)
    }
    if (!replace && n > total) {
      stop("Cannot sample more subjects than available without replacement.", call. = FALSE)
    }
    idx <- sample.int(total, size = n, replace = replace)
  } else {
    if (is.character(strata) && length(strata) == 1L) {
      strata_values <- df[[strata]]
    } else {
      strata_values <- strata
    }
    split_idx <- split(seq_len(total), strata_values, drop = TRUE)
    strata_keys <- names(split_idx)
    if (is.null(strata_keys)) {
      strata_keys <- vapply(split_idx, function(idx) {
        val <- strata_values[idx][1]
        if (length(val) == 0L || is.na(val)) "NA" else as.character(val)
      }, character(1))
    } else {
      empty <- is.na(strata_keys) | strata_keys == ""
      if (any(empty)) {
        strata_keys[empty] <- "NA"
      }
    }
    idx <- integer(0)
    for (i in seq_along(split_idx)) {
      key <- strata_keys[i]
      group_idx <- split_idx[[i]]
      size <- if (length(n) == 1L) n else n[[key]]
      if (is.null(size)) {
        stop("Missing sample size for stratum `", key, "`.", call. = FALSE)
      }
      size <- as.integer(size)
      if (!replace && size > length(group_idx)) {
        stop("Cannot sample more subjects than available in stratum `", key, "` without replacement.", call. = FALSE)
      }
      idx <- c(idx, sample(group_idx, size = size, replace = replace))
    }
  }

  subjects(gd) <- df[idx, , drop = FALSE]
  gd
}
