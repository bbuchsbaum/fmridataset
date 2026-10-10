#' The observation-axis temporal contract
#'
#' The canonical frame model carries no acquisition timing of its own: an
#' `fmri_frame` is observations by features, and when the observations happen to
#' be volumes acquired in runs, that fact is ordinary observation metadata. The
#' package already writes it that way -- `read_bids_bold()` emits `run_id` and
#' `TR` columns -- but nothing validated the convention, so every consumer
#' reinvented it and none could rely on it.
#'
#' These functions make it a contract. `temporal_schema()` derives a validated
#' description from the observation metadata; `as_sampling_frame()` reconstructs
#' the `fmrihrf::sampling_frame` that design and HRF modelling code expects.
#'
#' The schema is derived, never stored. The columns are the truth, so the schema
#' cannot go stale, needs no serialization of its own, and follows subsetting,
#' reordering, and binding for free.
#'
#' @section Contract:
#' \describe{
#'   \item{`run_id`}{Required. One value per observation naming the acquisition
#'     run it belongs to. Any type; compared as character. No missing values.}
#'   \item{`TR`}{Optional. Repetition time in seconds, positive and finite, and
#'     constant within each run (compared with a relative tolerance of
#'     `sqrt(.Machine$double.eps)`, so floating-point noise is not a second
#'     value). Runs may differ from one another, matching
#'     `fmrihrf::sampling_frame()`.}
#'   \item{`start_time`}{Optional. Non-negative finite offset in seconds of the
#'     first volume of each run (the slice-time-correction reference, or any
#'     other first-volume acquisition time). Constant within each run; runs may
#'     differ. When absent, `as_sampling_frame()` leaves the offset to
#'     `fmrihrf::sampling_frame()`'s default of `TR/2`.}
#'   \item{`censor`}{Optional. Logical, one value per observation, `TRUE` where
#'     the observation is to be excluded. No missing values.}
#'   \item{order column}{Optional. A numeric within-run acquisition index or
#'     time, such as `volume_index` or `run_time` from `read_bids_bold()`. No
#'     missing values.}
#' }
#'
#' @section Timing of samples:
#' `read_bids_bold()` emits `run_time`, the acquisition onset of each volume
#' relative to its run (`(volume_index - 1) * TR`). The sampling frame built by
#' `as_sampling_frame()` places samples at `start_time + (volume_index - 1) *
#' TR`, and when no `start_time` column is supplied fmrihrf's default
#' `start_time = TR/2` applies, so sample times are `run_time + TR/2`. This
#' modelling default is kept on purpose. For onset-aligned sampling, add a
#' `start_time` column of `0` when constructing the frame, or set it on the
#' schema before converting:
#' ```
#' schema <- temporal_schema(frame)
#' schema$start_time <- stats::setNames(rep(0, schema$n_runs), names(schema$run_lengths))
#' as_sampling_frame(schema)
#' ```
#'
#' @section Order and contiguity:
#' Runs are numbered in order of first appearance, not by sorting, so
#' `block_ids` is stable under any operation that preserves observation order.
#' A frame is *contiguous* when each run occupies one unbroken stretch of
#' observations. Frames are not required to be contiguous -- `filter_obs()` and
#' ID-based reordering both produce legal interleaved views -- but a
#' `sampling_frame` is a run-length encoding and cannot represent one, so
#' `as_sampling_frame()` refuses a non-contiguous frame rather than silently
#' reordering it.
#'
#' Contiguity says nothing about order *within* a run. When an order column is
#' present (`order_col`, by default the first of `volume_index` and `run_time`
#' found), the schema records in `ordered` whether it increases strictly within
#' every run, and `as_sampling_frame()` refuses a frame whose volumes are out of
#' acquisition order. Without such a column the order within a run cannot be
#' checked: `ordered` is `NA`, and the observation order is taken as acquisition
#' order.
#'
#' A zero-observation frame has an empty schema but no sampling frame;
#' `as_sampling_frame()` refuses it.
#'
#' @param x An `fmri_frame` or `fmri_view`.
#' @param run_col,tr_col,start_time_col,censor_col Observation metadata columns
#'   holding the run label, repetition time, first-volume offset, and censoring
#'   indicator. Each is a single column name; `run_col` may also be `NULL` (the
#'   default) to use the key of an observation relation to a run entity, else
#'   `scan_id`, else `run_id`. A named column other than
#'   `run_col` that is absent is simply not used.
#' @param order_col `NULL` (default) to use the first of `volume_index` and
#'   `run_time` that is present, or a single column name holding a numeric
#'   within-run acquisition index or time, which must then exist.
#' @param ... Passed to `temporal_schema()`.
#' @return `temporal_schema()` returns a `frame_temporal_schema`.
#'   `has_temporal_schema()` returns a scalar logical.
#'   `as_sampling_frame()` returns an `fmrihrf::sampling_frame`.
#' @name temporal-schema
#' @examples
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(rnorm(12), 6, 2)),
#'   observations = data.frame(
#'     .obs_id = sprintf("t%02d", 1:6),
#'     run_id = rep(c("run-1", "run-2"), each = 3),
#'     TR = 2
#'   )
#' )
#' schema <- temporal_schema(frame)
#' schema$run_lengths
#' if (requireNamespace("fmrihrf", quietly = TRUE)) {
#'   as_sampling_frame(frame)
#' }
NULL

.temporal_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_temporal", ...)
}

#' @rdname temporal-schema
#' @export
temporal_schema <- function(x, run_col = NULL, tr_col = "TR",
                            start_time_col = "start_time",
                            censor_col = "censor", order_col = NULL) {
  .temporal_column_arg(run_col, "run_col", allow_null = TRUE)
  .temporal_column_arg(tr_col, "tr_col")
  .temporal_column_arg(start_time_col, "start_time_col")
  .temporal_column_arg(censor_col, "censor_col")
  .temporal_column_arg(order_col, "order_col", allow_null = TRUE)

  data <- observations(x)
  run_col <- .resolve_run_column(x, data, run_col)
  run_ids <- .temporal_run_ids(data[[run_col]], run_col)
  levels <- unique(run_ids)
  block_ids <- match(run_ids, levels)
  run_lengths <- tabulate(block_ids, nbins = length(levels))
  names(run_lengths) <- levels
  order_col <- .temporal_order_column(data, order_col)

  structure(
    list(
      run_ids = run_ids,
      block_ids = block_ids,
      run_lengths = run_lengths,
      n_runs = length(levels),
      TR = .temporal_tr(data, tr_col, block_ids, levels),
      start_time = .temporal_start_time(
        data, start_time_col, block_ids, levels
      ),
      censor = .temporal_censor(data, censor_col, length(run_ids)),
      contiguous = .temporal_contiguous(block_ids, length(levels)),
      ordered = .temporal_ordered(data, order_col, block_ids, length(levels)),
      columns = list(
        run = run_col,
        tr = tr_col,
        start_time = start_time_col,
        censor = censor_col,
        order = order_col
      )
    ),
    class = "frame_temporal_schema"
  )
}

#' @rdname temporal-schema
#' @export
has_temporal_schema <- function(x, ...) {
  !inherits(tryCatch(temporal_schema(x, ...), error = function(e) e), "error")
}

# Which observation column names the acquisition run.
#
# A frame that carries entities already declares this: read_bids_bold() builds
# a `run` entity keyed on scan_id and an observation->run key_relation, so the
# relation's key IS the run column. Prefer that over guessing, because the
# obvious guess is wrong: BIDS `run_id` is a within-session label, so a subject
# with two sessions each holding "run-1" would have two distinct acquisitions
# merged into one run, and a dataset with no run entity would collapse every
# scan into a single "run-none". scan_id is unique per acquisition by
# construction; run_id is not.
.resolve_run_column <- function(x, data, run_col) {
  if (!is.null(run_col)) {
    if (!run_col %in% names(data)) {
      .temporal_abort(
        sprintf(
          "Observation metadata has no %s column. Available: %s.",
          encodeString(run_col, quote = "\""),
          if (length(names(data))) paste(names(data), collapse = ", ") else "none"
        ),
        column = run_col, available = names(data)
      )
    }
    return(run_col)
  }

  declared <- .declared_run_column(x)
  candidates <- c(declared, "scan_id", "run_id")
  found <- candidates[candidates %in% names(data)]
  if (length(found)) {
    return(found[[1L]])
  }

  .temporal_abort(
    sprintf(
      paste(
        "No temporal schema: nothing identifies the acquisition run.",
        "Looked for %s. Available: %s.",
        "Pass run_col, or relate observations to a run entity."
      ),
      paste(encodeString(candidates, quote = "\""), collapse = ", "),
      if (length(names(data))) paste(names(data), collapse = ", ") else "none"
    ),
    candidates = candidates, available = names(data)
  )
}

# The key of an observation-sourced relation pointing at a run-typed entity.
.declared_run_column <- function(x) {
  registry <- tryCatch(relations(x), error = function(e) NULL)
  entity_set <- tryCatch(entities(x), error = function(e) NULL)
  if (!length(registry) || !length(entity_set)) {
    return(character())
  }
  keys <- vapply(registry, function(relation) {
    target <- relation$target
    if (!identical(relation$source, "observation") || is.null(target)) {
      return(NA_character_)
    }
    target_entity <- entity_set[[target]]
    if (is.null(target_entity) || !identical(target_entity$entity_type, "run")) {
      return(NA_character_)
    }
    relation$key %||% NA_character_
  }, character(1))
  unname(keys[!is.na(keys)])
}

.temporal_run_ids <- function(values, run_col) {
  if (anyNA(values)) {
    .temporal_abort(
      sprintf(
        "Column %s has %d missing value(s); every observation must name its run.",
        encodeString(run_col, quote = "\""), sum(is.na(values))
      ),
      column = run_col
    )
  }
  ids <- as.character(values)
  if (length(ids) && any(!nzchar(ids))) {
    .temporal_abort(
      sprintf("Column %s has empty run labels.", encodeString(run_col, quote = "\"")),
      column = run_col
    )
  }
  ids
}

.temporal_tr <- function(data, tr_col, block_ids, levels) {
  .temporal_per_run_seconds(
    data = data,
    column = tr_col,
    block_ids = block_ids,
    levels = levels,
    allow_zero = FALSE,
    singular_label = "repetition time"
  )
}

.temporal_start_time <- function(data, start_time_col, block_ids, levels) {
  .temporal_per_run_seconds(
    data = data,
    column = start_time_col,
    block_ids = block_ids,
    levels = levels,
    allow_zero = TRUE,
    singular_label = "start time"
  )
}

# Optional numeric-seconds columns that are constant within a run (TR,
# start_time). Positive when allow_zero is FALSE; non-negative otherwise.
.temporal_per_run_seconds <- function(data, column, block_ids, levels,
                                      allow_zero, singular_label) {
  if (!column %in% names(data)) {
    return(NULL)
  }
  values <- data[[column]]
  if (!is.numeric(values)) {
    .temporal_abort(
      sprintf(
        "Column %s must be numeric seconds, not %s.",
        encodeString(column, quote = "\""), class(values)[1L]
      ),
      column = column
    )
  }
  out_of_range <- if (allow_zero) {
    any(values < 0)
  } else {
    any(values <= 0)
  }
  if (anyNA(values) || any(!is.finite(values)) || out_of_range) {
    .temporal_abort(
      sprintf(
        "Column %s must be %s and finite in every observation.",
        encodeString(column, quote = "\""),
        if (allow_zero) "non-negative" else "positive"
      ),
      column = column
    )
  }

  # Values within a run are compared with a small relative tolerance so that
  # floating-point noise (0.1 * 3 vs 0.3) is not mistaken for a second value.
  tolerance <- sqrt(.Machine$double.eps)
  per_run <- vapply(seq_along(levels), function(i) {
    run_vals <- values[block_ids == i]
    reference <- run_vals[[1L]]
    scale <- pmax(abs(run_vals), abs(reference), 1)
    if (any(abs(run_vals - reference) > tolerance * scale)) {
      unique_vals <- unique(run_vals)
      .temporal_abort(
        sprintf(
          "Run %s has %d different %s values (%s); a run has one %s.",
          encodeString(levels[[i]], quote = "\""), length(unique_vals),
          encodeString(column, quote = "\""),
          paste(format(unique_vals, digits = 15), collapse = ", "),
          singular_label
        ),
        column = column,
        run = levels[[i]],
        actual = unique_vals
      )
    }
    as.numeric(reference)
  }, numeric(1))

  names(per_run) <- levels
  per_run
}

.temporal_censor <- function(data, censor_col, n) {
  if (!censor_col %in% names(data)) {
    return(NULL)
  }
  values <- data[[censor_col]]
  if (!is.logical(values)) {
    .temporal_abort(
      sprintf(
        "Column %s must be logical, not %s.",
        encodeString(censor_col, quote = "\""), class(values)[1L]
      ),
      column = censor_col
    )
  }
  if (anyNA(values)) {
    .temporal_abort(
      sprintf(
        "Column %s has missing values; censoring must be decided for every observation.",
        encodeString(censor_col, quote = "\"")
      ),
      column = censor_col
    )
  }
  stopifnot(length(values) == n)
  values
}

# A run is contiguous when it occupies one unbroken stretch. Comparing the
# number of runs against the number of runs of equal values in the observation
# order detects any interleaving or repetition, including the reordered views
# that filter_obs() and ID-based selection produce.
.temporal_contiguous <- function(block_ids, n_runs) {
  if (!length(block_ids)) {
    return(TRUE)
  }
  length(rle(block_ids)$lengths) == n_runs
}

.temporal_column_arg <- function(value, arg, allow_null = FALSE) {
  if (is.null(value) && allow_null) {
    return(invisible(NULL))
  }
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
    !nzchar(value)) {
    .temporal_abort(
      sprintf(
        "`%s` must be a single non-empty column name%s.",
        arg, if (allow_null) " or NULL" else ""
      ),
      argument = arg
    )
  }
  invisible(value)
}

# The column that orders volumes within a run, or NULL when none is present.
.temporal_order_column <- function(data, order_col) {
  if (!is.null(order_col)) {
    if (!order_col %in% names(data)) {
      .temporal_abort(
        sprintf(
          "Observation metadata has no %s column to order volumes by.",
          encodeString(order_col, quote = "\"")
        ),
        column = order_col, available = names(data)
      )
    }
    return(order_col)
  }
  found <- intersect(c("volume_index", "run_time"), names(data))
  if (length(found)) found[[1L]] else NULL
}

# TRUE when the order column increases strictly within every run, FALSE when
# it does not, and NA when there is no order column to check against.
.temporal_ordered <- function(data, order_col, block_ids, n_runs) {
  if (is.null(order_col)) {
    return(NA)
  }
  values <- data[[order_col]]
  if (!is.numeric(values) || anyNA(values) || any(!is.finite(values))) {
    .temporal_abort(
      sprintf(
        "Column %s must be finite numeric acquisition order in every observation.",
        encodeString(order_col, quote = "\"")
      ),
      column = order_col
    )
  }
  for (i in seq_len(n_runs)) {
    if (any(diff(values[block_ids == i]) <= 0)) {
      return(FALSE)
    }
  }
  TRUE
}

#' @rdname temporal-schema
#' @export
as_sampling_frame <- function(x, ...) {
  schema <- if (inherits(x, "frame_temporal_schema")) x else temporal_schema(x, ...)

  if (is.null(schema$TR)) {
    .temporal_abort(
      sprintf(
        "A sampling frame needs repetition times; observation metadata has no %s column.",
        encodeString(schema$columns$tr, quote = "\"")
      ),
      column = schema$columns$tr
    )
  }
  if (!isTRUE(schema$contiguous)) {
    .temporal_abort(
      paste(
        "A sampling frame is a run-length encoding and cannot describe a frame",
        "whose runs are interleaved or reordered. Restore acquisition order",
        "before converting, or work from temporal_schema() directly."
      ),
      contiguous = FALSE,
      n_runs = schema$n_runs
    )
  }
  if (isFALSE(schema$ordered)) {
    .temporal_abort(
      sprintf(
        paste(
          "A sampling frame needs volumes in acquisition order, but column %s",
          "does not increase strictly within every run. Restore acquisition",
          "order before converting, or work from temporal_schema() directly."
        ),
        encodeString(schema$columns$order, quote = "\"")
      ),
      column = schema$columns$order,
      ordered = FALSE
    )
  }
  if (!length(schema$run_ids)) {
    .temporal_abort(
      "A sampling frame needs at least one observation; this frame has none.",
      n_observations = 0L
    )
  }

  if (!requireNamespace("fmrihrf", quietly = TRUE)) {
    .temporal_abort(
      "Package 'fmrihrf' is required to build a sampling frame. Install it or work from temporal_schema() directly.",
      package = "fmrihrf"
    )
  }
  args <- list(
    blocklens = unname(schema$run_lengths),
    TR = unname(schema$TR)
  )
  # Omit start_time when the frame does not declare one so fmrihrf keeps its
  # TR/2 default; an explicit NULL would still override that default.
  if (!is.null(schema$start_time)) {
    args$start_time <- unname(schema$start_time)
  }
  do.call(fmrihrf::sampling_frame, args)
}

#' @export
print.frame_temporal_schema <- function(x, ...) {
  cat(sprintf(
    "<frame_temporal_schema> %d observations in %d run%s%s\n",
    length(x$run_ids), x$n_runs, if (x$n_runs == 1L) "" else "s",
    paste0(
      if (x$contiguous) "" else " (not contiguous)",
      if (isFALSE(x$ordered)) " (not in acquisition order)" else ""
    )
  ))
  if (x$n_runs) {
    shown <- seq_len(min(x$n_runs, 6L))
    for (i in shown) {
      extras <- character()
      if (!is.null(x$TR)) {
        extras <- c(extras, sprintf("TR %s s", format(x$TR[[i]])))
      }
      if (!is.null(x$start_time)) {
        extras <- c(
          extras,
          sprintf("start_time %s s", format(x$start_time[[i]]))
        )
      }
      cat(sprintf(
        "  %-16s %4d observations%s\n",
        names(x$run_lengths)[[i]], x$run_lengths[[i]],
        if (length(extras)) paste0("  ", paste(extras, collapse = "  ")) else ""
      ))
    }
    if (x$n_runs > length(shown)) {
      cat(sprintf("  ... %d more\n", x$n_runs - length(shown)))
    }
  }
  if (!is.null(x$censor)) {
    cat(sprintf("  censored: %d of %d\n", sum(x$censor), length(x$censor)))
  }
  invisible(x)
}
