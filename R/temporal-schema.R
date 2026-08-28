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
#' the `fmrihrf::sampling_frame` that the legacy accessors and the design
#' machinery expect.
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
#'     constant within each run. Runs may differ from one another, matching
#'     `fmrihrf::sampling_frame()`.}
#'   \item{`censor`}{Optional. Logical, one value per observation, `TRUE` where
#'     the observation is to be excluded. No missing values.}
#' }
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
#' @param x An `fmri_frame` or `fmri_view`.
#' @param run_col,tr_col,censor_col Observation metadata columns holding the
#'   run label, repetition time, and censoring indicator.
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
#' as_sampling_frame(frame)
NULL

.temporal_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_temporal", ...)
}

#' @rdname temporal-schema
#' @export
temporal_schema <- function(x, run_col = NULL, tr_col = "TR",
                            censor_col = "censor") {
  data <- observations(x)
  run_col <- .resolve_run_column(x, data, run_col)
  run_ids <- .temporal_run_ids(data[[run_col]], run_col)
  levels <- unique(run_ids)
  block_ids <- match(run_ids, levels)
  run_lengths <- tabulate(block_ids, nbins = length(levels))
  names(run_lengths) <- levels

  structure(
    list(
      run_ids = run_ids,
      block_ids = block_ids,
      run_lengths = run_lengths,
      n_runs = length(levels),
      TR = .temporal_tr(data, tr_col, block_ids, levels),
      censor = .temporal_censor(data, censor_col, length(run_ids)),
      contiguous = .temporal_contiguous(block_ids, length(levels)),
      columns = list(run = run_col, tr = tr_col, censor = censor_col)
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
  if (!tr_col %in% names(data)) {
    return(NULL)
  }
  values <- data[[tr_col]]
  if (!is.numeric(values)) {
    .temporal_abort(
      sprintf(
        "Column %s must be numeric seconds, not %s.",
        encodeString(tr_col, quote = "\""), class(values)[1L]
      ),
      column = tr_col
    )
  }
  if (anyNA(values) || any(!is.finite(values)) || any(values <= 0)) {
    .temporal_abort(
      sprintf(
        "Column %s must be positive and finite in every observation.",
        encodeString(tr_col, quote = "\"")
      ),
      column = tr_col
    )
  }

  per_run <- vapply(seq_along(levels), function(i) {
    unique_tr <- unique(values[block_ids == i])
    if (length(unique_tr) != 1L) {
      .temporal_abort(
        sprintf(
          "Run %s has %d different %s values (%s); a run has one repetition time.",
          encodeString(levels[[i]], quote = "\""), length(unique_tr),
          encodeString(tr_col, quote = "\""),
          paste(format(unique_tr), collapse = ", ")
        ),
        column = tr_col,
        run = levels[[i]],
        actual = unique_tr
      )
    }
    as.numeric(unique_tr)
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

  fmrihrf::sampling_frame(
    blocklens = unname(schema$run_lengths),
    TR = unname(schema$TR)
  )
}

#' @export
print.frame_temporal_schema <- function(x, ...) {
  cat(sprintf(
    "<frame_temporal_schema> %d observations in %d run%s%s\n",
    length(x$run_ids), x$n_runs, if (x$n_runs == 1L) "" else "s",
    if (x$contiguous) "" else " (not contiguous)"
  ))
  if (x$n_runs) {
    shown <- seq_len(min(x$n_runs, 6L))
    for (i in shown) {
      cat(sprintf(
        "  %-16s %4d observations%s\n",
        names(x$run_lengths)[[i]], x$run_lengths[[i]],
        if (is.null(x$TR)) "" else sprintf("  TR %s s", format(x$TR[[i]]))
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
