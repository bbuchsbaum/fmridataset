# Convenience constructors for run-structured frames.
#
# fmri_frame() is the general constructor. These build the same object from
# the inputs people usually hold (a matrix, neuroim2 volumes, NIfTI files, a
# latent decomposition) by generating the observation axis the temporal
# contract expects: `.obs_id`, `run_id`, `volume_index`, and optionally `TR`
# and `censor`.

.constructor_abort <- function(message, parameter, ...) {
  .frame_abort(message, "fmridataset_error_config", parameter = parameter, ...)
}

.run_lengths_arg <- function(run_length, n_obs, default = n_obs) {
  if (is.null(run_length)) run_length <- default
  if (!is.numeric(run_length) || !length(run_length) || anyNA(run_length) ||
    any(run_length <= 0) || any(run_length != floor(run_length))) {
    .constructor_abort(
      "run_length must contain positive whole numbers, one per run.",
      "run_length",
      value = run_length
    )
  }
  if (sum(run_length) != n_obs) {
    .constructor_abort(
      sprintf(
        "run_length sums to %s but there are %d observations.",
        format(sum(run_length)), n_obs
      ),
      "run_length",
      value = run_length,
      n_observation = n_obs
    )
  }
  as.integer(run_length)
}

.censor_arg <- function(censor, n_obs) {
  if (is.logical(censor)) {
    if (length(censor) != n_obs || anyNA(censor)) {
      .constructor_abort(
        "A logical censor must have one non-missing value per observation.",
        "censor"
      )
    }
    return(censor)
  }
  if (!is.numeric(censor) || anyNA(censor)) {
    .constructor_abort(
      "censor must be logical, a 0/1 vector, or 1-based observation indices.",
      "censor"
    )
  }
  if (length(censor) == n_obs && all(censor %in% c(0, 1))) {
    return(censor == 1)
  }
  if (any(censor < 1) || any(censor > n_obs) || any(censor != floor(censor)) ||
    anyDuplicated(censor)) {
    .constructor_abort(
      paste(
        "A numeric censor must be a 0/1 vector with one value per observation",
        "or distinct 1-based observation indices."
      ),
      "censor"
    )
  }
  out <- logical(n_obs)
  out[as.integer(censor)] <- TRUE
  out
}

.run_observations <- function(run_length, TR = NULL, censor = NULL) {
  n_runs <- length(run_length)
  n_obs <- sum(run_length)
  run <- rep.int(seq_len(n_runs), run_length)
  index <- sequence(run_length)
  obs <- data.frame(
    .obs_id = sprintf("run-%d-vol-%05d", run, index),
    run_id = sprintf("run-%d", run),
    volume_index = index,
    stringsAsFactors = FALSE
  )
  if (!is.null(TR)) {
    if (!is.numeric(TR) || !length(TR) || any(!is.finite(TR)) || any(TR <= 0) ||
      !length(TR) %in% c(1L, n_runs)) {
      .constructor_abort(
        "TR must be positive and finite: one value, or one per run.",
        "TR",
        value = TR
      )
    }
    obs$TR <- rep.int(rep_len(as.numeric(TR), n_runs), run_length)
  }
  if (!is.null(censor)) obs$censor <- .censor_arg(censor, n_obs)
  obs
}

.events_arg <- function(events) {
  if (is.null(events)) {
    return(list())
  }
  if (inherits(events, "fmri_event_table")) {
    return(list(events = events))
  }
  if (!is.data.frame(events)) {
    .constructor_abort(
      "events must be a data frame or an event_table().",
      "events",
      value = class(events)
    )
  }
  if (!nrow(events)) {
    return(list())
  }
  if (!"event_id" %in% names(events)) {
    events <- cbind(
      data.frame(
        event_id = sprintf("event-%d", seq_len(nrow(events))),
        stringsAsFactors = FALSE
      ),
      as.data.frame(events)
    )
  }
  list(events = event_table(events, key = "event_id"))
}

.usable_ids <- function(ids, n) {
  !is.null(ids) && length(ids) == n && !anyNA(ids) && all(nzchar(ids)) &&
    !anyDuplicated(ids)
}

.volume_space_from_neurovol <- function(mask, parameter = "mask") {
  if (inherits(mask, "volume_space")) {
    return(mask)
  }
  if (!methods::is(mask, "NeuroVol")) {
    .constructor_abort(
      sprintf("%s must be a neuroim2 NeuroVol or a volume_space().", parameter),
      parameter,
      value = class(mask)
    )
  }
  support <- which(as.vector(mask) != 0)
  if (!length(support)) {
    .frame_abort(
      sprintf("%s contains no active voxels.", parameter),
      "fmridataset_error_space_mismatch"
    )
  }
  volume_space(
    dim = as.integer(dim(mask)[1:3]),
    affine = unname(neuroim2::trans(neuroim2::space(mask))),
    support = support
  )
}

.assert_on_volume_grid <- function(x, space, what) {
  grid <- neuroim2::space(x)
  same_dim <- identical(as.integer(dim(x)[1:3]), as.integer(space$dim))
  same_affine <- isTRUE(all.equal(
    unname(neuroim2::trans(grid)), unname(space$affine),
    tolerance = 1e-7, check.attributes = FALSE
  ))
  if (!same_dim || !same_affine) {
    .frame_abort(
      sprintf(
        "%s is not on the mask's grid (dimensions and affine must both match).",
        what
      ),
      "fmridataset_error_space_mismatch"
    )
  }
  invisible(TRUE)
}

.assay_list <- function(value, assay) {
  .assert_one_string(assay, "assay", .alignment_abort)
  stats::setNames(list(value), assay)
}

.neurovec_runs_frame <- function(runs, mask, TR, run_length, events, censor,
                                 assay) {
  if (missing(mask) || is.null(mask)) {
    .constructor_abort(
      "mask is required to turn NeuroVec volumes into features.", "mask"
    )
  }
  space <- .volume_space_from_neurovol(mask)
  for (i in seq_along(runs)) {
    .assert_on_volume_grid(runs[[i]], space, sprintf("Run %d", i))
  }
  per_run <- vapply(runs, function(run) as.integer(dim(run)[[4L]]), integer(1))
  run_length <- .run_lengths_arg(run_length, sum(per_run), default = per_run)
  # series() reads only the masked voxels, so a sparse or file-backed NeuroVec
  # is never expanded to a dense 4D array.
  values <- do.call(rbind, lapply(runs, function(run) {
    neuroim2::series(run, space$support)
  }))
  # series() labels columns by voxel index; the space names the features.
  dimnames(values) <- NULL
  fmri_frame(
    assays = .assay_list(values, assay),
    observations = .run_observations(run_length, TR, censor),
    space = space,
    tables = .events_arg(events)
  )
}

#' Build a run-structured frame from common inputs
#'
#' @description
#' [fmri_frame()] is the general constructor. These [as_fmri_frame()] methods
#' and `read_nifti_bold()` cover the common case of volumes acquired in runs,
#' generating the observation metadata that [temporal_schema()] expects so you
#' do not write it by hand.
#'
#' * A **matrix** (observations by features) becomes a frame over an
#'   [index_space()], or over `space` when you supply one.
#' * A `neuroim2` **`NeuroVec`**, or a list of them with one per run, becomes a
#'   frame over the [volume_space()] of `mask`.
#' * A **`LatentNeuroVec`** (from `fmrilatent` or `fmristore`) becomes a frame
#'   of temporal scores over a synthesis-only [basis_space()] whose parent is
#'   the mask's volume space.
#' * `read_nifti_bold()` wraps NIfTI files in a lazy [nifti_source()]: headers
#'   and the mask are read at construction, voxel values only on demand. For
#'   fMRIPrep derivatives use [read_bids_bold()].
#'
#' @details
#' Observations are numbered within runs in the order given. Each frame gets
#' the columns `.obs_id` (`run-<r>-vol-<index>`), `run_id` (`run-<r>`), and
#' `volume_index`, plus `TR` and `censor` when supplied. The IDs are derived
#' from run and volume position, so the same call always produces the same
#' IDs.
#'
#' Matrix feature IDs come from `feature_ids`, else from unique, non-empty
#' column names, else `feature-<j>`. Dimension names are then dropped from the
#' assay because the axes carry the IDs.
#'
#' A `LatentNeuroVec`'s per-voxel `offset` is not part of a basis space and is
#' not carried into the frame; add it back yourself after reconstructing
#' voxel values if you need absolute signal.
#'
#' @param x A numeric matrix, a `NeuroVec`, a list of `NeuroVec` objects (one
#'   per run), or a `LatentNeuroVec`.
#' @param paths Character vector of NIfTI files, one per run, on one grid.
#' @param mask For `NeuroVec` input, a `neuroim2::NeuroVol` whose non-zero
#'   voxels are the features, or a [volume_space()]. `read_nifti_bold()` also
#'   accepts a NIfTI mask path. The mask and the data must share dimensions
#'   and affine.
#' @param TR Repetition time in seconds: one value, or one per run. `NULL`
#'   omits the `TR` column; [as_sampling_frame()] then cannot be used.
#' @param run_length Number of observations in each run. Defaults to one run
#'   for a matrix or `LatentNeuroVec`, and to each input's time dimension for
#'   `NeuroVec` and NIfTI input.
#' @param events Optional events: a data frame, stored as a keyed
#'   [event_table()] named `"events"` (an `event_id` column is added when
#'   absent), or an existing `event_table`.
#' @param censor Optional censoring indicator: a logical vector or 0/1 vector
#'   with one value per observation, or distinct 1-based observation indices.
#'   Stored as the logical `censor` column, `TRUE` meaning excluded.
#' @param feature_ids Optional stable feature IDs, one per matrix column.
#' @param component_ids Optional stable component IDs for a `LatentNeuroVec`.
#'   Defaults to the score or loading column names when they are unique, else
#'   `component-<k>`.
#' @param space Optional feature space for a matrix. It must have one feature
#'   per column; `feature_ids` is then not allowed.
#' @param chunks Optional chunk shape passed to [nifti_source()].
#' @param assay Name of the assay.
#' @param ... Unused; must be empty.
#' @return An `fmri_frame`.
#' @seealso [fmri_frame()], [read_bids_bold()], [temporal_schema()]
#' @examples
#' Y <- matrix(rnorm(20 * 3), nrow = 20, ncol = 3)
#' events <- data.frame(onset = c(2, 12), condition = c("a", "b"), run = 1:2)
#' frame <- as_fmri_frame(Y, TR = 2, run_length = c(10, 10), events = events)
#' temporal_schema(frame)$run_lengths
#'
#' grid <- neuroim2::NeuroSpace(c(2, 2, 1, 4))
#' vec <- neuroim2::NeuroVec(array(rnorm(16), c(2, 2, 1, 4)), grid)
#' mask <- neuroim2::LogicalNeuroVol(
#'   array(c(TRUE, TRUE, FALSE, TRUE), c(2, 2, 1)),
#'   neuroim2::NeuroSpace(c(2, 2, 1))
#' )
#' dim(as_fmri_frame(vec, mask = mask, TR = 2))
#' @name frame-constructors
NULL

#' @rdname frame-constructors
#' @export
as_fmri_frame.matrix <- function(x, TR = NULL, run_length = NULL, events = NULL,
                                 censor = NULL, feature_ids = NULL,
                                 space = NULL, assay = "signal", ...) {
  rlang::check_dots_empty()
  if (!is.numeric(x) && !is.logical(x)) {
    .constructor_abort(
      "x must be a numeric matrix.", "x",
      value = class(x)
    )
  }
  run_length <- .run_lengths_arg(run_length, nrow(x))
  if (is.null(space)) {
    if (is.null(feature_ids)) {
      names <- colnames(x)
      feature_ids <- if (.usable_ids(names, ncol(x))) {
        names
      } else {
        sprintf("feature-%d", seq_len(ncol(x)))
      }
    } else if (!.usable_ids(feature_ids, ncol(x))) {
      .constructor_abort(
        "feature_ids must be distinct, non-empty, and one per column of x.",
        "feature_ids"
      )
    }
    space <- index_space(ncol(x), ids = as.character(feature_ids))
  } else {
    if (!is.null(feature_ids)) {
      .constructor_abort(
        "Supply feature_ids or space, not both: a space names its features.",
        "feature_ids"
      )
    }
    if (n_features(space) != ncol(x)) {
      .frame_abort(
        sprintf(
          "space has %d features but x has %d columns.",
          n_features(space), ncol(x)
        ),
        "fmridataset_error_space_mismatch"
      )
    }
  }
  dimnames(x) <- NULL
  fmri_frame(
    assays = .assay_list(x, assay),
    observations = .run_observations(run_length, TR, censor),
    space = space,
    tables = .events_arg(events)
  )
}

#' @rdname frame-constructors
#' @export
as_fmri_frame.NeuroVec <- function(x, mask, TR = NULL, run_length = NULL,
                                   events = NULL, censor = NULL,
                                   assay = "signal", ...) {
  rlang::check_dots_empty()
  .neurovec_runs_frame(list(x), mask, TR, run_length, events, censor, assay)
}

#' @rdname frame-constructors
#' @export
as_fmri_frame.list <- function(x, mask, TR = NULL, run_length = NULL,
                               events = NULL, censor = NULL,
                               assay = "signal", ...) {
  rlang::check_dots_empty()
  is_vec <- vapply(x, function(run) {
    methods::is(run, "NeuroVec") && !methods::is(run, "LatentNeuroVec")
  }, logical(1))
  if (!length(x) || !all(is_vec)) {
    .constructor_abort(
      "A list passed to as_fmri_frame() must hold NeuroVec objects, one per run.",
      "x",
      value = unique(vapply(x, function(run) class(run)[[1L]], character(1)))
    )
  }
  .neurovec_runs_frame(unname(x), mask, TR, run_length, events, censor, assay)
}

#' @rdname frame-constructors
#' @export
as_fmri_frame.LatentNeuroVec <- function(x, TR = NULL, run_length = NULL,
                                         events = NULL, censor = NULL,
                                         component_ids = NULL,
                                         assay = "scores", ...) {
  rlang::check_dots_empty()
  scores <- as.matrix(methods::slot(x, "basis"))
  loadings <- methods::slot(x, "loadings")
  if (!inherits(loadings, "Matrix")) loadings <- as.matrix(loadings)
  parent <- .volume_space_from_neurovol(methods::slot(x, "mask"), "The latent mask")
  if (ncol(scores) != ncol(loadings)) {
    .frame_abort(
      "Latent scores and loadings disagree on the number of components.",
      "fmridataset_error_alignment"
    )
  }
  if (nrow(loadings) != n_features(parent)) {
    .frame_abort(
      "Latent loadings must have one row per voxel in the mask.",
      "fmridataset_error_space_mismatch"
    )
  }
  n_component <- ncol(loadings)
  if (is.null(component_ids)) {
    component_ids <- sprintf("component-%d", seq_len(n_component))
    for (candidate in list(colnames(scores), colnames(loadings))) {
      if (.usable_ids(candidate, n_component)) {
        component_ids <- candidate
        break
      }
    }
  } else if (!.usable_ids(component_ids, n_component)) {
    .constructor_abort(
      "component_ids must be distinct, non-empty, and one per component.",
      "component_ids"
    )
  }
  run_length <- .run_lengths_arg(run_length, nrow(scores))
  dimnames(scores) <- NULL
  dimnames(loadings) <- NULL
  fmri_frame(
    assays = .assay_list(scores, assay),
    observations = .run_observations(run_length, TR, censor),
    space = basis_space_from_decoder(
      parent,
      component_ids = as.character(component_ids),
      decoder = loadings,
      encoder = "none"
    ),
    tables = .events_arg(events)
  )
}

#' @rdname frame-constructors
#' @export
read_nifti_bold <- function(paths, mask, TR = NULL, run_length = NULL,
                            events = NULL, censor = NULL, chunks = NULL,
                            assay = "signal") {
  if (missing(mask) || is.null(mask)) {
    .constructor_abort("mask is required: a NIfTI path, NeuroVol, or volume_space().", "mask")
  }
  if (!is.character(mask)) mask <- .volume_space_from_neurovol(mask)
  source <- nifti_source(paths, mask, chunks = chunks)
  run_length <- .run_lengths_arg(
    run_length, source_shape(source)[[1L]],
    default = source$time_per_file
  )
  fmri_frame(
    assays = .assay_list(source, assay),
    observations = .run_observations(run_length, TR, censor),
    space = nifti_source_space(source),
    tables = .events_arg(events)
  )
}
