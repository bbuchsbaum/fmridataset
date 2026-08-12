#' Coerce an object to the canonical frame protocol
#'
#' `as_fmri_frame()` is the non-warning conversion protocol used by companion
#' packages. `upgrade_dataset()` is the user-facing migration entry point for
#' supported 0.x objects and emits a migration warning by default.
#'
#' Supported self-contained inputs are canonical or provisional `fmri_frame`
#' objects, `matrix_dataset`, in-memory `fmri_series`, and `NeuroVec` objects.
#' File/backend datasets are deliberately rejected: their spatial identity and
#' serializable source descriptor must be supplied by a backend-specific
#' adapter rather than inferred from dimensions.
#'
#' @param x An object convertible to an `fmri_frame`.
#' @param ... Method-specific arguments.
#' @param warn Whether `upgrade_dataset()` should report the legacy boundary.
#' @return An `fmri_frame`.
#' @export
as_fmri_frame <- function(x, ...) UseMethod("as_fmri_frame")

.migration_abort <- function(message, field = NULL, ...) {
  .frame_abort(message, "fmridataset_error_schema", field = field, ...)
}

.migration_warning <- function(x, warn) {
  if (isTRUE(warn)) {
    warning(
      "Migrating legacy ", x,
      " to the canonical fmri_frame contract. Persist the returned frame ",
      "to retain its stable axis IDs.",
      call. = FALSE
    )
  }
}

.validate_provisional_frame <- function(x) {
  if (!identical(x$schema_version, 1L)) {
    .migration_abort(
      sprintf(
        "Unsupported provisional fmri_frame schema version '%s'; only version 1 can be upgraded.",
        paste(x$schema_version %||% "missing", collapse = ", ")
      ),
      field = "schema_version",
      supported = 1L,
      actual = x$schema_version
    )
  }
  tryCatch(
    fds_frame_manifest(x),
    fmridataset_error_schema = function(error) stop(error),
    error = function(error) {
      .migration_abort(
        paste0("Invalid provisional fmri_frame: ", conditionMessage(error)),
        field = "frame"
      )
    }
  )
  x
}

.upgrade_provisional_frame <- function(x) {
  if (inherits(x, "fmri_view")) {
    x$base <- .upgrade_provisional_frame(x$base)
    class(x) <- c("fmri_view", "fmri_frame")
  } else {
    class(x) <- "fmri_frame"
  }
  .validate_provisional_frame(x)
}

#' @export
as_fmri_frame.fmri_frame <- function(x, ...) {
  canonical_class <- if (inherits(x, "fmri_view")) {
    c("fmri_view", "fmri_frame")
  } else {
    "fmri_frame"
  }
  if (identical(class(x), canonical_class)) return(x)
  .upgrade_provisional_frame(x)
}

.legacy_observation_data <- function(n, run_length = n, TR = NA_real_) {
  run_length <- as.integer(run_length)
  if (!length(run_length) || anyNA(run_length) || any(run_length <= 0L) ||
      sum(run_length) != n) {
    .migration_abort(
      "Legacy run lengths must be positive and sum to the observation count.",
      field = "run_length"
    )
  }
  TR <- as.numeric(TR)
  if (!length(TR) || anyNA(TR) || any(!is.finite(TR)) || any(TR <= 0)) {
    .migration_abort("Legacy TR must be positive and finite.", field = "TR")
  }
  if (length(TR) == 1L) TR <- rep(TR, length(run_length))
  if (length(TR) != length(run_length)) {
    .migration_abort("Legacy TR must be scalar or have one value per run.", field = "TR")
  }
  run <- rep(seq_along(run_length), run_length)
  run_timepoint <- sequence(run_length)
  data.frame(
    .obs_id = sprintf("legacy-observation-%06d", seq_len(n)),
    run_id = sprintf("run-%03d", run),
    run_timepoint = as.integer(run_timepoint),
    time = (run_timepoint - 1L) * TR[run],
    stringsAsFactors = FALSE
  )
}

.legacy_index_space <- function(n, namespace, data = NULL) {
  ids <- sprintf("%s-feature-%06d", namespace, seq_len(n))
  if (is.null(data)) data <- data.frame(.feature_id = ids)
  data$.feature_id <- ids
  index_space(n, ids = ids, namespace = namespace, data = data)
}

.legacy_namespace <- function(prefix, identity) {
  paste0(prefix, "-", substr(.canonical_digest(identity), 1L, 16L))
}

.legacy_frame_metadata <- function(source_class, ...) {
  list(migration = c(
    list(source_class = source_class, target_schema_version = 1L),
    list(...)
  ))
}

.legacy_table <- function(data, role = "legacy") {
  if (!is.data.frame(data)) {
    .migration_abort("Legacy table data must be a data frame.", field = "tables")
  }
  key <- if ("event_id" %in% names(data) &&
             !anyNA(data$event_id) && !anyDuplicated(as.character(data$event_id))) {
    "event_id"
  } else {
    NULL
  }
  auxiliary_table(data, key = key, role = role,
                  metadata = list(migrated = TRUE))
}

#' @export
as_fmri_frame.matrix_dataset <- function(x, ...) {
  values <- x$datamat
  if (!is.matrix(values)) {
    .migration_abort("matrix_dataset$datamat must be a matrix.", field = "datamat")
  }
  run_length <- x$sampling_frame$blocklens %||% nrow(values)
  TR <- x$sampling_frame$TR %||% x$TR
  spatial <- .legacy_index_space(
    ncol(values),
    .legacy_namespace(
      "legacy-matrix",
      list(values = values, column_names = colnames(values))
    )
  )
  fmri_frame(
    assays = list(signal = memory_source(values)),
    observations = .legacy_observation_data(nrow(values), run_length, TR),
    features = feature_data(spatial),
    space = spatial,
    tables = list(events = .legacy_table(
      x$event_table %||% data.frame(), role = "legacy_events"
    )),
    active_assay = "signal",
    metadata = .legacy_frame_metadata("matrix_dataset")
  )
}

#' @export
as_fmri_frame.fmri_series <- function(x, ...) {
  if (!is.matrix(x$data)) {
    .migration_abort(
      "Only self-contained in-memory fmri_series objects can be upgraded; realize and persist an explicit assay source first.",
      field = "data"
    )
  }
  values <- x$data
  observation <- as.data.frame(x$temporal_info, stringsAsFactors = FALSE)
  observation$.obs_id <- sprintf("legacy-series-observation-%06d", seq_len(nrow(values)))
  observation <- observation[c(".obs_id", setdiff(names(observation), ".obs_id"))]
  feature <- as.data.frame(x$voxel_info, stringsAsFactors = FALSE)
  migration_state <- list(
    selection_info = x$selection_info,
    dataset_info = x$dataset_info
  )
  if (.source_contains_runtime_state(migration_state)) {
    .migration_abort(
      "fmri_series migration metadata cannot contain runtime state.",
      field = "metadata"
    )
  }
  spatial <- .legacy_index_space(
    ncol(values),
    .legacy_namespace(
      "legacy-series",
      list(feature_data = feature, selection_info = x$selection_info)
    ),
    feature
  )
  fmri_frame(
    assays = list(signal = memory_source(values)),
    observations = observation,
    features = feature_data(spatial),
    space = spatial,
    active_assay = "signal",
    metadata = .legacy_frame_metadata(
      "fmri_series",
      dataset_info = x$dataset_info
    ),
    provenance = as_provenance_graph(list(
      source_class = "fmri_series",
      selection_info = x$selection_info,
      dataset_info = x$dataset_info
    ))
  )
}

.legacy_neurovec_space <- function(x) {
  dims <- dim(x)
  spatial_dim <- as.integer(dims[1:3])
  legacy_space <- attr(x, "space", exact = TRUE)
  spacing <- as.numeric(legacy_space$spacing %||% rep(1, 3))[1:3]
  origin <- as.numeric(legacy_space$origin %||% rep(0, 3))[1:3]
  affine <- diag(4)
  diag(affine)[1:3] <- spacing
  affine[1:3, 4] <- origin
  volume_space(
    spatial_dim,
    affine = affine,
    support = seq_len(prod(spatial_dim)),
    template = "legacy-native",
    metadata = list(source_class = class(x)[1L])
  )
}

#' @export
as_fmri_frame.NeuroVec <- function(x, TR, run_length = dim(x)[4L], ...) {
  dims <- dim(x)
  if (length(dims) != 4L || anyNA(dims) || any(dims <= 0L)) {
    .migration_abort("NeuroVec migration requires four positive dimensions.", field = "dim")
  }
  spatial <- .legacy_neurovec_space(x)
  values <- t(matrix(as.numeric(x), nrow = prod(dims[1:3]), ncol = dims[4L]))
  fmri_frame(
    assays = list(signal = memory_source(values)),
    observations = .legacy_observation_data(nrow(values), run_length, TR),
    features = feature_data(spatial),
    space = spatial,
    active_assay = "signal",
    metadata = .legacy_frame_metadata(class(x)[1L])
  )
}

#' @export
as_fmri_frame.fmri_dataset <- function(x, ...) {
  .migration_abort(
    paste0(
      "Legacy ", class(x)[1L],
      " has no supported self-contained migration. Use a backend-specific ",
      "ArraySource and explicit FeatureSpace to construct an fmri_frame."
    ),
    field = "class",
    source_class = class(x)[1L]
  )
}

#' @export
as_fmri_frame.default <- function(x, ...) {
  stop(
    "No as_fmri_frame method is registered for class '",
    class(x)[[1L]], "'.",
    call. = FALSE
  )
}

.is_legacy_series_envelope <- function(x) {
  is.list(x) && identical(sort(names(x)), sort(c("data", "dims", "class"))) &&
    is.matrix(x$data) && identical(as.integer(dim(x$data)), as.integer(x$dims))
}

.upgrade_series_envelope <- function(x) {
  series <- structure(
    list(
      data = x$data,
      voxel_info = data.frame(voxel = seq_len(ncol(x$data))),
      temporal_info = data.frame(timepoint = seq_len(nrow(x$data))),
      selection_info = list(source = "legacy_serialized_envelope"),
      dataset_info = list(serialized_class = x$class)
    ),
    class = "fmri_series"
  )
  as_fmri_frame(series)
}

#' @rdname as_fmri_frame
#' @export
upgrade_dataset <- function(x, ..., warn = TRUE) {
  if (.is_legacy_series_envelope(x)) {
    .migration_warning("fmri_series envelope", warn)
    return(.upgrade_series_envelope(x))
  }
  if (inherits(x, "sampling_frame")) {
    .migration_abort(
      "A sampling_frame is temporal metadata, not assay-bearing data, and cannot be upgraded to an fmri_frame.",
      field = "class"
    )
  }
  provisional <- inherits(x, "fmri_frame") &&
    !identical(class(x), "fmri_frame") &&
    !identical(class(x), c("fmri_view", "fmri_frame"))
  label <- if (provisional) "provisional canonical frame" else class(x)[1L]
  .migration_warning(label, warn)
  as_fmri_frame(x, ...)
}
