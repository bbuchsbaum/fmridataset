.nifti_header_dtype <- function(header) {
  bits <- 8L * as.integer(methods::slot(header, "bytes_per_element"))
  storage <- toupper(as.character(methods::slot(header, "data_type")))
  if (grepl("DOUBLE", storage, fixed = TRUE)) {
    return("float64")
  }
  if (grepl("FLOAT", storage, fixed = TRUE)) {
    return(if (bits <= 32L) "float32" else "float64")
  }
  if (grepl("UINT", storage, fixed = TRUE)) {
    return(paste0("uint", bits))
  }
  if (grepl("INT|SHORT|LONG|BYTE", storage)) {
    return(paste0("int", bits))
  }
  if (grepl("BINARY|LOGICAL", storage)) {
    return("logical")
  }
  "float64"
}

.nifti_header_affine <- function(header) {
  spatial_dim <- as.integer(methods::slot(header, "dims")[1:3])
  neurospace <- neuroim2::NeuroSpace(
    dim = spatial_dim,
    spacing = methods::slot(header, "spacing")[1:3],
    origin = methods::slot(header, "origin"),
    axes = methods::slot(header, "spatial_axes")
  )
  unname(neuroim2::trans(neurospace))
}

.nifti_file_state <- function(paths) {
  info <- file.info(paths)
  lapply(seq_along(paths), function(index) {
    list(
      path = paths[[index]],
      size = unname(info$size[[index]]),
      mtime = as.numeric(info$mtime[[index]])
    )
  })
}

.assert_nifti_source_fresh <- function(x) {
  paths <- c(x$uri, x$mask_uri)
  paths <- paths[!is.na(paths) & nzchar(paths)]
  current <- .nifti_file_state(paths)
  if (!identical(current, x$file_state)) {
    .frame_abort(
      "A NIfTI source file changed after the descriptor was created.",
      "fmridataset_error_backend_io",
      operation = "fingerprint_check",
      files = paths
    )
  }
  invisible(TRUE)
}

.nifti_selected_mask <- function(x, features) {
  support <- x$support[features]
  values <- rep(FALSE, prod(x$spatial_dim))
  values[support] <- TRUE
  neurospace <- neuroim2::NeuroSpace(dim = x$spatial_dim, trans = x$affine)
  neuroim2::LogicalNeuroVol(array(values, dim = x$spatial_dim), neurospace)
}

.nifti_read_vec <- function(...) neuroim2::read_vec(...)

#' Construct a pushdown-aware NIfTI array source
#'
#' The descriptor reads headers and one mask at construction, but no fMRI
#' volumes. Numerical reads split requested global observations by file, pass
#' local volume indices into [neuroim2::read_vec()], and restrict the mask to
#' requested packed features before materialization. Native reads return
#' full-volume `NeuroVec` objects in requested observation order.
#'
#' @param paths One or more NIfTI files with a common spatial grid.
#' @param mask A NIfTI mask path or a compatible `volume_space`.
#' @param chunks Optional logical observation-by-feature chunk hint.
#' @return A serializable `nifti_array_source`.
#' @export
nifti_array_source <- function(paths, mask, chunks = NULL) {
  if (!is.character(paths) || !length(paths) || anyNA(paths) || any(!nzchar(paths))) {
    .frame_abort(
      "paths must contain one or more non-empty NIfTI file names.",
      "fmridataset_error_backend_io",
      operation = "construct"
    )
  }
  missing <- paths[!file.exists(paths)]
  if (length(missing)) {
    .frame_abort(
      paste("NIfTI source files do not exist:", paste(missing, collapse = ", ")),
      "fmridataset_error_backend_io",
      operation = "construct",
      files = missing
    )
  }
  paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
  headers <- lapply(paths, neuroim2::read_header)
  spatial_dims <- lapply(headers, function(header) {
    as.integer(methods::slot(header, "dims")[1:3])
  })
  if (!all(vapply(spatial_dims, identical, logical(1), spatial_dims[[1L]]))) {
    .frame_abort(
      "NIfTI source files do not share spatial dimensions.",
      "fmridataset_error_space_mismatch"
    )
  }
  affines <- lapply(headers, .nifti_header_affine)
  if (!all(vapply(affines, function(value) {
    isTRUE(all.equal(value, affines[[1L]], tolerance = 1e-7, check.attributes = FALSE))
  }, logical(1)))) {
    .frame_abort(
      "NIfTI source files do not share a spatial affine.",
      "fmridataset_error_space_mismatch"
    )
  }
  dtypes <- vapply(headers, .nifti_header_dtype, character(1))
  if (length(unique(dtypes)) != 1L) {
    .frame_abort(
      "NIfTI source files do not share a dtype.",
      "fmridataset_error_source_contract"
    )
  }
  time_per_file <- vapply(headers, function(header) {
    dims <- methods::slot(header, "dims")
    as.integer(if (length(dims) >= 4L) dims[[4L]] else 1L)
  }, integer(1))

  mask_uri <- NA_character_
  if (inherits(mask, "volume_space")) {
    if (!identical(as.integer(mask$dim), spatial_dims[[1L]]) ||
      !isTRUE(all.equal(mask$affine, affines[[1L]], tolerance = 1e-7))) {
      .frame_abort(
        "Mask volume_space is incompatible with the NIfTI files.",
        "fmridataset_error_space_mismatch"
      )
    }
    support <- mask$support
  } else {
    if (!is.character(mask) || length(mask) != 1L || is.na(mask) ||
      !nzchar(mask) || !file.exists(mask)) {
      .frame_abort(
        "mask must be an existing NIfTI path or a volume_space.",
        "fmridataset_error_backend_io",
        operation = "construct"
      )
    }
    mask_uri <- normalizePath(mask, winslash = "/", mustWork = TRUE)
    mask_volume <- suppressWarnings(neuroim2::read_vol(mask_uri))
    if (!identical(as.integer(dim(mask_volume)), spatial_dims[[1L]])) {
      .frame_abort(
        "NIfTI mask dimensions do not match the source files.",
        "fmridataset_error_space_mismatch"
      )
    }
    support <- which(as.logical(as.vector(mask_volume)))
  }
  if (!length(support)) {
    .frame_abort(
      "NIfTI mask contains no active features.",
      "fmridataset_error_space_mismatch"
    )
  }

  shape <- c(sum(time_per_file), length(support))
  chunks <- chunks %||% c(1L, min(length(support), 8192L))
  if (!is.numeric(chunks) || length(chunks) != 2L || anyNA(chunks) ||
    any(chunks <= 0) || any(chunks != as.integer(chunks))) {
    .frame_abort(
      "chunks must contain two positive integers.",
      "fmridataset_error_source_contract",
      field = "chunks",
      actual = chunks
    )
  }
  chunks <- as.integer(chunks)
  chunks <- pmin(chunks, pmax(1L, shape))
  file_paths <- c(paths, mask_uri)
  file_paths <- file_paths[!is.na(file_paths) & nzchar(file_paths)]
  state <- .nifti_file_state(file_paths)
  out <- structure(
    list(
      uri = paths,
      mask_uri = mask_uri,
      spatial_dim = spatial_dims[[1L]],
      affine = affines[[1L]],
      support = as.integer(support),
      time_per_file = time_per_file,
      boundaries = c(0L, cumsum(time_per_file)),
      shape = as.integer(shape),
      dtype = dtypes[[1L]],
      chunks = chunks,
      capabilities = c(
        "row_slice", "column_slice", "block_slice", "native_read", "serializable"
      ),
      file_state = state,
      schema_version = 1L
    ),
    class = c("nifti_array_source", "array_source")
  )
  out$fingerprint <- .canonical_digest(list(
    type = "nifti_array_source",
    schema_version = out$schema_version,
    files = out$file_state,
    spatial_dim = out$spatial_dim,
    affine = out$affine,
    support = out$support,
    time_per_file = out$time_per_file,
    dtype = out$dtype,
    chunks = out$chunks
  ))
  validate_array_source(out)
  out
}

#' Recover the spatial domain of a NIfTI source
#'
#' @param x A `nifti_array_source`.
#' @param template Optional template or native-space label.
#' @return A compatible `volume_space`.
#' @export
nifti_source_space <- function(x, template = NULL) {
  if (!inherits(x, "nifti_array_source")) {
    .frame_abort("x must be a nifti_array_source.", "fmridataset_error_space_mismatch")
  }
  volume_space(
    dim = x$spatial_dim,
    affine = x$affine,
    support = x$support,
    template = template,
    metadata = list(source_fingerprint = x$fingerprint)
  )
}

#' @export
source_shape.nifti_array_source <- function(x, ...) x$shape
#' @export
source_dtype.nifti_array_source <- function(x, ...) x$dtype
#' @export
source_chunks.nifti_array_source <- function(x, ...) x$chunks
#' @export
source_capabilities.nifti_array_source <- function(x, ...) x$capabilities
#' @export
source_fingerprint.nifti_array_source <- function(x, ...) x$fingerprint
#' @export
source_open.nifti_array_source <- function(x, ...) {
  .assert_nifti_source_fresh(x)
  structure(list(source = x), class = c("nifti_array_source_handle", "array_source_handle"))
}
#' @export
source_close.nifti_array_source <- function(x, ...) invisible(TRUE)

#' @export
source_read.nifti_array_source <- function(x, observations = NULL, features = NULL, ...) {
  .assert_nifti_source_fresh(x)
  observations <- .normalize_source_index(observations, x$shape[[1L]])
  features <- .normalize_source_index(features, x$shape[[2L]])
  if (!length(observations) || !length(features)) {
    return(matrix(numeric(), nrow = length(observations), ncol = length(features)))
  }
  selected_mask <- .nifti_selected_mask(x, features)
  selected_support <- x$support[features]
  file_index <- findInterval(
    observations - 1L,
    x$boundaries[-length(x$boundaries)]
  )
  out <- matrix(NA_real_, nrow = length(observations), ncol = length(features))
  for (file in unique(file_index)) {
    at <- which(file_index == file)
    local <- observations[at] - x$boundaries[[file]]
    vec <- suppressWarnings(.nifti_read_vec(
      x$uri[[file]],
      indices = local,
      mask = selected_mask,
      mode = "normal"
    ))
    out[at, ] <- neuroim2::series(vec, selected_support, drop = FALSE)
  }
  out
}

#' @export
source_read_native.nifti_array_source <- function(x, observations = NULL, ...) {
  .assert_nifti_source_fresh(x)
  observations <- .normalize_source_index(observations, x$shape[[1L]])
  if (!length(observations)) {
    return(list())
  }
  volumes <- lapply(observations, function(observation) {
    file <- findInterval(
      observation - 1L,
      x$boundaries[-length(x$boundaries)]
    )
    local <- observation - x$boundaries[[file]]
    suppressWarnings(.nifti_read_vec(
      x$uri[[file]],
      indices = local,
      mode = "normal"
    ))
  })
  if (length(volumes) == 1L) volumes[[1L]] else do.call(neuroim2::NeuroVecSeq, volumes)
}
