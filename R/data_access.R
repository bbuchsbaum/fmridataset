#' @importFrom neuroim2 series
#' @import memoise

#' @export
#' @importFrom neuroim2 NeuroVecSeq 
get_data.latent_dataset <- function(x, ...) {
  x$lvec@basis
}

#' @export
#' @importFrom neuroim2 NeuroVecSeq 
get_data.fmri_mem_dataset <- function(x, ...) {
  if (length(x$scans) > 1) {
    do.call(neuroim2::NeuroVecSeq, x$scans)
  } else {
    x$scans[[1]]
  }
}

#' @export
#' @importFrom neuroim2 NeuroVecSeq 
get_data.matrix_dataset <- function(x, ...) {
  x$datamat
}

#' @export
#' @importFrom neuroim2 NeuroVecSeq FileBackedNeuroVec
get_data.fmri_file_dataset <- function(x, ...) {
  if (!is.null(x$backend)) {
    # New backend path - return raw data matrix
    backend_get_data(x$backend, ...)
  } else if (is.null(x$vec)) {
    # Legacy path
    get_data_from_file(x,...)
  } else {
    x$vec
  }
}

#' @export
get_data_matrix.matrix_dataset <- function(x, ...) {
  x$datamat
}


#' @export
get_data_matrix.fmri_mem_dataset <- function(x, ...) {
  bvec <- get_data(x)
  mask <- get_mask(x)
  series(bvec, which(mask != 0))
}


#' @export
get_data_matrix.fmri_file_dataset <- function(x, ...) {
  if (!is.null(x$backend)) {
    # New backend path - already returns matrix in correct format
    backend_get_data(x$backend, ...)
  } else {
    # Legacy path
    bvec <- get_data(x)
    mask <- get_mask(x)
    series(bvec, which(mask != 0))
  }
}



#' @import memoise
#' @keywords internal
#' @noRd
get_data_from_file <- memoise::memoise(function(x, ...) {
  m <- get_mask(x)
  neuroim2::read_vec(x$scans, mask=m, mode=x$mode, ...)
})



#' @export
get_mask.fmri_file_dataset <- function(x, ...) {
  if (!is.null(x$backend)) {
    # New backend path - returns logical vector
    mask_vec <- backend_get_mask(x$backend)
    # Need to reshape to 3D volume for compatibility
    dims <- backend_get_dims(x$backend)$spatial
    array(mask_vec, dims)
  } else if (is.null(x$mask)) {
    # Legacy path
    neuroim2::read_vol(x$mask_file)
  } else {
    x$mask
  }
}


#' @export
get_mask.fmri_mem_dataset <- function(x, ...) {
  x$mask
}

#' @export
get_mask.matrix_dataset <- function(x, ...) {
  x$mask
}

#' @export
get_mask.latent_dataset <- function(x, ...) {
  x$lvec@mask
}

#' @export
blocklens.matrix_dataset <- function(x, ...) {
  blocklens(x$sampling_frame)
} 