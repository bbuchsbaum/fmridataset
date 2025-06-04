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
  if (is.null(x$vec)) {
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
  bvec <- get_data(x)
  mask <- get_mask(x)
  series(bvec, which(mask != 0))
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
  if (is.null(x$mask)) {
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