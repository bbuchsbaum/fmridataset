#' @importFrom assertthat assert_that
#' @importFrom deflist deflist

#' @keywords internal
#' @noRd
data_chunk <- function(mat, voxel_ind, row_ind, chunk_num) {
  ret <- list(
    data = mat,
    voxel_ind = voxel_ind,
    row_ind = row_ind,
    chunk_num = chunk_num
  )

  class(ret) <- c("data_chunk", "list")
  ret
}

#' @keywords internal
#' @noRd
chunk_iter <- function(x, nchunks, get_chunk) {
  chunk_num <- 1

  nextEl <- function() {
    if (chunk_num > nchunks) {
      stop("StopIteration")
    } else {
      ret <- get_chunk(chunk_num)
      chunk_num <<- chunk_num + 1
      ret
    }
  }

  iter <- list(nchunks = nchunks, nextElem = nextEl)
  class(iter) <- c("chunkiter", "abstractiter", "iter")
  iter
}

#' Create Data Chunks for fmri_mem_dataset Objects
#'
#' This function creates data chunks for fmri_mem_dataset objects. It allows for the retrieval of run-wise or sequence-wise data chunks, as well as arbitrary chunks.
#'
#' @param x An object of class 'fmri_mem_dataset'.
#' @param nchunks The number of data chunks to create. Default is 1.
#' @param runwise If TRUE, the data chunks are created run-wise. Default is FALSE.
#' @param ... Additional arguments.
#'
#' @return A list of data chunks, with each chunk containing the data, voxel indices, row indices, and chunk number.
#' @importFrom neuroim2 series
#' @export
#'
#' @examples
#' \dontrun{
#' # Create a simple fmri_mem_dataset for demonstration
#' d <- c(10, 10, 10, 10)
#' nvec <- neuroim2::NeuroVec(array(rnorm(prod(d)), d), space = neuroim2::NeuroSpace(d))
#' mask <- neuroim2::LogicalNeuroVol(array(TRUE, d[1:3]), neuroim2::NeuroSpace(d[1:3]))
#' dset <- fmri_mem_dataset(list(nvec), mask, TR = 2)
#'
#' # Create an iterator with 5 chunks
#' iter <- data_chunks(dset, nchunks = 5)
#' `%do%` <- foreach::`%do%`
#' y <- foreach::foreach(chunk = iter) %do% {
#'   colMeans(chunk$data)
#' }
#' length(y) == 5
#'
#' # Create an iterator with 100 chunks
#' iter <- data_chunks(dset, nchunks = 100)
#' y <- foreach::foreach(chunk = iter) %do% {
#'   colMeans(chunk$data)
#' }
#' length(y) == 100
#'
#' # Create a "runwise" iterator
#' iter <- data_chunks(dset, runwise = TRUE)
#' y <- foreach::foreach(chunk = iter) %do% {
#'   colMeans(chunk$data)
#' }
#' length(y) == 1
#' }
data_chunks.fmri_mem_dataset <- function(x, nchunks = 1, runwise = FALSE, ...) {
  mask <- get_mask(x)
  # print("data chunks")
  # print(nchunks)
  get_run_chunk <- function(chunk_num) {
    bvec <- x$scans[[chunk_num]]
    voxel_ind <- which(mask > 0)
    # print(voxel_ind)
    row_ind <- which(blockids(x$sampling_frame) == chunk_num)
    ret <- data_chunk(neuroim2::series(bvec, voxel_ind),
      voxel_ind = voxel_ind,
      row_ind = row_ind,
      chunk_num = chunk_num
    )
  }

  get_seq_chunk <- function(chunk_num) {
    bvecs <- x$scans
    voxel_ind <- maskSeq[[chunk_num]]
    # print(voxel_ind)

    m <- do.call(rbind, lapply(bvecs, function(bv) neuroim2::series(bv, voxel_ind)))
    ret <- data_chunk(do.call(rbind, lapply(bvecs, function(bv) neuroim2::series(bv, voxel_ind))),
      voxel_ind = voxel_ind,
      row_ind = 1:nrow(m),
      chunk_num = chunk_num
    )
  }

  maskSeq <- NULL
  if (runwise) {
    chunk_iter(x, length(x$scans), get_run_chunk)
  } else if (nchunks == 1) {
    maskSeq <- one_chunk()
    chunk_iter(x, 1, get_seq_chunk)
    # } #else if (nchunks == dim(mask)[3]) {
    # maskSeq <<- slicewise_chunks(x)
    # chunk_iter(x, length(maskSeq), get_seq_chunk)
  } else {
    maskSeq <- arbitrary_chunks(x, nchunks)
    chunk_iter(x, length(maskSeq), get_seq_chunk)
  }
}


#' Create Data Chunks for fmri_file_dataset Objects
#'
#' This function creates data chunks for fmri_file_dataset objects. It allows for the retrieval of run-wise or sequence-wise data chunks, as well as arbitrary chunks.
#'
#' @param x An object of class 'fmri_file_dataset'.
#' @param nchunks The number of data chunks to create. Default is 1.
#' @param runwise If TRUE, the data chunks are created run-wise. Default is FALSE.
#' @param ... Additional arguments.
#'
#' @return A list of data chunks, with each chunk containing the data, voxel indices, row indices, and chunk number.
#' @noRd
data_chunks.fmri_file_dataset <- function(x, nchunks = 1, runwise = FALSE, ...) {
  maskSeq <- NULL

  if (!is.null(x$backend)) {
    # New backend path - stream data directly
    mask_vec <- backend_get_mask(x$backend)
    voxel_ind <- which(mask_vec)
    n_voxels <- sum(mask_vec)
    dims <- backend_get_dims(x$backend)

    get_run_chunk <- function(chunk_num) {
      # Get row indices for this run
      row_ind <- which(blockids(x$sampling_frame) == chunk_num)
      # Stream only the needed rows from backend
      mat <- backend_get_data(x$backend, rows = row_ind, cols = NULL)
      data_chunk(mat, voxel_ind = voxel_ind, row_ind = row_ind, chunk_num = chunk_num)
    }

    get_seq_chunk <- function(chunk_num) {
      # Get column indices for this chunk
      col_ind <- maskSeq[[chunk_num]]
      # Map voxel indices to valid column indices
      valid_cols <- match(col_ind, voxel_ind)
      valid_cols <- valid_cols[!is.na(valid_cols)]
      # Stream only the needed columns from backend
      mat <- backend_get_data(x$backend, rows = NULL, cols = valid_cols)
      data_chunk(mat, voxel_ind = col_ind, row_ind = 1:dims$time, chunk_num = chunk_num)
    }
  } else {
    # Legacy path
    mask <- get_mask(x)

    get_run_chunk <- function(chunk_num) {
      bvec <- neuroim2::read_vec(file.path(x$scans[chunk_num]), mask = mask)
      ret <- data_chunk(bvec@data,
        voxel_ind = which(x$mask > 0),
        row_ind = which(blockids(x$sampling_frame) == chunk_num),
        chunk_num = chunk_num
      )
    }

    get_seq_chunk <- function(chunk_num) {
      v <- get_data(x)
      vind <- maskSeq[[chunk_num]]
      m <- series(v, vind)
      ret <- data_chunk(m,
        voxel_ind = vind,
        row_ind = 1:nrow(x$event_table),
        chunk_num = chunk_num
      )
    }
  }


  # Then create iterator based on strategy
  if (runwise) {
    if (!is.null(x$backend)) {
      # For backend, use number of runs from sampling frame
      chunk_iter(x, x$nruns, get_run_chunk)
    } else {
      # Legacy path uses number of scan files
      chunk_iter(x, length(x$scans), get_run_chunk)
    }
  } else if (nchunks == 1) {
    maskSeq <- one_chunk(x)
    chunk_iter(x, 1, get_seq_chunk)
  } else {
    maskSeq <- arbitrary_chunks(x, nchunks)
    chunk_iter(x, length(maskSeq), get_seq_chunk)
  }
}


#' Create Data Chunks for matrix_dataset Objects
#'
#' This function creates data chunks for matrix_dataset objects. It allows for the retrieval
#' of run-wise or sequence-wise data chunks, as well as arbitrary chunks.
#'
#' @param x An object of class 'matrix_dataset'
#' @param nchunks The number of chunks to split the data into. Default is 1.
#' @param runwise If TRUE, creates run-wise chunks instead of arbitrary chunks
#' @param ... Additional arguments passed to methods
#' @return A list of data chunks, each containing data, indices and chunk number
#' @export
data_chunks.matrix_dataset <- function(x, nchunks = 1, runwise = FALSE, ...) {
  get_run_chunk <- function(chunk_num) {
    ind <- which(blockids(x$sampling_frame) == chunk_num)
    mat <- x$datamat[ind, , drop = FALSE]
    # browser()
    data_chunk(mat, voxel_ind = 1:ncol(mat), row_ind = ind, chunk_num = chunk_num)
  }

  get_one_chunk <- function(chunk_num) {
    data_chunk(x$datamat, voxel_ind = 1:ncol(x$datamat), row_ind = 1:nrow(x$datamat), chunk_num = chunk_num)
  }

  if (runwise) {
    chunk_iter(x, length(blocklens(x$sampling_frame)), get_run_chunk)
  } else if (nchunks == 1) {
    chunk_iter(x, 1, get_one_chunk)
  } else {
    sidx <- split(1:ncol(x$datamat), sort(rep(1:nchunks, length.out = ncol(x$datamat))))
    get_chunk <- function(chunk_num) {
      data_chunk(x$datamat[, sidx[[chunk_num]], drop = FALSE],
        voxel_ind = sidx[[chunk_num]],
        row_ind = 1:nrow(x$datamat),
        chunk_num = chunk_num
      )
    }
    chunk_iter(x, nchunks, get_chunk)
  }
}

#' @keywords internal
#' @noRd
exec_strategy <- function(strategy = c("voxelwise", "runwise", "chunkwise"), nchunks = NULL) {
  strategy <- match.arg(strategy)

  function(dset) {
    if (strategy == "runwise") {
      data_chunks(dset, runwise = TRUE)
    } else if (strategy == "voxelwise") {
      m <- get_mask(dset)
      data_chunks(dset, nchunks = sum(m), runwise = FALSE)
    } else if (strategy == "chunkwise") {
      m <- get_mask(dset)
      ## message("nchunks is", nchunks)
      assert_that(!is.null(nchunks) && is.numeric(nchunks))
      if (nchunks > sum(m)) {
        warning("requested number of chunks is greater than number of voxels in mask")
        nchunks <- sum(m)
      }
      data_chunks(dset, nchunks = nchunks, runwise = FALSE)
    }
  }
}

#' Collect all chunks from a chunk iterator
#' @keywords internal
#' @noRd
collect_chunks <- function(chunk_iter) {
  chunks <- list()
  for (i in seq_len(chunk_iter$nchunks)) {
    chunks[[i]] <- chunk_iter$nextElem()
  }
  chunks
}


#' @keywords internal
#' @noRd
#' @importFrom deflist deflist
arbitrary_chunks <- function(x, nchunks) {
  # print("arbitrary chunks")
  # browser()
  mask <- get_mask(x)
  # print(mask)
  indices <- as.integer(which(mask != 0))

  # If more chunks requested than voxels, cap to number of voxels
  if (nchunks > length(indices)) {
    warning("requested number of chunks (", nchunks, ") is greater than number of voxels (", length(indices), "). Using ", length(indices), " chunks instead.")
    nchunks <- length(indices)
  }

  chsize <- round(length(indices) / nchunks)
  # print(indices)

  assert_that(chsize > 0)
  chunkids <- sort(rep(1:nchunks, each = chsize, length.out = length(indices)))
  # print(chunkids)

  mfun <- function(i) indices[chunkids == i]
  # print(mfun)

  ret <- deflist::deflist(mfun, len = nchunks)
  # print(ret[[1]])
  return(ret)
}

#' @keywords internal
#' @noRd
slicewise_chunks <- function(x) {
  mask <- x$mask
  template <- neuroim2::NeuroVol(array(0, dim(mask)), neuroim2::space(mask))
  nchunks <- dim(mask)[3]

  maskSeq <- lapply(1:nchunks, function(i) {
    m <- template
    m[, , i] <- 1
    m
  })

  maskSeq
}

#' @keywords internal
#' @noRd
one_chunk <- function(x) {
  mask <- get_mask(x)
  voxel_ind <- which(mask > 0)
  list(voxel_ind)
}
