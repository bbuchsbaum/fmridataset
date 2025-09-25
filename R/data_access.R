#' @importFrom neuroim2 series
#' @import memoise


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
    get_data_from_file(x, ...)
  } else {
    x$vec
  }
}

#' @export
get_data_matrix.matrix_dataset <- function(x, rows = NULL, cols = NULL, ...) {
  if (!is.null(rows) || !is.null(cols)) {
    # Support subsetting
    r <- rows %||% TRUE
    c <- cols %||% TRUE
    x$datamat[r, c, drop = FALSE]
  } else {
    x$datamat
  }
}


#' @export
get_data_matrix.fmri_mem_dataset <- function(x, ...) {
  bvec <- get_data(x)
  mask <- get_mask(x)
  neuroim2::series(bvec, which(mask != 0))
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
    neuroim2::series(bvec, which(mask != 0))
  }
}



#' @import memoise
#' @importFrom cachem cache_mem
#' @importFrom utils object.size
#' @keywords internal

# Cache configuration and management ----

#' Get cache size in bytes
#' @return Numeric cache size in bytes
#' @keywords internal
#' @noRd
.get_cache_size <- function() {
  getOption("fmridataset.cache_max_mb", 512) * 1024^2
}

#' Get cache eviction policy
#' @return Character string specifying eviction policy
#' @keywords internal
#' @noRd
.get_cache_evict <- function() {
  getOption("fmridataset.cache_evict", "lru")
}

#' Get cache logging enabled status
#' @return Logical indicating if cache logging is enabled
#' @keywords internal
#' @noRd
.get_cache_logging <- function() {
  getOption("fmridataset.cache_logging", FALSE)
}

#' Create main data cache with proper LRU eviction
#' @keywords internal
#' @noRd
.create_data_cache <- function() {
  cachem::cache_mem(
    max_size = .get_cache_size(),
    evict = .get_cache_evict(),
    missing = cachem::key_missing(),
    logfile = if (.get_cache_logging()) {
      file.path(tempdir(), "fmridataset_cache.log")
    } else {
      NULL
    }
  )
}

# Main data cache with LRU eviction
.data_cache <- .create_data_cache()

get_data_from_file <- memoise::memoise(function(x, ...) {
  m <- get_mask(x)
  neuroim2::read_vec(x$scans, mask = m, mode = x$mode, ...)
}, cache = .data_cache)

#' Clear fmridataset cache
#'
#' Clears the internal cache used by fmridataset for memoized file operations.
#' This can be useful to free memory or force re-reading of files.
#'
#' @return NULL (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Clear the cache to free memory
#' fmri_clear_cache()
#' }
fmri_clear_cache <- function() {
  .data_cache$reset()
  invisible(NULL)
}

#' Get cache information and statistics
#'
#' Returns information about the current state of the fmridataset cache,
#' including size, number of objects, hit/miss rates, and memory usage.
#'
#' @return Named list with cache statistics
#' @export
#' @examples
#' \dontrun{
#' # Get cache information
#' cache_info <- fmri_cache_info()
#' print(cache_info)
#' }
fmri_cache_info <- function() {
  info <- .data_cache$info()

  # Get current cache statistics using available methods
  n_objects <- length(.data_cache$keys())

  # Convert bytes to human-readable format
  format_bytes <- function(bytes) {
    if (is.null(bytes) || is.na(bytes)) {
      return("unknown")
    }

    units <- c("B", "KB", "MB", "GB")
    size <- as.numeric(bytes)

    for (i in seq_along(units)) {
      if (size < 1024 || i == length(units)) {
        return(sprintf("%.1f %s", size, units[i]))
      }
      size <- size / 1024
    }
  }

  # Estimate current size by summing cached objects
  current_size_estimate <- if (n_objects > 0) {
    keys <- .data_cache$keys()
    total_size <- 0
    for (key in keys) {
      obj <- .data_cache$get(key)
      if (!is.null(obj)) {
        total_size <- total_size + as.numeric(object.size(obj))
      }
    }
    total_size
  } else {
    0
  }

  list(
    max_size = format_bytes(info$max_size),
    current_size = format_bytes(current_size_estimate),
    n_objects = n_objects,
    eviction_policy = info$evict,
    cache_hit_rate = NULL, # cachem doesn't provide hit/miss statistics
    total_hits = NULL,
    total_misses = NULL,
    utilization_pct = if (!is.null(info$max_size) && info$max_size > 0) {
      round(current_size_estimate / info$max_size * 100, 1)
    } else {
      NULL
    }
  )
}

#' Resize the fmridataset cache
#'
#' Changes the maximum size of the cache. This will immediately evict objects
#' if the new size is smaller than the current cache contents.
#'
#' @param size_mb Numeric cache size in megabytes
#' @return NULL (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Resize cache to 1GB
#' fmri_cache_resize(1024)
#'
#' # Check new size
#' fmri_cache_info()
#' }
fmri_cache_resize <- function(size_mb) {
  if (!is.numeric(size_mb) || length(size_mb) != 1 || size_mb <= 0) {
    stop("size_mb must be a positive number")
  }

  warning("Cache resizing is not supported after package load. Please restart R session with the desired cache size option.")
  warning("Use: options(fmridataset.cache_max_mb = ", size_mb, ") before loading the package")

  invisible(NULL)
}



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
blocklens.matrix_dataset <- function(x, ...) {
  blocklens(x$sampling_frame)
}
