#' Series Selector Classes for fMRI Data
#'
#' A family of S3 classes for specifying spatial selections in fMRI datasets.
#' These selectors provide a type-safe, explicit interface for selecting voxels
#' in \code{fmri_series()} and related functions.
#'
#' @name series_selector
#' @family selectors
#' @importFrom assertthat assert_that
NULL


#' Index-based Series Selector
#'
#' Select voxels by their direct indices in the masked data.
#'
#' @param indices Integer vector of voxel indices
#' @return An object of class \code{index_selector}
#' @export
#' @examples
#' # Select first 10 voxels
#' sel <- index_selector(1:10)
#' 
#' # Select specific voxels
#' sel <- index_selector(c(1, 5, 10, 20))
index_selector <- function(indices) {
  assert_that(is.numeric(indices))
  indices <- as.integer(indices)
  assert_that(all(indices > 0))
  
  structure(
    list(indices = indices),
    class = c("index_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices index_selector
resolve_indices.index_selector <- function(selector, dataset, ...) {
  # Validate indices are within bounds
  n_voxels <- sum(backend_get_mask(dataset$backend))
  invalid <- selector$indices > n_voxels
  if (any(invalid)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = paste0("Index selector contains out-of-bounds indices. ",
                      "Dataset has ", n_voxels, " voxels, but indices up to ",
                      max(selector$indices), " were requested."),
      parameter = "indices",
      value = selector$indices[invalid][1:min(5, sum(invalid))]
    )
  }
  selector$indices
}

#' Voxel Coordinate Series Selector
#'
#' Select voxels by their 3D coordinates in the image space.
#'
#' @param coords Matrix with 3 columns (x, y, z) or vector of length 3
#' @return An object of class \code{voxel_selector}
#' @export
#' @examples
#' # Select single voxel
#' sel <- voxel_selector(c(10, 20, 15))
#' 
#' # Select multiple voxels
#' coords <- cbind(x = c(10, 20), y = c(20, 30), z = c(15, 15))
#' sel <- voxel_selector(coords)
voxel_selector <- function(coords) {
  if (is.vector(coords)) {
    assert_that(length(coords) == 3)
    coords <- matrix(coords, nrow = 1)
  }
  assert_that(is.matrix(coords))
  assert_that(ncol(coords) == 3)
  assert_that(is.numeric(coords))
  
  structure(
    list(coords = coords),
    class = c("voxel_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices voxel_selector
resolve_indices.voxel_selector <- function(selector, dataset, ...) {
  dims <- backend_get_dims(dataset$backend)$spatial
  coords <- selector$coords
  
  # Validate coordinates are within bounds
  invalid_x <- coords[,1] < 1 | coords[,1] > dims[1]
  invalid_y <- coords[,2] < 1 | coords[,2] > dims[2]  
  invalid_z <- coords[,3] < 1 | coords[,3] > dims[3]
  
  if (any(invalid_x | invalid_y | invalid_z)) {
    stop_fmridataset(
      fmridataset_error_config,
      message = paste0("Voxel selector contains out-of-bounds coordinates. ",
                      "Volume dimensions are ", dims[1], "x", dims[2], "x", dims[3]),
      parameter = "coords",
      value = head(coords[invalid_x | invalid_y | invalid_z, , drop = FALSE], 5)
    )
  }
  
  # Convert to linear indices
  ind <- coords[,1] + (coords[,2] - 1) * dims[1] + (coords[,3] - 1) * dims[1] * dims[2]
  
  # Map to masked indices
  mask_vec <- backend_get_mask(dataset$backend)
  mask_ind <- which(mask_vec)
  
  # Find which linear indices are in the mask
  matched <- match(ind, mask_ind)
  not_in_mask <- is.na(matched)
  
  if (any(not_in_mask)) {
    warning("Some requested voxels are outside the dataset mask and will be ignored")
    matched <- matched[!not_in_mask]
  }
  
  if (length(matched) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "No requested voxels are within the dataset mask",
      parameter = "coords",
      value = head(coords, 5)
    )
  }
  
  as.integer(matched)
}

#' ROI-based Series Selector
#'
#' Select voxels within a region of interest (ROI) volume or mask.
#'
#' @param roi A 3D array, ROIVol, LogicalNeuroVol, or similar mask object
#' @return An object of class \code{roi_selector}
#' @export
#' @examples
#' \dontrun{
#' # Using a binary mask
#' mask <- array(FALSE, dim = c(64, 64, 30))
#' mask[30:40, 30:40, 15:20] <- TRUE
#' sel <- roi_selector(mask)
#' }
roi_selector <- function(roi) {
  if (!is.array(roi) && !inherits(roi, c("ROIVol", "LogicalNeuroVol"))) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "ROI must be a 3D array or ROI volume object",
      parameter = "roi",
      value = class(roi)[1]
    )
  }
  
  structure(
    list(roi = roi),
    class = c("roi_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices roi_selector
resolve_indices.roi_selector <- function(selector, dataset, ...) {
  # Get linear indices where ROI is TRUE
  roi_ind <- which(as.logical(as.vector(selector$roi)))
  
  # Get mask indices
  mask_vec <- backend_get_mask(dataset$backend)
  mask_ind <- which(mask_vec)
  
  # Find intersection
  matched <- match(roi_ind, mask_ind)
  matched <- matched[!is.na(matched)]
  
  if (length(matched) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "ROI does not overlap with dataset mask",
      parameter = "roi",
      value = paste0("ROI has ", sum(as.logical(selector$roi)), " voxels")
    )
  }
  
  as.integer(matched)
}

#' Spherical ROI Series Selector
#'
#' Select voxels within a spherical region.
#'
#' @param center Numeric vector of length 3 (x, y, z) specifying sphere center
#' @param radius Numeric radius in voxel units
#' @return An object of class \code{sphere_selector}
#' @export
#' @examples
#' # Select 10-voxel radius sphere around voxel (30, 30, 20)
#' sel <- sphere_selector(center = c(30, 30, 20), radius = 10)
sphere_selector <- function(center, radius) {
  assert_that(is.numeric(center))
  assert_that(length(center) == 3)
  assert_that(is.numeric(radius))
  assert_that(length(radius) == 1)
  assert_that(radius > 0)
  
  structure(
    list(center = center, radius = radius),
    class = c("sphere_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices sphere_selector
resolve_indices.sphere_selector <- function(selector, dataset, ...) {
  dims <- backend_get_dims(dataset$backend)$spatial
  center <- selector$center
  radius <- selector$radius
  
  # Create coordinate grid
  x <- seq_len(dims[1])
  y <- seq_len(dims[2])
  z <- seq_len(dims[3])
  
  # Find voxels within radius
  coords <- expand.grid(x = x, y = y, z = z)
  dist <- sqrt((coords$x - center[1])^2 + 
               (coords$y - center[2])^2 + 
               (coords$z - center[3])^2)
  
  sphere_ind <- which(dist <= radius)
  
  # Get mask indices
  mask_vec <- backend_get_mask(dataset$backend)
  mask_ind <- which(mask_vec)
  
  # Find intersection
  matched <- match(sphere_ind, mask_ind)
  matched <- matched[!is.na(matched)]
  
  if (length(matched) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Spherical ROI does not overlap with dataset mask",
      parameter = c("center", "radius"),
      value = list(center = center, radius = radius)
    )
  }
  
  as.integer(matched)
}

#' All Voxels Series Selector
#'
#' Select all voxels in the dataset mask.
#'
#' @return An object of class \code{all_selector}
#' @export
#' @examples
#' # Select all voxels
#' sel <- all_selector()
all_selector <- function() {
  structure(
    list(),
    class = c("all_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices all_selector
resolve_indices.all_selector <- function(selector, dataset, ...) {
  mask_vec <- backend_get_mask(dataset$backend)
  seq_len(sum(mask_vec))
}

#' Mask-based Series Selector
#'
#' Select voxels that are TRUE in a binary mask.
#'
#' @param mask A logical vector matching the dataset's mask length, or a 3D logical array
#' @return An object of class \code{mask_selector}
#' @export
#' @examples
#' \dontrun{
#' # Using a logical vector
#' mask_vec <- backend_get_mask(dataset$backend)
#' sel <- mask_selector(mask_vec > 0.5)
#' }
mask_selector <- function(mask) {
  if (!is.logical(mask)) {
    mask <- as.logical(mask)
  }
  
  structure(
    list(mask = mask),
    class = c("mask_selector", "series_selector")
  )
}

#' @export
#' @method resolve_indices mask_selector
resolve_indices.mask_selector <- function(selector, dataset, ...) {
  mask <- selector$mask
  
  if (is.array(mask)) {
    # Convert 3D mask to vector
    mask <- as.vector(mask)
  }
  
  # Get dataset mask
  dataset_mask <- backend_get_mask(dataset$backend)
  
  if (length(mask) == length(dataset_mask)) {
    # Full volume mask - extract indices within dataset mask
    mask_ind <- which(dataset_mask)
    selection_ind <- which(mask)
    
    matched <- match(selection_ind, mask_ind)
    matched <- matched[!is.na(matched)]
  } else if (length(mask) == sum(dataset_mask)) {
    # Mask already in masked space
    matched <- which(mask)
  } else {
    stop_fmridataset(
      fmridataset_error_config,
      message = paste0("Mask length (", length(mask), ") does not match ",
                      "volume size (", length(dataset_mask), ") or ",
                      "masked size (", sum(dataset_mask), ")"),
      parameter = "mask",
      value = length(mask)
    )
  }
  
  if (length(matched) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      message = "Mask selector selected no voxels",
      parameter = "mask",
      value = paste0("Mask has ", sum(mask), " TRUE values")
    )
  }
  
  as.integer(matched)
}

#' Print Methods for Series Selectors
#' @export
#' @method print series_selector
print.series_selector <- function(x, ...) {
  cat("<", class(x)[1], ">\n", sep = "")
  invisible(x)
}

#' @export
#' @method print index_selector
print.index_selector <- function(x, ...) {
  cat("<index_selector>\n")
  cat("  indices: ", 
      if(length(x$indices) <= 10) {
        paste(x$indices, collapse = ", ")
      } else {
        paste0(paste(head(x$indices, 5), collapse = ", "), 
               ", ... (", length(x$indices), " total)")
      }, "\n", sep = "")
  invisible(x)
}

#' @export
#' @method print voxel_selector
print.voxel_selector <- function(x, ...) {
  cat("<voxel_selector>\n")
  cat("  coordinates: ", nrow(x$coords), " voxel(s)\n", sep = "")
  if (nrow(x$coords) <= 5) {
    for (i in seq_len(nrow(x$coords))) {
      cat("    [", x$coords[i,1], ", ", x$coords[i,2], ", ", x$coords[i,3], "]\n", sep = "")
    }
  } else {
    for (i in 1:3) {
      cat("    [", x$coords[i,1], ", ", x$coords[i,2], ", ", x$coords[i,3], "]\n", sep = "")
    }
    cat("    ... (", nrow(x$coords) - 3, " more)\n", sep = "")
  }
  invisible(x)
}

#' @export
#' @method print sphere_selector
print.sphere_selector <- function(x, ...) {
  cat("<sphere_selector>\n")
  cat("  center: [", paste(x$center, collapse = ", "), "]\n", sep = "")
  cat("  radius: ", x$radius, " voxels\n", sep = "")
  invisible(x)
}

#' @export
#' @method print roi_selector
print.roi_selector <- function(x, ...) {
  cat("<roi_selector>\n")
  if (is.array(x$roi)) {
    cat("  dimensions: ", paste(dim(x$roi), collapse = " x "), "\n", sep = "")
    cat("  active voxels: ", sum(as.logical(x$roi)), "\n", sep = "")
  } else {
    cat("  type: ", class(x$roi)[1], "\n", sep = "")
  }
  invisible(x)
}

#' @export
#' @method print mask_selector
print.mask_selector <- function(x, ...) {
  cat("<mask_selector>\n")
  cat("  length: ", length(x$mask), "\n", sep = "")
  cat("  TRUE values: ", sum(x$mask), "\n", sep = "")
  invisible(x)
}