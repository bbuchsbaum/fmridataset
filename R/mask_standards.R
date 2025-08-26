#' Mask Representation Standards
#'
#' @description
#' This package enforces consistent mask representation across all components:
#'
#' @details
#' ## Backend Level (Internal)
#' - `backend_get_mask()` always returns a **logical vector**
#' - Length equals the product of spatial dimensions
#' - TRUE indicates valid voxels, FALSE indicates excluded voxels
#' - No NA values allowed
#' - Must contain at least one TRUE value
#'
#' ## User Level (Public API)
#' - `get_mask()` returns format appropriate to dataset type:
#'   - For volumetric datasets: 3D array or NeuroVol object
#'   - For matrix datasets: logical vector
#'   - For latent datasets: logical vector (components, not voxels)
#'
#' ## Conversion Helpers
#' - `mask_to_logical()`: Convert any mask representation to logical vector
#' - `mask_to_volume()`: Convert logical vector to 3D array given dimensions
#'
#' @name mask-standards
#' @keywords internal
NULL

#' Convert Mask to Logical Vector
#'
#' @description
#' Converts any mask representation to a logical vector.
#'
#' @param mask A mask in any supported format (numeric, logical, array, NeuroVol)
#' @return A logical vector
#' @export
#' @keywords internal
mask_to_logical <- function(mask) {
  if (inherits(mask, "NeuroVol")) {
    as.logical(as.vector(mask))
  } else if (is.array(mask)) {
    as.logical(as.vector(mask))
  } else {
    as.logical(mask)
  }
}

#' Convert Logical Vector to Volume
#'
#' @description
#' Converts a logical vector mask to a 3D array.
#'
#' @param mask_vec A logical vector
#' @param dims Spatial dimensions (length 3)
#' @return A 3D logical array
#' @export
#' @keywords internal
mask_to_volume <- function(mask_vec, dims) {
  if (length(mask_vec) != prod(dims)) {
    stop(sprintf(
      "Mask length (%d) doesn't match spatial dimensions (%s)",
      length(mask_vec), paste(dims, collapse = "x")
    ))
  }
  array(as.logical(mask_vec), dims)
}
