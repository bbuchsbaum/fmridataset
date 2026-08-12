# Backend test helper functions for fmridataset package
# Provides shared test data generators for backend testing

#' Create a test Zarr store
#'
#' @param dims Numeric vector of length 4 (x, y, z, time)
#' @param path Character path to Zarr store (default: tempfile)
#' @return Character path to created Zarr store
create_test_zarr <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")

  # Generate reproducible data
  set.seed(42)
  data_array <- array(rnorm(prod(dims)), dim = dims)

  # Use temporary path if not specified
  if (is.null(path)) {
    path <- tempfile(pattern = "zarr_test_")
  }

  # Create Zarr store using CRAN zarr package
  # as_zarr creates a single-array store at the location
  zarr::as_zarr(data_array, location = path)

  path
}

#' Create a test HDF5 file
#'
#' @param dims Numeric vector of length 4 (x, y, z, time)
#' @param path Character path to HDF5 file (default: tempfile)
#' @return Character path to created HDF5 file
create_test_h5 <- function(dims = c(4, 4, 4, 10), path = NULL) {
  skip_if_not_installed("hdf5r")

  # Generate reproducible data
  set.seed(42)
  data_array <- array(rnorm(prod(dims)), dim = dims)

  # Use temporary path if not specified
  if (is.null(path)) {
    path <- tempfile(pattern = "h5_test_", fileext = ".h5")
  }

  # Create HDF5 file
  h5file <- hdf5r::H5File$new(path, mode = "w")

  # Ensure cleanup on error
  on.exit(
    if (h5file$is_valid) h5file$close_all(),
    add = TRUE,
    after = FALSE
  )

  # Create dataset
  h5file[["data"]] <- data_array

  # Close file
  h5file$close_all()

  # Cancel cleanup handler since we closed successfully
  on.exit(NULL)

  path
}

#' Create a test matrix
#'
#' @param n_time Integer number of timepoints (rows)
#' @param n_voxels Integer number of voxels (columns)
#' @return Matrix of dimensions (n_time x n_voxels)
create_test_matrix <- function(n_time = 10, n_voxels = 64) {
  # Generate reproducible data
  set.seed(42)
  matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels)
}

#' Create a test mask
#'
#' @param n_voxels Integer number of voxels
#' @param proportion_valid Numeric proportion of TRUE values (0 to 1)
#' @return Logical vector of length n_voxels
create_test_mask <- function(n_voxels = 64, proportion_valid = 0.8) {
  # Generate reproducible mask
  set.seed(42)
  n_valid <- round(n_voxels * proportion_valid)
  mask <- rep(FALSE, n_voxels)
  mask[sample(n_voxels, n_valid)] <- TRUE
  mask
}
