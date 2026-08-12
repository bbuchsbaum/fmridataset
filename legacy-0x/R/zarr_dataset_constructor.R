#' Create an fMRI Dataset from Zarr Arrays
#'
#' @description
#' Creates an fMRI dataset object from Zarr array files. Zarr is a cloud-native
#' array format that supports chunked, compressed storage and is ideal for
#' large neuroimaging datasets.
#'
#' @section Experimental:
#' This function uses the CRAN zarr package which is relatively new (v0.1.1, Dec 2025).
#' It supports Zarr v3 format only - Zarr v2 stores cannot be read.
#' Please report any issues to help improve the package.
#'
#' @param zarr_source Path to Zarr store (directory or URL)
#' @param TR The repetition time in seconds
#' @param run_length Vector of integers indicating the number of scans in each run
#' @param event_table Optional data.frame containing event onsets and experimental variables
#' @param censor Optional binary vector indicating which scans to remove
#' @param preload Whether to load all data into memory (default: FALSE)
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
#'
#' @details
#' The Zarr backend expects data organized as a 4D array with dimensions
#' (x, y, z, time). The data is accessed lazily by default, loading only
#' the requested chunks into memory.
#'
#' Zarr stores should contain a single 4D array. For mask data, provide
#' it separately through the fmri_dataset interface if needed.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Local Zarr store
#' dataset <- fmri_zarr_dataset(
#'   "path/to/data.zarr",
#'   TR = 2,
#'   run_length = c(150, 150, 150)
#' )
#'
#' # Remote store
#' dataset <- fmri_zarr_dataset(
#'   "https://example.com/subject01.zarr",
#'   TR = 1.5,
#'   run_length = 300
#' )
#'
#' # Preload small dataset into memory
#' dataset <- fmri_zarr_dataset(
#'   "small_data.zarr",
#'   TR = 2,
#'   run_length = 100,
#'   preload = TRUE
#' )
#' }
#'
#' @seealso
#' \code{\link{zarr_backend}}, \code{\link{fmri_dataset}}
#'
fmri_zarr_dataset <- function(zarr_source,
                              TR,
                              run_length,
                              event_table = data.frame(),
                              censor = NULL,
                              preload = FALSE) {
  # Create zarr backend
  backend <- zarr_backend(
    source = zarr_source,
    preload = preload
  )

  # Use the generic fmri_dataset constructor
  fmri_dataset(
    scans = backend,
    TR = TR,
    run_length = run_length,
    event_table = event_table,
    censor = censor
  )
}
