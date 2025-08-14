#' Create an fMRI Dataset from Zarr Arrays
#'
#' @description
#' Creates an fMRI dataset object from Zarr array files. Zarr is a cloud-native
#' array format that supports chunked, compressed storage and is ideal for
#' large neuroimaging datasets.
#'
#' @param zarr_source Path to Zarr store (directory, zip file, or URL)
#' @param data_key Character key for the main data array within the store (default: "data")
#' @param mask_key Character key for the mask array (default: "mask"). Set to NULL for no mask.
#' @param TR The repetition time in seconds
#' @param run_length Vector of integers indicating the number of scans in each run
#' @param event_table Optional data.frame containing event onsets and experimental variables
#' @param censor Optional binary vector indicating which scans to remove
#' @param preload Whether to load all data into memory (default: FALSE)
#' @param cache_size Number of chunks to cache in memory (default: 100)
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
#'
#' @details
#' The Zarr backend expects data organized as a 4D array with dimensions
#' (x, y, z, time). The data is accessed lazily by default, loading only
#' the requested chunks into memory.
#'
#' Zarr stores can be:
#' - Local directories containing .zarr data
#' - Zip files containing zarr arrays  
#' - Remote URLs (S3, GCS, HTTP) for cloud-hosted data
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
#' # Remote S3 store with custom keys
#' dataset <- fmri_zarr_dataset(
#'   "s3://bucket/neuroimaging/subject01.zarr",
#'   data_key = "bold/data",
#'   mask_key = "bold/mask",
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
                             data_key = "data",
                             mask_key = "mask", 
                             TR,
                             run_length,
                             event_table = data.frame(),
                             censor = NULL,
                             preload = FALSE,
                             cache_size = 100) {
  
  # Create zarr backend
  backend <- zarr_backend(
    source = zarr_source,
    data_key = data_key,
    mask_key = mask_key,
    preload = preload,
    cache_size = cache_size
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