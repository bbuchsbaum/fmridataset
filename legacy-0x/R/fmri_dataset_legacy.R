#' Legacy fMRI Dataset Constructor
#'
#' @description
#' Backward compatibility wrapper for fmri_dataset. This function provides
#' the same interface as the original fmri_dataset function.
#'
#' @param scans Either a character vector of file paths to scans or a list of NeuroVec objects
#' @param mask Either a character file path to a mask or a NeuroVol mask object
#' @param TR The repetition time
#' @param run_length Numeric vector of run lengths
#' @param preload Whether to preload data into memory
#' @param ... Additional arguments passed to fmri_dataset
#'
#' @return An fmri_dataset object
#' @export
#'
#' @examples
#' \dontrun{
#' # Create dataset from files
#' dset <- fmri_dataset_legacy(
#'   scans = c("scan1.nii", "scan2.nii"),
#'   mask = "mask.nii",
#'   TR = 2,
#'   run_length = c(100, 100)
#' )
#' }
fmri_dataset_legacy <- function(scans, mask, TR, run_length, preload = FALSE, ...) {
  # Simply delegate to fmri_dataset with the same arguments
  fmri_dataset(
    scans = scans,
    mask = mask,
    TR = TR,
    run_length = run_length,
    preload = preload,
    ...
  )
}
