#' Package Startup and Utilities
#' 
#' This file contains package startup functions and utility functions
#' that support the fmridataset package functionality.
#' 
#' @name zzz
NULL

.onLoad <- function(libname, pkgname) {
  # Package startup message (optional, uncomment if desired)
  # packageStartupMessage("fmridataset loaded. Use fmri_dataset_create() to get started.")
  
  # Register S3 methods that might not be automatically detected
  # (Most should be handled by NAMESPACE but this is a backup)
  
  invisible()
}

.onAttach <- function(libname, pkgname) {
  # Startup message shown when package is attached via library()
  packageStartupMessage(
    "fmridataset ", utils::packageVersion("fmridataset"), " loaded.\n",
    "Use fmri_dataset_create() to create datasets from various sources.\n",
    "See vignette('fmridataset-intro') for examples."
  )
} 