#' S4 Compatibility layer for storage backends
#'
#' Registers the storage backend S3 classes so that S4 generics can dispatch on
#' them without warnings.
#' @keywords internal
setOldClass("storage_backend")
setOldClass(c("nifti_backend", "storage_backend"))
setOldClass(c("matrix_backend", "storage_backend"))
