#' Deprecated functions
#'
#' These names were replaced so that related functions are spelled the same
#' way. Each one warns, then calls its replacement with the same arguments.
#' They will be removed in the release after the one that deprecated them.
#'
#' | Deprecated | Use instead |
#' |---|---|
#' | `filter_obs()` | [filter_observations()] |
#' | `obs_blocks()` | [observation_blocks()] |
#' | `nifti_array_source()` | [nifti_source()] |
#' | `zarr_array_source()` | [zarr_source()] |
#' | `validate_fds_manifest()` | [validate_fds_frame_manifest()] |
#' | `fds_manifest_digest()` | [fds_frame_manifest_digest()] |
#'
#' The S3 classes `nifti_array_source` and `zarr_array_source`, and the source
#' type strings recorded in descriptors and manifests, are unchanged.
#'
#' @param ... Arguments passed to the replacement.
#' @return The value of the replacement function.
#' @name fmridataset-deprecated
#' @keywords internal
NULL

.deprecated_alias <- function(old, new) {
  .Deprecated(new, package = "fmridataset", old = old)
}

#' @rdname fmridataset-deprecated
#' @export
filter_obs <- function(...) {
  .deprecated_alias("filter_obs", "filter_observations")
  # The predicate is captured by the replacement, so forward the call unevaluated.
  call <- match.call()
  call[[1L]] <- quote(fmridataset::filter_observations)
  eval(call, parent.frame())
}

#' @rdname fmridataset-deprecated
#' @export
obs_blocks <- function(...) {
  .deprecated_alias("obs_blocks", "observation_blocks")
  observation_blocks(...)
}

#' @rdname fmridataset-deprecated
#' @export
nifti_array_source <- function(...) {
  .deprecated_alias("nifti_array_source", "nifti_source")
  nifti_source(...)
}

#' @rdname fmridataset-deprecated
#' @export
zarr_array_source <- function(...) {
  .deprecated_alias("zarr_array_source", "zarr_source")
  zarr_source(...)
}

#' @rdname fmridataset-deprecated
#' @export
validate_fds_manifest <- function(...) {
  .deprecated_alias("validate_fds_manifest", "validate_fds_frame_manifest")
  validate_fds_frame_manifest(...)
}

#' @rdname fmridataset-deprecated
#' @export
fds_manifest_digest <- function(...) {
  .deprecated_alias("fds_manifest_digest", "fds_frame_manifest_digest")
  fds_frame_manifest_digest(...)
}
