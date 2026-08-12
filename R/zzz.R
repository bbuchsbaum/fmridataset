# nocov start
.onLoad <- function(libname, pkgname) {
  op <- options()
  defaults <- list(
    fmridataset.collect_budget = 2 * 1024^3,
    fmridataset.block_budget = 512 * 1024^2,
    fmridataset.target_block_bytes = 4 * 1024^2,
    fmridataset.spatial_budget = 512 * 1024^2
  )
  missing <- !(names(defaults) %in% names(op))
  if (any(missing)) options(defaults[missing])
  invisible()
}
# nocov end
