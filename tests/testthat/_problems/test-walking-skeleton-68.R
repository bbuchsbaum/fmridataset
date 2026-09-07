# Extracted from test-walking-skeleton.R:68

# prequel ----------------------------------------------------------------------
.skip_without_walking_skeleton <- function() {
  packages <- c("fmristore", "multidesign", "fmrigds")
  for (package in packages) testthat::skip_if_not_installed(package)
  required <- list(
    fmristore = c("write_frame_h5", "open_frame_h5"),
    multidesign = c("design_spec", "compile_design", "model_matrix"),
    fmrigds = "fit_group"
  )
  for (package in names(required)) {
    testthat::skip_if_not(
      all(required[[package]] %in% getNamespaceExports(package)),
      paste("Installed", package, "lacks frame-native support")
    )
  }
}
.walking_design_spec <- function() {
  multidesign::design_spec(
    fixed = ~ Fac1 * Fac2 + age + mv(stimulus.visual_pca, 1:3),
    random = ~ 1 | subject_id
  )
}

# test -------------------------------------------------------------------------
.skip_without_walking_skeleton()
fixture <- make_walking_skeleton_fixture()
spec <- .walking_design_spec()
memory_fit <- fmrigds::fit_group(
    fixture$frame,
    estimate = "beta",
    variance = "variance",
    design = spec,
    memory_budget = 256 * 1024^2,
    block_size = 2L
  )
