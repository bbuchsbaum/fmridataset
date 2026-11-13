.deprecation_fixture <- function() {
  fmri_frame(
    assays = list(signal = matrix(as.double(1:12), nrow = 4)),
    observations = data.frame(
      .obs_id = sprintf("o%d", 1:4),
      run_id = rep(c("r1", "r2"), each = 2L)
    ),
    space = index_space(3L, id_policy = "deterministic", namespace = "deprecated")
  )
}

test_that("deprecated names warn and return what their replacements return", {
  frame <- .deprecation_fixture()

  expect_warning(
    old <- filter_obs(frame, run_id == "r2"),
    "filter_observations", class = "deprecatedWarning"
  )
  expect_identical(observation_ids(old), c("o3", "o4"))
  expect_identical(old, filter_observations(frame, run_id == "r2"))

  expect_warning(
    blocks <- obs_blocks(frame),
    "observation_blocks", class = "deprecatedWarning"
  )
  expect_identical(blocks, observation_blocks(frame))

  manifest <- fds_frame_manifest(frame)
  expect_warning(
    digest <- fds_manifest_digest(manifest),
    "fds_frame_manifest_digest", class = "deprecatedWarning"
  )
  expect_identical(digest, fds_frame_manifest_digest(manifest))
  expect_warning(
    expect_invisible(validate_fds_manifest(manifest)),
    "validate_fds_frame_manifest", class = "deprecatedWarning"
  )
})

test_that("a deprecated filter predicate sees the caller's variables", {
  frame <- .deprecation_fixture()
  wanted <- "r1"
  expect_warning(old <- filter_obs(frame, run_id == wanted), class = "deprecatedWarning")
  expect_identical(observation_ids(old), c("o1", "o2"))
})

test_that("deprecated source constructors keep their class and fingerprint", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  expect_warning(
    old <- nifti_array_source(path, path),
    "nifti_source", class = "deprecatedWarning"
  )
  new <- nifti_source(path, path)
  expect_s3_class(new, "nifti_array_source")
  expect_identical(source_fingerprint(old), source_fingerprint(new))

  args <- list(
    uri = tempfile(fileext = ".zarr"), shape = c(2L, 3L), dtype = "float64",
    chunks = c(1L, 3L)
  )
  new_zarr <- tryCatch(do.call(zarr_source, args), error = identity)
  skip_if(inherits(new_zarr, "error"), "descriptor-only Zarr construction is unavailable")
  expect_warning(
    old_zarr <- do.call(zarr_array_source, args),
    "zarr_source", class = "deprecatedWarning"
  )
  expect_s3_class(new_zarr, "zarr_array_source")
  expect_identical(source_fingerprint(old_zarr), source_fingerprint(new_zarr))
})
