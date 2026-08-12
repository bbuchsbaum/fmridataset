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

.walking_export <- function(package, name) {
  getExportedValue(package, name)
}

.walking_design_spec <- function() {
  .walking_export("multidesign", "design_spec")(
    fixed = ~ Fac1 * Fac2 + age + mv(stimulus.visual_pca, 1:3),
    random = ~ 1 | subject_id
  )
}

test_that("walking skeleton filters metadata and compiles entity blocks without assay I/O", {
  .skip_without_walking_skeleton()
  fixture <- make_walking_skeleton_fixture(instrument = TRUE)
  selected <- fixture$frame |>
    filter_obs(Fac1 == "A" & age >= 60) |>
    select_features(parcel == "hippocampus")

  expect_s3_class(selected, "fmri_view")
  expect_true(all(vapply(
    assays(fixture$frame),
    function(value) source_counts(value$source)$bytes,
    numeric(1)
  ) == 0))

  compiled <- .walking_export("multidesign", "compile_design")(
    fixture$frame, .walking_design_spec()
  )
  actual <- .walking_export("multidesign", "model_matrix")(compiled)
  expect_equal(unname(actual), unname(fixture$dense_design), tolerance = 0)
  expect_equal(
    unname(actual[, 5:7, drop = FALSE]),
    unname(fixture$lifted_pca),
    tolerance = 0
  )
  expect_identical(
    .walking_export("multidesign", "term_data")(compiled)$component_id[5:7],
    c("PC01", "PC02", "PC03")
  )
  expect_true(all(vapply(
    assays(fixture$frame),
    function(value) source_counts(value$source)$bytes,
    numeric(1)
  ) == 0))
})

test_that("walking skeleton agrees across memory, HDF5, block widths, maps, and round trip", {
  .skip_without_walking_skeleton()
  fixture <- make_walking_skeleton_fixture()
  spec <- .walking_design_spec()
  fit_group <- .walking_export("fmrigds", "fit_group")
  memory_fit <- fit_group(
    fixture$frame,
    estimate = "beta",
    variance = "variance",
    design = spec,
    memory_budget = 256 * 1024^2,
    block_size = 2L
  )
  alternate_fit <- fit_group(
    fixture$frame,
    estimate = "beta",
    variance = "variance",
    design = spec,
    memory_budget = 256 * 1024^2,
    block_size = 3L
  )

  input_path <- tempfile(fileext = ".fds.h5")
  result_path <- tempfile(fileext = ".result.fds.h5")
  on.exit(unlink(c(input_path, result_path)), add = TRUE)
  write_frame(fixture$frame, input_path)
  hdf5_frame <- open_frame(input_path)
  hdf5_fit <- fit_group(
    hdf5_frame,
    estimate = "beta",
    variance = "variance",
    design = spec,
    memory_budget = 256 * 1024^2,
    block_size = 2L
  )

  for (assay_name in names(assays(memory_fit$result))) {
    reference <- collect_assay(memory_fit$result, assay_name)
    expect_equal(
      collect_assay(alternate_fit$result, assay_name),
      reference,
      tolerance = 1e-10,
      info = paste(assay_name, "block width")
    )
    expect_equal(
      collect_assay(hdf5_fit$result, assay_name),
      reference,
      tolerance = 1e-10,
      info = paste(assay_name, "HDF5")
    )
  }
  expect_identical(feature_ids(memory_fit$result), feature_ids(fixture$frame))
  expect_identical(
    space_digest(space(memory_fit$result)),
    space_digest(space(fixture$frame))
  )

  map <- spatial_map(memory_fit$result, observation = "Fac1B", assay = "estimate")
  expect_s4_class(map, "NeuroVol")
  expect_identical(dim(map), c(2L, 2L, 2L))

  write_frame(memory_fit$result, result_path)
  reopened <- open_frame(result_path)
  expect_identical(observation_ids(reopened), observation_ids(memory_fit$result))
  expect_identical(feature_ids(reopened), feature_ids(memory_fit$result))
  expect_identical(space_digest(space(reopened)), space_digest(space(memory_fit$result)))
  for (assay_name in names(assays(memory_fit$result))) {
    expect_equal(
      collect_assay(reopened, assay_name),
      collect_assay(memory_fit$result, assay_name),
      tolerance = 0
    )
  }
  serialized <- serialize(hdf5_frame, NULL)
  expect_gt(length(serialized), 0L)
  expect_false(any(vapply(
    assays(hdf5_frame),
    function(value) inherits(value$source, c("environment", "externalptr")),
    logical(1)
  )))
})

test_that("walking skeleton storage failure leaves no committed result", {
  .skip_without_walking_skeleton()
  fixture <- make_walking_skeleton_fixture()
  parent <- tempfile("walking-skeleton-failure-")
  dir.create(parent)
  on.exit(unlink(parent, recursive = TRUE), add = TRUE)
  path <- file.path(parent, "result.fds.h5")

  withr::local_options(fmristore.frame_writer_fault_after_assay = 1L)
  expect_error(write_frame(fixture$frame, path), "Injected")
  expect_false(file.exists(path))
  expect_length(list.files(parent, pattern = "partial", all.files = TRUE), 0L)
})
