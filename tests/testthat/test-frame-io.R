.skip_without_frame_store <- function() {
  testthat::skip_if_not_installed("fmristore")
  exports <- getNamespaceExports("fmristore")
  testthat::skip_if_not(
    all(c("write_frame_h5", "open_frame_h5") %in% exports),
    "Installed fmristore lacks FDS frame support"
  )
}

test_that("public frame I/O delegates to lazy HDF5 storage", {
  .skip_without_frame_store()
  frame <- make_frame_fixture()$frame
  path <- tempfile(fileext = ".fds.h5")
  on.exit(unlink(path), add = TRUE)

  expect_identical(
    write_frame(frame, path),
    normalizePath(path, winslash = "/", mustWork = TRUE)
  )
  reopened <- open_frame(path)

  expect_s3_class(reopened, "fmri_frame")
  expect_s3_class(assay(reopened, "beta")$source, "h5_array_source")
  expect_identical(observation_ids(reopened), observation_ids(frame))
  expect_identical(feature_ids(reopened), feature_ids(frame))
  expect_identical(space_digest(space(reopened)), space_digest(space(frame)))
  expect_equal(collect_assay(reopened, "beta"), collect_assay(frame, "beta"))

  plan <- as_delarr(assay(reopened, "beta")$source)
  expect_s3_class(plan$seed, "delarr_provider_seed")
  restored_plan <- unserialize(serialize(plan, NULL))
  expect_equal(
    delarr::collect(restored_plan[c(7L, 1L), c(6L, 2L)]),
    collect_assay(frame, "beta")[c(7L, 1L), c(6L, 2L), drop = FALSE]
  )
})

test_that("public frame writer preserves atomic failure semantics", {
  .skip_without_frame_store()
  frame <- make_frame_fixture()$frame
  parent <- tempfile("frame-io-")
  dir.create(parent)
  on.exit(unlink(parent, recursive = TRUE), add = TRUE)
  path <- file.path(parent, "frame.fds.h5")

  withr::local_options(fmristore.frame_writer_fault_after_assay = 1L)
  expect_error(write_frame(frame, path), "Injected")
  expect_false(file.exists(path))
  expect_length(list.files(parent, pattern = "partial", all.files = TRUE), 0L)
})

test_that("frame I/O fails with a structured error without fmristore", {
  frame <- make_frame_fixture()$frame
  local_mocked_bindings(.frame_store_available = function() FALSE)
  expect_error(
    write_frame(frame, tempfile(fileext = ".h5")),
    "r-universe", class = "fmridataset_error_backend_io"
  )
  expect_error(
    open_frame(tempfile(fileext = ".h5")),
    "fmristore", class = "fmridataset_error_backend_io"
  )
})

test_that("frame I/O fails with a structured error when fmristore lacks frame support", {
  frame <- make_frame_fixture()$frame
  local_mocked_bindings(
    .frame_store_available = function() TRUE,
    .frame_store_exports = function() character()
  )
  expect_error(
    write_frame(frame, tempfile(fileext = ".h5")),
    "write_frame_h5", class = "fmridataset_error_backend_io"
  )
  expect_error(
    open_frame(tempfile(fileext = ".h5")),
    "open_frame_h5", class = "fmridataset_error_backend_io"
  )
})

test_that("frame I/O rejects unknown formats", {
  frame <- make_frame_fixture()$frame
  expect_error(write_frame(frame, tempfile(), format = "zarr"), "arg")
  expect_error(open_frame(tempfile(), format = "zarr"), "arg")
})
