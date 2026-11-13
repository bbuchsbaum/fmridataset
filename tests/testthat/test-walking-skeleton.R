# The walking skeleton exercises the storage protocol with fmristore. The
# end-to-end tests that also need multidesign and fmrigds live in
# examples/walking-skeleton/ (outside the package): those packages are not on
# CRAN.

test_that("walking skeleton storage failure leaves no committed result", {
  testthat::skip_if_not_installed("fmristore")
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
