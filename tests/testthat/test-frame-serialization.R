test_that("canonical frames serialize without runtime state", {
  fx <- make_frame_fixture(instrument = TRUE)
  expect_false(contains_runtime_state(fx$frame))

  path <- tempfile(fileext = ".rds")
  saveRDS(fx$frame, path)
  reopened <- readRDS(path)

  expect_identical(observation_ids(reopened), observation_ids(fx$frame))
  expect_identical(feature_ids(reopened), feature_ids(fx$frame))
  expect_identical(space_digest(space(reopened)), space_digest(space(fx$frame)))
  expect_equal(collect_assay(reopened), fx$beta)
})

test_that("runtime delarr plans are derived rather than stored", {
  fx <- make_frame_fixture()
  expect_false(any(vapply(assays(fx$frame), function(x) inherits(x$source, "delarr"), logical(1))))
  lazy <- as_delarr(assay(fx$frame)$source)
  expect_s3_class(lazy, "delarr")
})
