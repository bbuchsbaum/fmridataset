test_that("as_fmri_frame preserves canonical frames without realization", {
  fixture <- make_frame_fixture(instrument = TRUE)
  frame <- fixture$frame

  converted <- as_fmri_frame(frame)

  expect_identical(converted, frame)
  expect_true(all(vapply(
    assays(frame),
    function(value) source_counts(value$source)$bytes == 0,
    logical(1)
  )))
})

test_that("as_fmri_frame has an explicit unsupported-object failure", {
  expect_error(
    as_fmri_frame(matrix(1:4, 2L)),
    "No as_fmri_frame method"
  )
})
