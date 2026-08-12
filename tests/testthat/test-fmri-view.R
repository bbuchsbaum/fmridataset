test_that("frame slicing returns a synchronized non-dropping view", {
  fx <- make_frame_fixture()
  x <- fx$frame
  y <- x[c(7, 2), c(6, 1), drop = FALSE]

  expect_s3_class(y, "fmri_view")
  expect_identical(dim(y), c(2L, 2L))
  expect_identical(observation_ids(y), observation_ids(x)[c(7, 2)])
  expect_identical(feature_ids(y), feature_ids(x)[c(6, 1)])
  expect_equal(collect_assay(y), fx$beta[c(7, 2), c(6, 1), drop = FALSE])
  expect_identical(dim(axis_block_data(obs_blocks(y)$motion)), c(2L, 2L))
  expect_error(x[1, 1, drop = TRUE], class = "fmridataset_error_alignment")
})

test_that("views compose selectors back to the base frame", {
  fx <- make_frame_fixture()
  x <- fx$frame
  y <- x[c(7, 2, 5), c(6, 1, 4)]
  z <- y[c(3, 1), c(2, 3)]

  expect_equal(collect_assay(z), fx$beta[c(5, 7), c(1, 4), drop = FALSE])
})

test_that("view assay descriptors expose the visible rectangle", {
  fx <- make_frame_fixture(instrument = TRUE)
  base <- fx$frame
  view <- base[c(7L, 2L, 5L), c(6L, 1L, 4L)][c(3L, 1L), c(2L, 3L)]

  visible <- assays(view)
  expect_s3_class(visible, "aligned_assay_set")
  expect_identical(names(visible), names(assays(base)))

  for (name in names(visible)) {
    descriptor <- assay(view, name)
    original <- assay(base, name)

    expect_s3_class(descriptor$source, "source_view")
    expect_identical(source_shape(descriptor$source), dim(view))
    expect_identical(descriptor$name, original$name)
    expect_identical(descriptor$dtype, original$dtype)
    expect_identical(descriptor$role, original$role)
    expect_identical(descriptor$units, original$units)
    expect_identical(descriptor$metadata, original$metadata)
    expect_identical(
      descriptor$observation_digest,
      fmridataset:::.axis_digest(observation_axis(view))
    )
    expect_identical(
      descriptor$feature_digest,
      fmridataset:::.axis_digest(feature_axis(view))
    )
  }

  expect_true(all(vapply(
    assays(base),
    function(value) source_counts(value$source)$bytes,
    numeric(1)
  ) == 0))
})

test_that("view assay descriptors support IDs and empty axes", {
  fx <- make_frame_fixture()
  x <- fx$frame
  by_id <- x[observation_ids(x)[c(6L, 1L)], feature_ids(x)[c(5L, 2L)]]
  empty <- by_id[integer(), integer()]

  expect_identical(source_shape(assay(by_id, "beta")$source), c(2L, 2L))
  expect_identical(
    source_read(assay(by_id, "beta")$source),
    fx$beta[c(6L, 1L), c(5L, 2L), drop = FALSE]
  )
  expect_identical(source_shape(assay(empty, "beta")$source), c(0L, 0L))
  expect_identical(dim(source_read(assay(empty, "beta")$source)), c(0L, 0L))
})

test_that("duplicate missing and out-of-range selectors are rejected", {
  x <- make_frame_fixture()$frame
  expect_error(x[c(1, 1), ], class = "fmridataset_error_alignment")
  expect_error(x[c(TRUE, NA, rep(FALSE, 5)), ], class = "fmridataset_error_alignment")
  expect_error(x[8, ], class = "fmridataset_error_alignment")
})

test_that("filter_obs and select_features are metadata-only", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame
  y <- x |>
    filter_obs(Fac1 == "A" & accuracy > .85) |>
    select_features(parcel == "hippocampus")

  expect_identical(dim(y), c(3L, 2L))
  expect_true(all(vapply(assays(x), function(a) source_counts(a$source)$bytes, numeric(1)) == 0))
  expect_equal(collect_assay(y), fx$beta[c(1, 3, 6), 1:2, drop = FALSE])
})

test_that("zero-length views remain two dimensional", {
  x <- make_frame_fixture()$frame
  y <- x[integer(), integer()]
  expect_identical(dim(y), c(0L, 0L))
  expect_identical(dim(collect_assay(y)), c(0L, 0L))
})
