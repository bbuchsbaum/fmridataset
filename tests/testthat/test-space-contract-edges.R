# Two contract edges that both let contradictory or empty input through
# silently rather than being handled explicitly.

test_that("an emptied composite space rebuilds through its own constructor", {
  composite <- make_composite_space_fixture()
  empty <- restrict_space(composite, integer(0))

  expect_equal(n_features(empty), 0L)
  expect_equal(length(feature_ids(empty)), 0L)

  # paste() and data.frame() both recycle a scalar part name against a
  # zero-length index, so an emptied composite used to produce a "part::" key
  # and fail its own "every child feature exactly once" validation, and the
  # default route died on "arguments imply differing number of rows: 1, 0".
  expect_s3_class(
    composite_space(composite_parts(empty), route = empty$route),
    "composite_space"
  )
  expect_s3_class(composite_space(composite_parts(empty)), "composite_space")
})

test_that("an emptied composite keeps a stable digest and restricts again", {
  composite <- make_composite_space_fixture()

  expect_identical(
    space_digest(restrict_space(composite, integer(0))),
    space_digest(restrict_space(composite, integer(0)))
  )
  expect_equal(n_features(restrict_space(composite, c(1, 2))), 2L)
})

test_that("fmri_frame refuses a space that contradicts its feature axis", {
  ids <- c("f1", "f2", "f3")
  index <- index_space(3, ids = ids)
  axis <- feature_axis(tibble::tibble(.feature_id = ids), space = index)
  values <- matrix(as.double(1:9), 3, 3)
  observations <- data.frame(.obs_id = c("o1", "o2", "o3"))

  expect_error(
    fmri_frame(
      list(a = values),
      observations = observations,
      features = axis,
      space = volume_space(c(3, 1, 1), affine = diag(4), support = 1:3)
    ),
    class = "fmridataset_error_space_mismatch"
  )
})

test_that("a space agreeing with the feature axis is accepted", {
  ids <- c("f1", "f2", "f3")
  index <- index_space(3, ids = ids)
  axis <- feature_axis(tibble::tibble(.feature_id = ids), space = index)
  values <- matrix(as.double(1:9), 3, 3)
  observations <- data.frame(.obs_id = c("o1", "o2", "o3"))

  both <- fmri_frame(list(a = values),
    observations = observations,
    features = axis, space = index
  )
  axis_only <- fmri_frame(list(a = values),
    observations = observations,
    features = axis
  )
  space_only <- fmri_frame(list(a = values),
    observations = observations,
    space = index
  )

  expect_identical(space_digest(space(both)), space_digest(index))
  expect_identical(space_digest(space(axis_only)), space_digest(index))
  expect_identical(space_digest(space(space_only)), space_digest(index))
})

test_that("collect_spatial_maps rejects duplicated observations as documented", {
  fixture <- make_frame_fixture()
  frame <- fixture$frame

  expect_error(
    collect_spatial_maps(frame, observations = c(1L, 1L)),
    class = "fmridataset_error_alignment"
  )
})
