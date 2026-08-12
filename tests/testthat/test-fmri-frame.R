test_that("fmri_frame aligns assays and annotated axes", {
  fx <- make_frame_fixture()
  x <- fx$frame

  expect_s3_class(x, "fmri_frame")
  expect_identical(dim(x), c(7L, 6L))
  expect_identical(names(assays(x)), c("beta", "variance"))
  expect_identical(active_assay(x), "beta")
  expect_identical(levels(observations(x)$Fac1), c("A", "B"))
  expect_identical(features(x)$parcel[1:2], c("hippocampus", "hippocampus"))
  expect_identical(feature_ids(x), feature_ids(fx$feature_space))
  expect_identical(class(x), "fmri_frame")
  expect_false(inherits(x, "fmri_dataset"))
})

test_that("fmri_frame rejects assay and feature mismatches", {
  sp <- index_space(3)
  expect_error(
    fmri_frame(
      assays = list(a = matrix(1:8, 2, 4)),
      observations = data.frame(x = 1:2),
      space = sp
    ),
    class = "fmridataset_error_alignment"
  )
})

test_that("frame metadata operations read zero assay bytes", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame

  print(x)
  observations(x)
  features(x)
  space(x)
  expect_true(all(vapply(assays(x), function(a) source_counts(a$source)$bytes, numeric(1)) == 0))
})

test_that("collect_assay enforces its output budget before reading", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame

  expect_error(
    collect_assay(x, memory_budget = 8),
    class = "fmridataset_error_budget"
  )
  expect_equal(source_counts(assays(x)$beta$source)$bytes, 0)
  expect_equal(collect_assay(x), fx$beta)
})

test_that("spatial_map reconstructs one observation", {
  fx <- make_frame_fixture()
  x <- fx$frame
  id <- axis_ids(observation_axis(x))[2]
  img <- spatial_map(x, observation = id)

  expect_equal(vectorize_space(space(x), img), fx$beta[2, ])
})
