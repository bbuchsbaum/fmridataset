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
  sp <- index_space(3, namespace = "mismatch", id_policy = "deterministic")
  expect_error(
    fmri_frame(
      assays = list(a = matrix(1:8, 2, 4)),
      observations = data.frame(.obs_id = c("o1", "o2"), x = 1:2),
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

test_that("explain is bounded, descriptive, and reads zero numerical bytes", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame

  summary <- explain(x, sample_size = 2L)

  expect_identical(summary$shape, c(observation = 7L, feature = 6L))
  expect_identical(summary$axes$observation$ids$mode, "sample")
  expect_identical(
    summary$axes$observation$ids$values,
    observation_ids(x)[c(1L, 2L, 6L, 7L)]
  )
  expect_identical(
    summary$axes$feature$ids$values,
    feature_ids(x)[c(1L, 2L, 5L, 6L)]
  )
  expect_match(summary$digests$schema, "^[0-9a-f]{64}$")
  expect_match(summary$digests$semantic, "^[0-9a-f]{64}$")
  expect_identical(summary$space$digest, space_digest(space(x)))
  expect_identical(summary$assays$beta$source_type, "memory_source")
  expect_identical(summary$assays$beta$realization_bytes, 7 * 6 * 8)
  expect_identical(summary$realization$all_assays_bytes, 2 * 7 * 6 * 8)
  expect_true(all(vapply(
    assays(x), function(value) source_counts(value$source)$bytes == 0, logical(1)
  )))
})

test_that("explain separates semantic identity from physical source identity", {
  fx <- make_frame_fixture()
  frame <- fx$frame
  rechunked <- fmri_frame(
    assays = lapply(
      list(beta = fx$beta, variance = fx$variance),
      memory_source,
      chunks = c(1L, 2L)
    ),
    observations = observation_axis(frame),
    features = feature_axis(frame),
    entities = entities(frame),
    relations = relations(frame),
    tables = frame$tables,
    active_assay = active_assay(frame),
    metadata = frame$metadata,
    provenance = frame$provenance
  )

  original <- explain(frame, ids = "none")
  alternate <- explain(rechunked, ids = "none")

  expect_identical(original$digests$schema, alternate$digests$schema)
  expect_identical(original$digests$semantic, alternate$digests$semantic)
  expect_false(identical(
    original$assays$beta$fingerprint,
    alternate$assays$beta$fingerprint
  ))
})

test_that("complete explain IDs require an explicit mode", {
  x <- make_frame_fixture()$frame

  sampled <- explain(x)
  complete <- explain(x, ids = "complete")
  omitted <- explain(x, ids = "none")

  expect_lt(length(sampled$axes$observation$ids$values), nrow(x))
  expect_identical(complete$axes$observation$ids$values, observation_ids(x))
  expect_identical(complete$axes$feature$ids$values, feature_ids(x))
  expect_length(omitted$axes$observation$ids$values, 0L)
  expect_length(omitted$axes$feature$ids$values, 0L)
  expect_error(explain(x, sample_size = -1L), "non-negative integer")
})

test_that("explain reports the visible view and remains small for large axes", {
  observations <- data.frame(.obs_id = sprintf("observation-%06d", seq_len(100000L)))
  spatial <- index_space(2L, ids = c("left", "right"), namespace = "explain-scale")
  source <- counting_source(memory_source(matrix(0, nrow = 100000L, ncol = 2L)))
  frame <- fmri_frame(list(signal = source), observations, space = spatial)
  view <- frame[c(100000L, 1L), 2:1]

  summary <- explain(frame)
  view_summary <- explain(view, ids = "complete")

  expect_lte(as.numeric(object.size(summary)), 50000)
  expect_identical(summary$axes$observation$count, 100000L)
  expect_lte(length(summary$axes$observation$ids$values), 6L)
  expect_identical(view_summary$axes$observation$ids$values, observation_ids(view))
  expect_identical(view_summary$axes$feature$ids$values, feature_ids(view))
  expect_identical(view_summary$shape, c(observation = 2L, feature = 2L))
  expect_identical(source_counts(source)$bytes, 0)
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
