test_that("axis IDs are required by default", {
  expect_error(
    axis_frame(data.frame(value = 1:3)),
    class = "fmridataset_error_identity",
    regexp = "requires supplied"
  )

  supplied <- axis_frame(data.frame(.obs_id = c("a", "b")))
  expect_identical(axis_id_policy(supplied)$policy, "require")
  expect_true(ids_are_durable(supplied))
})

test_that("deterministic IDs reconstruct from declared keys", {
  data <- data.frame(
    subject = c("sub-01", "sub-01", "sub-02"),
    run = c("run-1", "run-2", "run-1"),
    index = c(1L, 1L, 1L),
    stringsAsFactors = FALSE
  )
  first <- axis_frame(
    data, id_policy = "deterministic",
    id_keys = c("subject", "run", "index"), id_namespace = "bids-bold"
  )
  second <- axis_frame(
    data, id_policy = "deterministic",
    id_keys = c("subject", "run", "index"), id_namespace = "bids-bold"
  )

  expect_identical(axis_ids(first), axis_ids(second))
  expect_match(axis_ids(first), "^obs-[0-9a-f]{64}$")
  expect_identical(axis_id_policy(first)$keys,
                   c("subject", "run", "index"))
  expect_true(ids_are_durable(first))
  expect_identical(axis_ids(first[c(3L, 1L)]), axis_ids(first)[c(3L, 1L)])

  reconstructed <- axis_frame(
    axis_data(first), id_policy = "deterministic",
    id_keys = c("subject", "run", "index"), id_namespace = "bids-bold"
  )
  expect_identical(axis_ids(reconstructed), axis_ids(first))
})

test_that("deterministic ID collisions and mismatches fail", {
  duplicate_keys <- data.frame(subject = c("sub-01", "sub-01"), run = 1L)
  expect_error(
    axis_frame(
      duplicate_keys, id_policy = "deterministic",
      id_keys = c("subject", "run"), id_namespace = "bids"
    ),
    class = "fmridataset_error_identity",
    regexp = "unique"
  )

  mismatched <- data.frame(.obs_id = "wrong", subject = "sub-01", run = 1L)
  expect_error(
    axis_frame(
      mismatched, id_policy = "deterministic",
      id_keys = c("subject", "run"), id_namespace = "bids"
    ),
    class = "fmridataset_error_identity",
    regexp = "do not match"
  )
})

test_that("ephemeral IDs are visible and cannot be certified or persisted", {
  observations <- axis_frame(
    data.frame(value = 1:2), id_policy = "ephemeral"
  )
  expect_match(axis_ids(observations), "^ephemeral-obs-")
  expect_false(ids_are_durable(observations))
  expect_identical(axis_id_policy(observations)$policy, "ephemeral")

  spatial <- volume_space(c(2L, 1L, 1L), diag(4))
  frame <- fmri_frame(
    assays = list(signal = matrix(1:4, 2L)),
    observations = observations,
    space = spatial
  )
  expect_error(
    fds_frame_manifest(frame),
    class = "fmridataset_error_identity",
    regexp = "ephemeral"
  )
  expect_error(
    identity_descriptor(frame),
    class = "fmridataset_error_identity",
    regexp = "ephemeral"
  )

  durable_observations <- data.frame(.obs_id = c("o1", "o2"))
  implicit_features <- fmri_frame(
    assays = list(signal = matrix(1:4, 2L)),
    observations = durable_observations
  )
  expect_true(ids_are_durable(observation_axis(implicit_features)))
  expect_false(ids_are_durable(space(implicit_features)))
  expect_error(
    fds_frame_manifest(implicit_features),
    class = "fmridataset_error_identity",
    regexp = "ephemeral"
  )
})

test_that("index spaces require durable identity unless explicitly ephemeral", {
  expect_error(index_space(3L), class = "fmridataset_error_identity")

  deterministic <- index_space(
    3L, namespace = "latent-components", id_policy = "deterministic"
  )
  again <- index_space(
    3L, namespace = "latent-components", id_policy = "deterministic"
  )
  expect_identical(feature_ids(deterministic), feature_ids(again))
  expect_true(ids_are_durable(deterministic))

  ephemeral <- index_space(2L, id_policy = "ephemeral")
  expect_match(feature_ids(ephemeral), "^ephemeral-feature-")
  expect_false(ids_are_durable(ephemeral))
})

test_that("FDS round trips retain explicit axis ID policies", {
  fx <- make_frame_fixture()
  manifest <- fds_frame_manifest(fx$frame)

  expect_identical(manifest$axes$observation$id_policy$policy, "require")
  expect_identical(manifest$axes$feature$id_policy$policy, "require")
  expect_identical(manifest$entities$stimulus$id_policy$policy, "require")
  expect_invisible(validate_fds_manifest(manifest))

  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(fx$frame))
  expect_identical(axis_id_policy(observation_axis(rebuilt)),
                   axis_id_policy(observation_axis(fx$frame)))
  expect_identical(axis_id_policy(feature_axis(rebuilt)),
                   axis_id_policy(feature_axis(fx$frame)))
})
