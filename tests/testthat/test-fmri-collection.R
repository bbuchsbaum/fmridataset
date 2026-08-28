.make_collection_frame <- function(subject = "sub-01", n_obs = 2L,
                                   space_variant = 1L,
                                   assay_names = c("beta", "variance"),
                                   condition_levels = c("A", "B"),
                                   extra_observation = FALSE,
                                   block_components = c("tx", "ty"),
                                   entity_extra = FALSE,
                                   index_feature_space = FALSE) {
  observations_data <- tibble::tibble(
    .obs_id = paste0(subject, "-obs-", seq_len(n_obs)),
    subject_id = rep(subject, n_obs),
    condition = factor(rep(c("A", "B"), length.out = n_obs), levels = condition_levels)
  )
  if (extra_observation) observations_data$site <- rep("north", n_obs)
  motion <- axis_block(
    matrix(seq_len(n_obs * length(block_components)), nrow = n_obs),
    components = tibble::tibble(.component_id = block_components),
    role = "confound"
  )
  subject_data <- tibble::tibble(
    subject_id = subject,
    age = if (subject == "sub-01") 63 else 71
  )
  if (entity_extra) subject_data$site <- "north"
  subject_entity <- entity_frame(
    subject_data,
    key = "subject_id",
    blocks = list(
      phenotype = axis_block(
        matrix(c(.2, .8), nrow = 1L),
        components = tibble::tibble(.component_id = c("memory", "attention"))
      )
    )
  )

  if (index_feature_space) {
    feature_space <- index_space(
      3L,
      ids = paste0(subject, "-component-", 1:3),
      namespace = subject
    )
  } else if (space_variant == 1L) {
    feature_space <- volume_space(
      c(2L, 2L, 2L),
      affine = diag(4),
      support = 1:3,
      template = paste0(subject, "-native")
    )
  } else {
    affine <- diag(4)
    affine[1L, 4L] <- 3
    feature_space <- volume_space(
      c(3L, 2L, 2L),
      affine = affine,
      support = c(1L, 2L, 4L, 5L),
      template = paste0(subject, "-native")
    )
  }
  n_feature <- n_features(feature_space)
  values <- matrix(seq_len(n_obs * n_feature), nrow = n_obs)
  sources <- list(
    counting_source(memory_source(values)),
    counting_source(memory_source(values / 10 + .1))
  )
  names(sources) <- assay_names
  feature_info <- feature_data(feature_space)
  feature_info$parcel <- "native"
  frame <- fmri_frame(
    assays = sources,
    observations = axis_frame(observations_data, blocks = list(motion = motion)),
    features = feature_axis(
      feature_info,
      space = feature_space
    ),
    entities = list(subject = subject_entity),
    relations = list(
      observation_subject = key_relation("subject_id", target = "subject")
    ),
    active_assay = assay_names[[1L]]
  )
  list(frame = frame, sources = sources)
}

test_that("fmri_collection owns named equivalent frames in distinct spaces", {
  first <- .make_collection_frame("sub-01", 2L, 1L)
  second <- .make_collection_frame("sub-02", 3L, 2L)
  collection <- fmri_collection(list(native_01 = first$frame, native_02 = second$frame))

  expect_s3_class(collection, "fmri_collection")
  expect_identical(collection_ids(collection), c("native_01", "native_02"))
  expect_identical(names(collection), c("native_01", "native_02"))
  expect_length(collection, 2L)
  expect_identical(collection_frame(collection, "native_01"), first$frame)
  expect_identical(collection[[2L]], second$frame)
  expect_identical(collection_frames(collection), list(
    native_01 = first$frame,
    native_02 = second$frame
  ))
  expect_false(collection_common_space(collection))
  expect_identical(
    collection_space_data(collection),
    tibble::tibble(
      .frame_id = c("native_01", "native_02"),
      n_observation = c(2L, 3L),
      n_feature = c(3L, 4L),
      space_type = c("volume_space", "volume_space"),
      space_digest = c(
        space_digest(space(first$frame)),
        space_digest(space(second$frame))
      )
    )
  )
})

test_that("collection inspection and serialization perform zero numerical reads", {
  first <- .make_collection_frame("sub-01", 2L, 1L)
  second <- .make_collection_frame("sub-02", 3L, 2L)
  collection <- fmri_collection(list(a = first$frame, b = second$frame))

  expect_output(print(collection), "2 frames")
  expect_output(print(collection), "heterogeneous")
  expect_match(collection_digest(collection), "^[0-9a-f]{64}$")
  restored <- unserialize(serialize(collection, NULL))
  expect_identical(collection_digest(restored), collection_digest(collection))
  expect_false(contains_runtime_state(collection))
  for (source in c(first$sources, second$sources)) {
    expect_equal(source_counts(source)$bytes, 0)
  }
})

test_that("collection subsetting preserves stable frame identities", {
  first <- .make_collection_frame("sub-01", 2L, 1L)$frame
  second <- .make_collection_frame("sub-02", 3L, 2L)$frame
  collection <- fmri_collection(list(a = first, b = second))

  selected <- collection[c("b", "a")]
  expect_s3_class(selected, "fmri_collection")
  expect_identical(collection_ids(selected), c("b", "a"))
  expect_identical(selected[["b"]], second)
  expect_error(collection[integer()], class = "fmridataset_error_collection")
  expect_error(collection[c("a", "a")], class = "fmridataset_error_collection")
  expect_error(collection[["missing"]], class = "fmridataset_error_collection")
})

test_that("collections allow synchronized views and detect a common space", {
  first <- .make_collection_frame("sub-01", 3L, 1L)$frame
  second <- .make_collection_frame("sub-02", 2L, 1L)$frame
  second <- fmri_frame(
    assays = lapply(assays(second), `[[`, "source"),
    observations = observation_axis(second),
    features = feature_axis(first),
    entities = entities(second),
    relations = relations(second),
    active_assay = active_assay(second)
  )
  collection <- fmri_collection(list(
    first = first[c(3L, 1L), ],
    second = second
  ))

  expect_true(collection_common_space(collection))
  expect_identical(nrow(collection[["first"]]), 2L)
  expect_identical(feature_ids(collection[["first"]]), feature_ids(first))
})

test_that("collection rejects incompatible assay and axis semantics", {
  reference <- .make_collection_frame("sub-01")$frame
  bad_assay <- .make_collection_frame(
    "sub-02",
    assay_names = c("signal", "variance")
  )$frame
  bad_observation <- .make_collection_frame(
    "sub-02",
    extra_observation = TRUE
  )$frame
  bad_factor <- .make_collection_frame(
    "sub-02",
    condition_levels = c("B", "A")
  )$frame
  bad_block <- .make_collection_frame(
    "sub-02",
    block_components = c("x", "y")
  )$frame

  expect_error(
    fmri_collection(list(a = reference, b = bad_assay)),
    "assay",
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(a = reference, b = bad_observation)),
    "observation",
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(a = reference, b = bad_factor)),
    "observation",
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(a = reference, b = bad_block)),
    "observation block",
    class = "fmridataset_error_collection"
  )
})

test_that("collection validates entity, relation, and feature-domain schemas", {
  reference <- .make_collection_frame("sub-01")$frame
  bad_entity <- .make_collection_frame("sub-02", entity_extra = TRUE)$frame
  bad_space_type <- .make_collection_frame(
    "sub-02",
    index_feature_space = TRUE
  )$frame
  bad_relation <- fmri_frame(
    assays = lapply(assays(reference), `[[`, "source"),
    observations = observation_axis(reference),
    features = feature_axis(reference),
    entities = entities(reference),
    relations = list(
      observation_subject_alias = key_relation("subject_id", target = "subject")
    )
  )

  expect_error(
    fmri_collection(list(a = reference, b = bad_entity)),
    "entity",
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(a = reference, b = bad_space_type)),
    "feature space",
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(a = reference, b = bad_relation)),
    "relation",
    class = "fmridataset_error_collection"
  )
})

test_that("collection constructor requires stable unique frame names", {
  frame <- .make_collection_frame()$frame

  expect_error(fmri_collection(list(frame)), class = "fmridataset_error_collection")
  expect_error(fmri_collection(list()), class = "fmridataset_error_collection")
  expect_error(
    fmri_collection(stats::setNames(list(frame, frame), c("a", "a"))),
    class = "fmridataset_error_collection"
  )

  malformed <- fmri_collection(list(frame = frame))
  malformed$schema_version <- 2L
  expect_error(
    validate_fmri_collection(malformed),
    class = "fmridataset_error_collection"
  )
  expect_error(
    fmri_collection(list(frame = frame), metadata = list(cache = new.env())),
    "runtime",
    class = "fmridataset_error_collection"
  )
})
