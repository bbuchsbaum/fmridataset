# Regression tests for the review findings on the 1.0 contraction PR. Each
# test names the behaviour the finding said was wrong.

# --- FDS v1 manifests written before id_policy and typed metadata -----------
# The schema identity did not change, so a reader must accept the older field
# set rather than refuse artifacts it wrote itself one release earlier.

.old_shape_manifest <- function(manifest) {
  # Strip exactly the fields the earlier writer did not emit.
  manifest$axes <- lapply(manifest$axes, function(axis) {
    axis$id_policy <- NULL
    axis$metadata <- unclass(axis$metadata %||% list())
    axis
  })
  manifest$entities <- lapply(manifest$entities, function(entity) {
    entity$id_policy <- NULL
    entity$metadata <- unclass(entity$metadata %||% list())
    entity
  })
  manifest$metadata <- unclass(manifest$metadata %||% list())
  manifest
}

.manifest_frame <- function() {
  ids <- paste0("feature-", 1:4)
  fmri_frame(
    assays = list(signal = memory_source(matrix(seq_len(12), nrow = 3))),
    observations = data.frame(.obs_id = paste0("obs-", 1:3), run_id = "run-1"),
    space = index_space(4L, ids = ids, namespace = "manifest-fixture")
  )
}

test_that("a v1 manifest without id_policy or typed metadata still validates", {
  frame <- .manifest_frame()
  current <- fds_frame_manifest(frame)
  old <- .old_shape_manifest(current)

  # The old shape really is missing the fields, so this is not a no-op test.
  expect_null(old$axes[[1L]]$id_policy)
  expect_false(inherits(old$metadata, "unaligned_record"))

  expect_invisible(validate_fds_manifest(old))
})

test_that("a v1 manifest without id_policy reconstructs a durable frame", {
  frame <- .manifest_frame()
  old <- .old_shape_manifest(fds_frame_manifest(frame))

  restored <- frame_from_fds_manifest(old, fds_frame_bindings(frame))

  # An ID that was persisted is by definition supplied, so it reads as durable.
  expect_true(ids_are_durable(observation_axis(restored)))
  expect_true(ids_are_durable(space(restored)))
  expect_identical(feature_ids(restored), feature_ids(frame))
  expect_identical(observation_ids(restored), observation_ids(frame))
  expect_identical(observations(restored), observations(frame))
  expect_equal(collect_assay(restored), collect_assay(frame))

  # The policy is reconstructed as `require`, and the ID namespace survives
  # because it is recorded with the space rather than in the absent field.
  restored_policy <- axis_id_policy(space(restored))
  expect_identical(restored_policy$policy, "require")
  expect_identical(
    restored_policy$namespace,
    axis_id_policy(space(frame))$namespace
  )
})

test_that("normalizing an old manifest is otherwise digest-preserving", {
  # Without a namespace there is nothing the old shape omits, so the
  # normalized manifest must be exactly what the writer emits today.
  frame <- fmri_frame(
    assays = list(signal = memory_source(matrix(seq_len(12), nrow = 3))),
    observations = data.frame(.obs_id = paste0("obs-", 1:3)),
    space = index_space(4L, ids = paste0("feature-", 1:4))
  )
  old <- .old_shape_manifest(fds_frame_manifest(frame))
  restored <- frame_from_fds_manifest(old, fds_frame_bindings(frame))

  expect_identical(
    fds_manifest_digest(fds_frame_manifest(restored)),
    fds_manifest_digest(fds_frame_manifest(frame))
  )
})

# --- wrapper realization cost ----------------------------------------------
# A feature-mapped read materializes the contributing source columns and the
# product temporaries, so a budget sized on the presented shape approved a
# read that then allocated orders of magnitude more.

test_that("a feature-mapped source charges the columns its read materializes", {
  n_source <- 200L
  source_space <- index_space(
    n_source,
    ids = sprintf("v-%03d", seq_len(n_source)), namespace = "wrapper-cost"
  )
  target_space <- index_space(1L, ids = "roi-1", namespace = "wrapper-cost-t")
  operator <- matrix(1 / n_source, nrow = 1L, ncol = n_source)
  map <- feature_map(source_space, target_space, operator)

  values <- memory_source(matrix(1, nrow = 4L, ncol = n_source))
  mapped <- feature_mapped_source(values, map)

  presented <- source_realization_cost(mapped)
  expect_identical(presented$shape, c(4L, 1L))

  # The output is one column; the read is not.
  expect_gt(
    presented$estimated_peak_bytes,
    presented$estimated_output_bytes * 10
  )
  # And the intermediates are what carry that weight.
  expect_gt(presented$estimated_temporary_bytes, presented$estimated_output_bytes)
})

test_that("a budget between the mapped output and its peak is refused", {
  n_source <- 200L
  source_space <- index_space(
    n_source,
    ids = sprintf("v-%03d", seq_len(n_source)), namespace = "wrapper-budget"
  )
  target_space <- index_space(1L, ids = "roi-1", namespace = "wrapper-budget-t")
  map <- feature_map(
    source_space, target_space,
    matrix(1 / n_source, nrow = 1L, ncol = n_source)
  )
  mapped <- feature_mapped_source(
    memory_source(matrix(1, nrow = 4L, ncol = n_source)), map
  )
  cost <- source_realization_cost(mapped)

  frame <- fmri_frame(
    assays = list(signal = mapped),
    observations = data.frame(.obs_id = paste0("o-", 1:4)),
    space = target_space
  )

  expect_error(
    collect_assay(frame, memory_budget = cost$estimated_output_bytes),
    class = "fmridataset_error_budget"
  )
  # With room for the intermediates the same read succeeds and is correct.
  collected <- collect_assay(frame, memory_budget = cost$estimated_peak_bytes)
  expect_equal(dim(collected), c(4L, 1L))
  expect_equal(as.numeric(collected), rep(1, 4))
})

# --- filter_entities with entity-addressed relations ------------------------

.validity_study <- function() {
  ids <- paste0("feature-", 1:4)
  space_value <- index_space(4L, ids = ids, namespace = "study-validity")
  subjects <- entity_frame(
    data.frame(subject_id = c("sub-1", "sub-2")),
    key = "subject_id"
  )
  validity <- entity_feature_validity(
    entity = "subject",
    entity_ids = c("sub-1", "sub-2"),
    masks = rbind(
      c(TRUE, TRUE, FALSE, TRUE),
      c(TRUE, FALSE, TRUE, TRUE)
    ),
    space = space_value
  )
  frame <- fmri_frame(
    assays = list(signal = counting_source(memory_source(matrix(seq_len(16), nrow = 4)))),
    observations = data.frame(
      .obs_id = paste0("obs-", 1:4),
      subject_id = c("sub-1", "sub-1", "sub-2", "sub-2"),
      scan_id = c("scan-a", "scan-a", "scan-b", "scan-b")
    ),
    space = space_value,
    entities = list(subject = subjects),
    relations = list(
      observation_subject = key_relation("subject_id"),
      subject_feature_validity = validity
    ),
    tables = list(
      # Keyed by the entity being filtered, which is the case the finding
      # described: a frame-local events table carrying the entity's key.
      events = event_table(
        data.frame(
          event_id = c("e1", "e2"),
          subject_id = c("sub-1", "sub-2"),
          onset = c(1, 2)
        ),
        key = "event_id"
      )
    )
  )
  fmri_study(frames = list(main = frame), entities = list(subject = subjects))
}

test_that("filter_entities keeps a study with a per-entity validity relation", {
  study <- .validity_study()

  filtered <- filter_entities(study, subject, subject_id == "sub-1")

  expect_s3_class(filtered, "fmri_study")
  frame <- study_frame(filtered, "main")
  expect_identical(observation_ids(frame), c("obs-1", "obs-2"))
  # The validity relation is restricted to the surviving entity, not carried
  # whole, which is what made rebuilding the frame fail.
  validity <- relations(frame)$subject_feature_validity
  expect_identical(validity_entity_ids(validity), "sub-1")
  expect_identical(validity_matrix(validity)[1L, ], c(TRUE, TRUE, FALSE, TRUE))
})

test_that("filter_entities reads no assay values", {
  study <- .validity_study()
  source <- assay(study_frame(study, "main"))$source
  reset_source_counts(source)

  filtered <- filter_entities(study, subject, subject_id == "sub-2")
  expect_identical(observation_ids(study_frame(filtered, "main")), c("obs-3", "obs-4"))
  expect_identical(source_counts(source)$values, 0)
})

test_that("filter_entities filters typed tables stored on a frame", {
  study <- .validity_study()

  filtered <- filter_entities(study, subject, subject_id == "sub-1")
  events <- study_frame(filtered, "main")$tables$events

  # e2 belongs to sub-2, which the filter removed; keeping its row would leave
  # a stale record no observation in the returned study refers to.
  expect_identical(table_data(events)$event_id, "e1")
})
