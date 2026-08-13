.make_resolution_frame <- function(missing_stimulus = FALSE,
                                   direct_subject = FALSE,
                                   conflicting_subject = FALSE,
                                   scalar_collision = FALSE,
                                   block_collision = FALSE) {
  subject_scores <- counting_source(memory_source(matrix(
    c(.2, .8, .4, .6),
    nrow = 2L, byrow = TRUE
  )))
  stimulus_pca <- counting_source(memory_source(matrix(
    c(.1, .2, .3, .4, .5, .6),
    nrow = 3L, byrow = TRUE
  )))
  run_qc <- counting_source(memory_source(matrix(c(.9, .8, .7), ncol = 1L)))
  subject <- entity_frame(
    tibble::tibble(
      subject_id = c("sub-02", "sub-01"),
      age = c(71, 63),
      cohort = factor(c("older", "younger"), levels = c("younger", "older"))
    ),
    key = "subject_id",
    blocks = list(
      phenotype = axis_block(
        subject_scores,
        components = tibble::tibble(.component_id = c("memory", "attention"))
      )
    )
  )
  session <- entity_frame(
    tibble::tibble(
      session_id = c("ses-01b", "ses-02a", "ses-01a"),
      subject_id = c("sub-01", "sub-02", "sub-01"),
      scanner = c("A", "B", "A")
    ),
    key = "session_id"
  )
  run <- entity_frame(
    tibble::tibble(
      run_id = c("run-02a", "run-01b", "run-01a"),
      session_id = c("ses-02a", "ses-01b", "ses-01a"),
      tr = c(1.5, 2, 2)
    ),
    key = "run_id",
    blocks = list(
      qc = axis_block(
        run_qc,
        components = tibble::tibble(.component_id = "quality")
      )
    )
  )
  stimulus <- entity_frame(
    tibble::tibble(
      stimulus_id = c("stim-1", "stim-2", "stim-3"),
      category = factor(
        c("face", "scene", "object"),
        levels = c("face", "scene", "object")
      )
    ),
    key = "stimulus_id",
    blocks = list(
      visual_pca = axis_block(
        stimulus_pca,
        components = tibble::tibble(.component_id = c("PC01", "PC02")),
        role = "embedding"
      )
    )
  )

  stimulus_id <- c("stim-1", "stim-2", "stim-1", "stim-3", "stim-2")
  if (missing_stimulus) stimulus_id[[2L]] <- NA_character_
  observations_data <- tibble::tibble(
    .obs_id = paste0("obs-", 1:5),
    run_id = c("run-01a", "run-01b", "run-02a", "run-01a", "run-02a"),
    stimulus_id = stimulus_id,
    condition = factor(c("A", "B", "A", "B", "A"), levels = c("A", "B"))
  )
  if (direct_subject) {
    observations_data$subject_id <- c("sub-01", "sub-01", "sub-02", "sub-01", "sub-02")
    if (conflicting_subject) observations_data$subject_id[[1L]] <- "sub-02"
  }
  if (scalar_collision) observations_data$subject.age <- seq_len(5L)
  local_blocks <- list(
    motion = axis_block(
      matrix(seq_len(10L), nrow = 5L),
      components = tibble::tibble(.component_id = c("x", "y")),
      role = "confound"
    )
  )
  if (block_collision) local_blocks$stimulus.visual_pca <- local_blocks$motion

  relation_values <- list(
    observation_run = key_relation("run_id", target = "run"),
    run_session = key_relation("session_id", source = "run", target = "session"),
    session_subject = key_relation(
      "subject_id",
      source = "session", target = "subject"
    ),
    observation_stimulus = key_relation(
      "stimulus_id",
      target = "stimulus", allow_missing = missing_stimulus
    )
  )
  if (direct_subject) {
    relation_values$observation_subject <- key_relation(
      "subject_id",
      target = "subject"
    )
  }

  assay_source <- counting_source(memory_source(matrix(seq_len(15L), nrow = 5L)))
  frame <- fmri_frame(
    assays = list(beta = assay_source),
    observations = axis_frame(observations_data, blocks = local_blocks),
    entities = list(
      subject = subject,
      session = session,
      run = run,
      stimulus = stimulus
    ),
    relations = relation_values
  )
  list(
    frame = frame,
    subject_scores = subject_scores,
    stimulus_pca = stimulus_pca,
    run_qc = run_qc,
    assay_source = assay_source
  )
}

test_that("resolved observations flatten reachable entity scalar metadata", {
  fixture <- .make_resolution_frame()
  raw <- observations(fixture$frame)
  resolved <- observations(fixture$frame, resolve = TRUE)

  expect_identical(raw, observations(fixture$frame, resolve = FALSE))
  expect_identical(names(raw), c(".obs_id", "run_id", "stimulus_id", "condition"))
  expect_true(all(c(
    "subject.subject_id", "subject.age", "subject.cohort",
    "session.session_id", "session.scanner", "run.tr",
    "stimulus.category"
  ) %in% names(resolved)))
  expect_identical(
    resolved$subject.subject_id,
    c("sub-01", "sub-01", "sub-02", "sub-01", "sub-02")
  )
  expect_identical(resolved$subject.age, c(63, 63, 71, 63, 71))
  expect_identical(levels(resolved$subject.cohort), c("younger", "older"))
  expect_identical(
    as.character(resolved$stimulus.category),
    c("face", "scene", "face", "object", "scene")
  )
  expect_equal(source_counts(fixture$assay_source)$bytes, 0)
  expect_equal(source_counts(fixture$subject_scores)$bytes, 0)
  expect_equal(source_counts(fixture$stimulus_pca)$bytes, 0)
})

test_that("resolved metadata powers zero-read filtering and follows views", {
  fixture <- .make_resolution_frame()
  selected <- filter_obs(
    fixture$frame,
    subject.age >= 70 & stimulus.category != "scene"
  )

  expect_identical(observation_ids(selected), "obs-3")
  expect_identical(observations(selected, resolve = TRUE)$subject.age, 71)
  reordered <- fixture$frame[c(5L, 2L, 1L), ]
  expect_identical(
    observations(reordered, resolve = TRUE)$run.run_id,
    c("run-02a", "run-01b", "run-01a")
  )
  expect_equal(source_counts(fixture$assay_source)$bytes, 0)
})

test_that("entity blocks lift lazily to the observation axis", {
  fixture <- .make_resolution_frame()
  blocks <- obs_blocks(fixture$frame, resolve = TRUE)

  expect_identical(
    names(blocks),
    c("motion", "subject.phenotype", "run.qc", "stimulus.visual_pca")
  )
  expect_s3_class(axis_block_data(blocks$subject.phenotype), "row_index_source")
  expect_identical(source_shape(axis_block_data(blocks$subject.phenotype)), c(5L, 2L))
  expect_identical(block_components(blocks$stimulus.visual_pca)$.component_id, c("PC01", "PC02"))
  expect_identical(blocks$stimulus.visual_pca$role, "embedding")
  expect_identical(
    blocks$stimulus.visual_pca$metadata$.fmridataset_lift$entity,
    "stimulus"
  )
  expect_equal(source_counts(fixture$subject_scores)$bytes, 0)
  expect_equal(source_counts(fixture$stimulus_pca)$bytes, 0)

  lifted_subject <- source_read(axis_block_data(blocks$subject.phenotype))
  lifted_stimulus <- source_read(axis_block_data(blocks$stimulus.visual_pca))
  expect_equal(
    lifted_subject,
    matrix(c(.4, .6, .4, .6, .2, .8, .4, .6, .2, .8), nrow = 5L, byrow = TRUE)
  )
  expect_equal(
    lifted_stimulus,
    matrix(c(.1, .2, .3, .4, .1, .2, .5, .6, .3, .4), nrow = 5L, byrow = TRUE)
  )
  expect_equal(source_counts(fixture$subject_scores)$reads, 1)
  expect_equal(source_counts(fixture$stimulus_pca)$reads, 1)
  expect_false(contains_runtime_state(axis_block_data(blocks$stimulus.visual_pca)))
})

test_that("lifted blocks satisfy the reusable array-source contract", {
  source <- axis_block_data(
    obs_blocks(.make_resolution_frame()$frame, resolve = TRUE)$stimulus.visual_pca
  )
  reference <- matrix(
    c(.1, .2, .3, .4, .1, .2, .5, .6, .3, .4),
    nrow = 5L,
    byrow = TRUE
  )

  expect_array_source_conformance(source, reference)
})

test_that("sparse entity blocks remain sparse beneath the lazy gather", {
  fixture <- .make_resolution_frame()
  frame <- fixture$frame
  subject <- entity(frame, "subject")
  sparse_data <- Matrix::Matrix(
    matrix(c(1, 0, 0, 2, 3, 0), nrow = 2L, byrow = TRUE),
    sparse = TRUE
  )
  replacement <- entity_frame(
    entity_data(subject),
    key = entity_key(subject),
    blocks = c(
      entity_blocks(subject),
      list(
        sparse_scores = axis_block(
          sparse_data,
          components = tibble::tibble(.component_id = c("s1", "s2", "s3"))
        )
      )
    )
  )
  entity_values <- entities(frame)
  entity_values$subject <- replacement
  rebuilt <- fmri_frame(
    assays = lapply(assays(frame), `[[`, "source"),
    observations = observation_axis(frame),
    features = feature_axis(frame),
    entities = entity_values,
    relations = relations(frame)
  )
  source <- axis_block_data(
    obs_blocks(rebuilt, resolve = TRUE)$subject.sparse_scores
  )
  expected <- as.matrix(sparse_data[c(2L, 2L, 1L, 2L, 1L), , drop = FALSE])

  expect_s3_class(source$source, "sparse_entity_source")
  expect_s4_class(source$source$data, "sparseMatrix")
  expect_array_source_conformance(source, expected)
})

test_that("lifted blocks preserve lazy selection and duplicate observation order", {
  fixture <- .make_resolution_frame()
  view <- fixture$frame[c(5L, 1L, 3L), ]
  block <- obs_blocks(view, resolve = TRUE)$stimulus.visual_pca
  source <- axis_block_data(block)

  expect_identical(source_shape(source), c(3L, 2L))
  expect_equal(
    source_read(source, observations = c(3L, 1L), features = 2L),
    matrix(c(.2, .4), ncol = 1L)
  )
  expect_equal(source_counts(fixture$stimulus_pca)$values, 2)
})

test_that("missing entity keys yield NA metadata and lazy block rows", {
  fixture <- .make_resolution_frame(missing_stimulus = TRUE)
  resolved <- observations(fixture$frame, resolve = TRUE)
  source <- axis_block_data(
    obs_blocks(fixture$frame, resolve = TRUE)$stimulus.visual_pca
  )
  lifted <- source_read(source)

  expect_true(is.na(resolved$stimulus.category[[2L]]))
  expect_true(all(is.na(lifted[2L, ])))
  expect_equal(source_counts(fixture$stimulus_pca)$values, 6)
})

test_that("multiple entity paths coalesce agreement and reject conflicts", {
  agreed <- .make_resolution_frame(direct_subject = TRUE)$frame
  expect_identical(
    observations(agreed, resolve = TRUE)$subject.age,
    c(63, 63, 71, 63, 71)
  )

  conflicted <- .make_resolution_frame(
    direct_subject = TRUE,
    conflicting_subject = TRUE
  )$frame
  expect_error(
    observations(conflicted, resolve = TRUE),
    "conflicting paths",
    class = "fmridataset_error_resolution"
  )
  expect_error(
    obs_blocks(conflicted, resolve = TRUE),
    class = "fmridataset_error_resolution"
  )
})

test_that("resolved names never overwrite canonical observation annotations", {
  scalar_collision <- .make_resolution_frame(scalar_collision = TRUE)$frame
  block_collision <- .make_resolution_frame(block_collision = TRUE)$frame

  expect_error(
    observations(scalar_collision, resolve = TRUE),
    "collides",
    class = "fmridataset_error_resolution"
  )
  expect_error(
    obs_blocks(block_collision, resolve = TRUE),
    "collides",
    class = "fmridataset_error_resolution"
  )
  expect_error(
    observations(scalar_collision, resolve = NA),
    class = "fmridataset_error_resolution"
  )
})

test_that("lifted entity sources serialize and execute through delarr", {
  source <- axis_block_data(
    obs_blocks(.make_resolution_frame()$frame, resolve = TRUE)$run.qc
  )
  restored <- unserialize(serialize(source, NULL))

  expect_identical(source_fingerprint(restored), source_fingerprint(source))
  expect_equal(delarr::collect(as_delarr(restored)), source_read(source))
  expect_invisible(source_close(source_open(restored)))
})
