.make_study_frame <- function(name, observations_data, n_feature = 3L) {
  local_entities <- list()
  relation_values <- list()
  if ("subject_id" %in% names(observations_data)) {
    ids <- unique(observations_data$subject_id[!is.na(observations_data$subject_id)])
    local_entities$subject <- entity_frame(
      tibble::tibble(subject_id = ids),
      key = "subject_id"
    )
    relation_values$observation_subject <- key_relation(
      "subject_id",
      target = "subject"
    )
  }
  if ("stimulus_id" %in% names(observations_data)) {
    ids <- unique(observations_data$stimulus_id[!is.na(observations_data$stimulus_id)])
    local_entities$stimulus <- entity_frame(
      tibble::tibble(stimulus_id = ids),
      key = "stimulus_id"
    )
    relation_values$observation_stimulus <- key_relation(
      "stimulus_id",
      target = "stimulus"
    )
  }
  if ("run_id" %in% names(observations_data)) {
    ids <- unique(observations_data$run_id[!is.na(observations_data$run_id)])
    local_entities$run <- entity_frame(
      tibble::tibble(run_id = ids),
      key = "run_id"
    )
    relation_values$observation_run <- key_relation("run_id", target = "run")
  }
  values <- matrix(
    seq_len(nrow(observations_data) * n_feature),
    nrow = nrow(observations_data),
    ncol = n_feature
  )
  source <- counting_source(memory_source(values))
  frame <- fmri_frame(
    assays = list(signal = source),
    observations = observations_data,
    space = index_space(
      n_feature,
      ids = paste0(name, "-feature-", seq_len(n_feature)),
      namespace = name
    ),
    entities = local_entities,
    relations = relation_values
  )
  list(frame = frame, source = source)
}

.make_native_study_frame <- function(subject, variant = 1L) {
  observations_data <- tibble::tibble(
    .obs_id = paste0(subject, "-native-", 1:2),
    subject_id = subject,
    condition = factor(c("A", "B"), levels = c("A", "B"))
  )
  entity_value <- entity_frame(
    tibble::tibble(subject_id = subject),
    key = "subject_id"
  )
  if (variant == 1L) {
    feature_space <- volume_space(c(2L, 2L, 2L), support = 1:3, template = subject)
  } else {
    feature_space <- volume_space(c(3L, 2L, 2L), support = c(1L, 2L, 4L, 5L), template = subject)
  }
  source <- counting_source(memory_source(matrix(
    seq_len(2L * n_features(feature_space)),
    nrow = 2L
  )))
  feature_info <- feature_data(feature_space)
  feature_info$parcel <- "native"
  frame <- fmri_frame(
    assays = list(signal = source),
    observations = observations_data,
    features = feature_axis(
      feature_info,
      space = feature_space
    ),
    entities = list(subject = entity_value),
    relations = list(
      observation_subject = key_relation("subject_id", target = "subject")
    )
  )
  list(frame = frame, source = source)
}

.make_study_fixture <- function() {
  subject_scores <- counting_source(memory_source(matrix(
    c(.2, .8, .4, .6),
    nrow = 2L, byrow = TRUE
  )))
  shared_entities <- list(
    subject = entity_frame(
      tibble::tibble(
        subject_id = c("sub-01", "sub-02"),
        age = c(63, 71),
        cohort = factor(c("younger", "older"), levels = c("younger", "older"))
      ),
      key = "subject_id",
      blocks = list(
        phenotype = axis_block(
          subject_scores,
          components = tibble::tibble(.component_id = c("memory", "attention"))
        )
      )
    ),
    stimulus = entity_frame(
      tibble::tibble(
        stimulus_id = c("stim-1", "stim-2", "stim-3"),
        category = factor(c("face", "scene", "object"))
      ),
      key = "stimulus_id"
    ),
    run = entity_frame(
      tibble::tibble(
        run_id = c("run-1", "run-2"),
        tr = c(2, 1.5)
      ),
      key = "run_id"
    )
  )
  bold <- .make_study_frame(
    "bold",
    tibble::tibble(
      .obs_id = paste0("bold-", 1:4),
      subject_id = c("sub-01", "sub-01", "sub-02", "sub-02"),
      run_id = c("run-1", "run-1", "run-2", "run-2"),
      stimulus_id = c("stim-1", "stim-2", "stim-1", "stim-3")
    )
  )
  betas <- .make_study_frame(
    "betas",
    tibble::tibble(
      .obs_id = paste0("beta-", 1:3),
      subject_id = c("sub-01", "sub-02", "sub-02"),
      stimulus_id = c("stim-1", "stim-2", "stim-3")
    )
  )
  behavior <- .make_study_frame(
    "behavior",
    tibble::tibble(
      .obs_id = paste0("behavior-", 1:3),
      subject_id = c("sub-01", "sub-02", "sub-01"),
      stimulus_id = c("stim-1", "stim-2", "stim-3")
    ),
    n_feature = 2L
  )
  native_01 <- .make_native_study_frame("sub-01", 1L)
  native_02 <- .make_native_study_frame("sub-02", 2L)
  native <- fmri_collection(list(
    sub_01 = native_01$frame,
    sub_02 = native_02$frame
  ))
  events_value <- event_table(tibble::tibble(
    event_id = c("event-1", "event-2", "event-3"),
    run_id = c("run-1", "run-1", "run-2"),
    onset = c(0, 3.5, 1),
    duration = c(1, 1.5, 2),
    stimulus_id = c("stim-1", "stim-2", "stim-3")
  ))
  links <- list(
    betas_from_bold = frame_link(
      from = "betas",
      to = "bold",
      type = "derived_from",
      map = tibble::tibble(
        .from_id = paste0("beta-", 1:3),
        .to_id = c("bold-1", "bold-3", "bold-4")
      )
    ),
    behavior_stimuli = frame_link(
      from = "behavior",
      to = "betas",
      type = "corresponds_to"
    )
  )
  study <- fmri_study(
    frames = list(
      bold = bold$frame,
      betas = betas$frame,
      behavior = behavior$frame,
      native = native
    ),
    entities = shared_entities,
    links = links,
    tables = list(events = events_value)
  )
  list(
    study = study,
    sources = c(
      list(subject_scores),
      list(bold$source, betas$source, behavior$source),
      list(native_01$source, native_02$source)
    )
  )
}

test_that("fmri_study links distinct frames through shared authoritative entities", {
  fixture <- .make_study_fixture()
  study <- fixture$study

  expect_s3_class(study, "fmri_study")
  expect_identical(study_ids(study), c("bold", "betas", "behavior", "native"))
  expect_s3_class(study_frame(study, "bold"), "fmri_frame")
  expect_s3_class(study_frame(study, "native"), "fmri_collection")
  expect_identical(entity_names(study), c("subject", "stimulus", "run"))
  expect_identical(
    observations(study_frame(study, "bold"), resolve = TRUE)$subject.age,
    c(63, 63, 71, 71)
  )
  expect_s3_class(
    obs_blocks(study_frame(study, "bold"), resolve = TRUE)$subject.phenotype,
    "axis_block"
  )
  expect_identical(names(study_links(study)), c("betas_from_bold", "behavior_stimuli"))
  expect_identical(study_link(study, "betas_from_bold")$type, "derived_from")
})

test_that("event tables remain keyed and relational rather than volume aligned", {
  study <- .make_study_fixture()$study
  events_value <- events(study)

  expect_s3_class(events_value, "fmri_event_table")
  expect_identical(event_key(events_value), "event_id")
  expect_identical(nrow(event_data(events_value)), 3L)
  expect_identical(nrow(study_frame(study, "bold")), 4L)
  expect_identical(study_table(study, "events"), events_value)
})

test_that("entity filtering propagates lazily through frames and collections", {
  fixture <- .make_study_fixture()
  older <- filter_entities(fixture$study, subject, age >= 65)

  expect_s3_class(older, "fmri_study_view")
  expect_identical(entity_ids(entity(older, "subject")), "sub-02")
  expect_identical(nrow(study_frame(older, "bold")), 2L)
  expect_identical(nrow(study_frame(older, "betas")), 2L)
  expect_identical(nrow(study_frame(older, "behavior")), 1L)
  native <- study_frame(older, "native")
  expect_identical(nrow(native[["sub_01"]]), 0L)
  expect_identical(nrow(native[["sub_02"]]), 2L)
  expect_identical(
    observations(study_frame(older, "bold"), resolve = TRUE)$subject.age,
    c(71, 71)
  )
  expect_identical(
    study_link(older, "betas_from_bold")$map$.from_id,
    c("beta-2", "beta-3")
  )
  for (source in fixture$sources) expect_equal(source_counts(source)$bytes, 0)
})

test_that("entity filters compose across crossed subject and stimulus relations", {
  study <- .make_study_fixture()$study
  selected <- study |>
    filter_entities(subject, age >= 65) |>
    filter_entities(stimulus, category == "scene")

  expect_identical(observation_ids(study_frame(selected, "bold")), character())
  expect_identical(observation_ids(study_frame(selected, "betas")), "beta-2")
  expect_identical(observation_ids(study_frame(selected, "behavior")), "behavior-2")
  expect_identical(entity_ids(entity(selected, "subject")), "sub-02")
  expect_identical(entity_ids(entity(selected, "stimulus")), "stim-2")
  expect_identical(nrow(event_data(events(selected))), 1L)
  expect_identical(nrow(study_link(selected, "betas_from_bold")$map), 0L)
})

test_that("study links validate endpoints axes and optional maps", {
  fixture <- .make_study_fixture()
  frames <- study_frames(fixture$study, contextual = FALSE)
  entities_value <- entities(fixture$study)

  expect_error(
    fmri_study(
      frames,
      entities = entities_value,
      links = list(bad = frame_link("missing", "bold", "derived_from"))
    ),
    "endpoint",
    class = "fmridataset_error_study"
  )
  expect_error(
    fmri_study(
      frames,
      entities = entities_value,
      links = list(bad = frame_link(
        "betas", "bold", "derived_from",
        map = tibble::tibble(.from_id = "unknown", .to_id = "bold-1")
      ))
    ),
    "unknown",
    class = "fmridataset_error_study"
  )
  expect_error(
    frame_link("betas", "bold", "invented"),
    class = "fmridataset_error_study"
  )
})

test_that("study validates shared entities and event references", {
  fixture <- .make_study_fixture()
  frames <- study_frames(fixture$study, contextual = FALSE)
  entities_value <- entities(fixture$study)
  bad_events <- event_table(tibble::tibble(
    event_id = "event-x", run_id = "unknown", onset = 0, duration = 1
  ))

  expect_error(
    fmri_study(frames, entities = entities_value, tables = list(events = bad_events)),
    "unknown",
    class = "fmridataset_error_study"
  )
  changed_subject <- entity_frame(
    tibble::tibble(subject_id = c("sub-01", "sub-02"), age = c(99, 71)),
    key = "subject_id"
  )
  bold <- frames$bold
  local_entities <- entities(bold)
  local_entities$subject <- entity_frame(
    tibble::tibble(subject_id = c("sub-01", "sub-02"), age = c(63, 71)),
    key = "subject_id"
  )
  frames$bold <- fmri_frame(
    assays = lapply(assays(bold), `[[`, "source"),
    observations = observation_axis(bold),
    features = feature_axis(bold),
    entities = local_entities,
    relations = relations(bold)
  )
  expect_error(
    fmri_study(
      frames,
      entities = list(
        subject = changed_subject,
        stimulus = entity(entities_value, "stimulus"),
        run = entity(entities_value, "run")
      )
    ),
    "disagrees",
    class = "fmridataset_error_study"
  )
})

test_that("event_table validates keys and temporal fields", {
  expect_error(
    event_table(tibble::tibble(
      event_id = c("a", "a"), onset = c(0, 1), duration = c(1, 1)
    )),
    "unique",
    class = "fmridataset_error_event"
  )
  expect_error(
    event_table(tibble::tibble(event_id = "a", onset = -1, duration = 1)),
    "onset",
    class = "fmridataset_error_event"
  )
  expect_error(
    event_table(tibble::tibble(event_id = "a", onset = 0, duration = -1)),
    "duration",
    class = "fmridataset_error_event"
  )
  malformed <- event_table(tibble::tibble(event_id = "a", onset = 0, duration = 1))
  malformed$schema_version <- 2L
  expect_error(validate_event_table(malformed), class = "fmridataset_error_event")
})

test_that("studies and filtered views serialize without runtime state", {
  fixture <- .make_study_fixture()
  study <- fixture$study
  view <- filter_entities(study, subject, age > 60)

  expect_match(study_digest(study), "^[0-9a-f]{64}$")
  expect_identical(
    study_digest(unserialize(serialize(study, NULL))),
    study_digest(study)
  )
  expect_identical(
    study_digest(unserialize(serialize(view, NULL))),
    study_digest(view)
  )
  expect_false(contains_runtime_state(study))
  expect_false(contains_runtime_state(view))
  expect_output(print(study), "4 representations")
  expect_output(print(view), "filtered view")
  expect_error(study_frame(study, character()), class = "fmridataset_error_study")
  expect_error(study_frames(study, contextual = NA), class = "fmridataset_error_study")
  malformed <- view
  malformed$entity_selections$subject <- "unknown"
  expect_error(validate_fmri_study(malformed), class = "fmridataset_error_study")
  for (source in fixture$sources) expect_equal(source_counts(source)$bytes, 0)
})
