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
      source = "bold",
      target = "betas",
      type = "derivation",
      map = tibble::tibble(
        .source_id = c("bold-1", "bold-3", "bold-4"),
        .target_id = paste0("beta-", 1:3)
      )
    ),
    behavior_stimuli = frame_link(
      source = "behavior",
      target = "betas",
      type = "correspondence"
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
    tables = list(
      events = events_value,
      subject_qc = auxiliary_table(
        tibble::tibble(
          subject_id = c("sub-01", "sub-02"),
          mean_fd = c(.12, .18)
        ),
        key = "subject_id", role = "quality_control"
      )
    )
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
  expect_identical(study_link(study, "betas_from_bold")$type, "derivation")
  expect_identical(study_link(study, "betas_from_bold")$source, "bold")
  expect_identical(study_link(study, "betas_from_bold")$target, "betas")
})

test_that("event tables remain keyed and relational rather than volume aligned", {
  study <- .make_study_fixture()$study
  events_value <- study_table(study, "events")

  expect_s3_class(events_value, "fmri_event_table")
  expect_identical(event_key(events_value), "event_id")
  expect_identical(nrow(event_data(events_value)), 3L)
  expect_identical(nrow(study_frame(study, "bold")), 4L)
  expect_identical(study_table(study, "events"), events_value)
})

test_that("entity filtering propagates lazily through frames and collections", {
  fixture <- .make_study_fixture()
  older <- filter_entities(fixture$study, subject, age >= 65)

  expect_s3_class(older, "fmri_study")
  expect_false(inherits(older, "fmri_study_view"))
  expect_false(any(c("base", "entity_selections") %in% names(unclass(older))))
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
  expect_s3_class(study_frame(older, "bold"), "fmri_frame")
  expect_false(any(c("base", "observation_selection") %in%
    names(unclass(study_frame(older, "bold")))))
  expect_identical(
    source_shape(assays(study_frame(older, "bold"))$signal$source),
    c(2L, 3L)
  )
  expect_identical(
    table_data(study_table(older, "subject_qc"))$subject_id,
    "sub-02"
  )
  expect_identical(
    study_link(older, "betas_from_bold")$map$.target_id,
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
  expect_identical(nrow(event_data(study_table(selected, "events"))), 1L)
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
      links = list(bad = frame_link("missing", "bold", "derivation"))
    ),
    "endpoint",
    class = "fmridataset_error_study"
  )
  expect_error(
    fmri_study(
      frames,
      entities = entities_value,
      links = list(bad = frame_link(
        "bold", "betas", "derivation",
        map = tibble::tibble(.source_id = "unknown", .target_id = "beta-1")
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
  bad_qc <- auxiliary_table(
    tibble::tibble(subject_id = "unknown", value = 1),
    key = "subject_id", role = "quality_control"
  )
  expect_error(
    fmri_study(frames, entities = entities_value, tables = list(qc = bad_qc)),
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

test_that("studies and filtered studies serialize without runtime state", {
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
  expect_output(print(view), "4 representations")
  expect_error(study_frame(study, character()), class = "fmridataset_error_study")
  expect_error(study_frames(study, contextual = NA), class = "fmridataset_error_study")
  malformed <- view
  malformed$entities$subject$data$subject_id <- "unknown"
  expect_error(validate_fmri_study(malformed), class = "fmridataset_error")
  for (source in fixture$sources) expect_equal(source_counts(source)$bytes, 0)
})

test_that("provisional frame links migrate explicitly to source-target links", {
  source_space <- index_space(1L, ids = "voxel-1", namespace = "native")
  target_space <- index_space(1L, ids = "parcel-1", namespace = "parcel")
  operator <- feature_map(source_space, target_space, matrix(1, 1L, 1L))
  legacy <- structure(list(
    from = "parcel", to = "native", type = "mapped_from",
    map = tibble::tibble(.from_id = "parcel-1", .to_id = "voxel-1"),
    from_axis = "feature", to_axis = "feature",
    metadata = list(feature_map = operator, note = "legacy"),
    schema_version = 1L
  ), class = "frame_link")

  migrated <- fmridataset:::.upgrade_frame_link(legacy)
  expect_identical(migrated$source, "native")
  expect_identical(migrated$target, "parcel")
  expect_identical(migrated$type, "mapping")
  expect_identical(migrated$map$.source_id, "voxel-1")
  expect_identical(migrated$map$.target_id, "parcel-1")
  expect_s3_class(migrated$operator, "feature_map")
  expect_null(migrated$metadata$feature_map)
  expect_true(migrated$metadata$note == "legacy")
  expect_identical(fmridataset:::.upgrade_frame_link(migrated), migrated)
})

test_that("frame links compose in source-to-target order", {
  first <- frame_link(
    "raw", "beta", "derivation",
    map = tibble::tibble(
      .source_id = c("raw-1", "raw-2"),
      .target_id = c("beta-1", "beta-2")
    )
  )
  second <- frame_link(
    "beta", "contrast", "derivation",
    map = tibble::tibble(
      .source_id = c("beta-2", "beta-1"),
      .target_id = c("contrast-2", "contrast-1")
    )
  )
  composed <- compose_frame_links(first, second)

  expect_identical(composed$source, "raw")
  expect_identical(composed$target, "contrast")
  expect_identical(composed$map$.source_id, c("raw-1", "raw-2"))
  expect_identical(composed$map$.target_id, c("contrast-1", "contrast-2"))
  expect_error(
    compose_frame_links(second, first),
    "intermediate endpoint",
    class = "fmridataset_error_study"
  )
})

.chain_study_parts <- function(extra_relations = list()) {
  subject <- entity_frame(
    data.frame(subject_id = c("s1", "s2", "s3"), age = c(20, 60, 70)),
    key = "subject_id"
  )
  session <- entity_frame(
    data.frame(
      session_id = c("s1-a", "s1-b", "s2-a", "s3-a"),
      subject_id = c("s1", "s1", "s2", "s3")
    ),
    key = "session_id"
  )
  observations_data <- data.frame(
    .obs_id = paste0("o", 1:6),
    session_id = c("s1-a", "s1-a", "s1-b", "s2-a", "s2-a", "s3-a")
  )
  space <- index_space(4, ids = paste0("f", 1:4), namespace = "chain-study")
  frame <- fmri_frame(
    assays = list(bold = matrix(as.double(1:24), nrow = 6)),
    observations = observations_data,
    space = space,
    entities = list(subject = subject, session = session),
    relations = c(list(
      obs_session = key_relation("session_id", target = "session"),
      session_subject = key_relation(
        "subject_id",
        target = "subject", source = "session"
      )
    ), extra_relations)
  )
  list(subject = subject, session = session, frame = frame, space = space)
}

test_that("entity filters cascade through parent key relations", {
  parts <- .chain_study_parts()
  study <- fmri_study(
    list(main = parts$frame),
    entities = list(subject = parts$subject, session = parts$session),
    tables = list(visits = auxiliary_table(
      data.frame(
        visit = c("v1", "v2", "v3"),
        session_id = c("s1-a", "s2-a", NA)
      ),
      key = "visit", role = "visits"
    ))
  )

  older <- filter_entities(study, subject, age > 30)
  frame <- study_frame(older, "main")
  expect_identical(observation_ids(frame), c("o4", "o5", "o6"))
  expect_identical(entity_ids(entity(older, "subject")), c("s2", "s3"))
  # Sessions of dropped subjects are dropped transitively, in registry order.
  expect_identical(entity_ids(entity(older, "session")), c("s2-a", "s3-a"))
  # Cascaded selections also restrict tables; rows without a key survive.
  expect_identical(
    table_data(study_table(older, "visits"))$visit,
    c("v2", "v3")
  )
  expect_identical(
    observations(frame, resolve = TRUE)$subject.age,
    c(60, 60, 70)
  )

  # Filtering a leaf leaves parents untouched.
  leaf <- filter_entities(study, session, session_id == "s2-a")
  expect_identical(observation_ids(study_frame(leaf, "main")), c("o4", "o5"))
  expect_identical(entity_ids(entity(leaf, "subject")), c("s1", "s2", "s3"))

  # An empty selection is a valid, empty study rather than an error.
  none <- filter_entities(study, subject, age > 100)
  expect_identical(observation_ids(study_frame(none, "main")), character())
  expect_identical(entity_ids(entity(none, "session")), character())
  expect_identical(nrow(table_data(study_table(none, "visits"))), 1L)
  expect_invisible(validate_fmri_study(none))
})

test_that("entity filters restrict sparse entity edges and validity relations", {
  subject <- entity_frame(
    data.frame(subject_id = c("s1", "s2", "s3"), age = c(20, 60, 70)),
    key = "subject_id"
  )
  space <- index_space(4, ids = paste0("f", 1:4), namespace = "sparse-study")
  validity <- entity_feature_validity(
    entity = "subject", entity_ids = c("s1", "s2", "s3"),
    masks = rbind(
      c(TRUE, TRUE, FALSE, TRUE),
      c(TRUE, FALSE, TRUE, TRUE),
      c(FALSE, TRUE, TRUE, TRUE)
    ),
    space = space
  )
  frame <- fmri_frame(
    assays = list(bold = matrix(as.double(1:12), nrow = 3)),
    observations = data.frame(.obs_id = c("o1", "o2", "o3"), subject_id = c("s1", "s2", "s3")),
    space = space,
    entities = list(subject = subject),
    relations = list(
      os = key_relation("subject_id", target = "subject"),
      similar = sparse_relation(
        data.frame(.from_id = c("s1", "s2"), .to_id = c("s2", "s3")),
        from = "subject", to = "subject"
      ),
      valid = validity
    )
  )
  study <- fmri_study(list(m = frame), entities = list(subject = subject))

  older <- filter_entities(study, subject, age > 30)
  filtered <- study_frame(older, "m")
  expect_identical(observation_ids(filtered), c("o2", "o3"))
  edges <- relation(filtered, "similar")$data
  expect_identical(edges$.from_id, "s2")
  expect_identical(edges$.to_id, "s3")
  expect_identical(validity_entity_ids(filtered, "valid"), c("s2", "s3"))
  expect_identical(
    unname(validity_matrix(filtered, "valid")),
    rbind(c(TRUE, FALSE, TRUE, TRUE), c(FALSE, TRUE, TRUE, TRUE))
  )
})

test_that("entity filters keep validated study and collection metadata", {
  subject <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), age = c(20, 60)),
    key = "subject_id"
  )
  frame <- fmri_frame(
    assays = list(bold = matrix(as.double(1:24), nrow = 6)),
    observations = data.frame(
      .obs_id = paste0("o", 1:6),
      subject_id = rep(c("s1", "s2"), c(3, 3))
    ),
    space = index_space(4, ids = paste0("f", 1:4), namespace = "meta-study"),
    entities = list(subject = subject),
    relations = list(os = key_relation("subject_id", target = "subject"))
  )
  # Length 3 matches no axis of the unfiltered study, so it is accepted.
  study <- fmri_study(
    list(main = frame),
    entities = list(subject = subject),
    metadata = list(w = 1:3)
  )
  older <- filter_entities(study, subject, age > 30)
  expect_identical(observation_ids(study_frame(older, "main")), c("o4", "o5", "o6"))
  expect_identical(unclass(older$metadata)$w, 1:3)
  expect_invisible(validate_fmri_study(older))

  collection <- fmri_collection(list(a = frame, b = frame), metadata = list(w = 1:3))
  collected <- fmri_study(list(c = collection), entities = list(subject = subject))
  filtered <- filter_entities(collected, subject, age > 30)
  expect_identical(
    unclass(study_frame(filtered, "c")$metadata)$w,
    1:3
  )
})

test_that("study table filters keep rows whose entity key is missing", {
  subject <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), age = c(20, 60)),
    key = "subject_id"
  )
  frame <- fmri_frame(
    assays = list(bold = matrix(as.double(1:8), nrow = 2)),
    observations = data.frame(.obs_id = c("o1", "o2"), subject_id = c("s1", "s2")),
    space = index_space(4, ids = paste0("f", 1:4), namespace = "na-table"),
    entities = list(subject = subject),
    relations = list(os = key_relation("subject_id", target = "subject"))
  )
  table <- auxiliary_table(
    data.frame(k = c("a", "b", "c"), subject_id = c("s2", NA, "s1")),
    key = "k", role = "notes"
  )
  study <- fmri_study(list(m = frame), entities = list(subject = subject), tables = list(t = table))
  everyone <- filter_entities(study, subject, age > 0)
  expect_identical(table_data(study_table(everyone, "t"))$k, c("a", "b", "c"))
  older <- filter_entities(study, subject, age > 30)
  expect_identical(table_data(study_table(older, "t"))$k, c("a", "b"))
})

test_that("shared entity agreement is exact after type coercion", {
  shared <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), age = c(20, 60), site = c("x", "y")),
    key = "subject_id"
  )
  local_frame <- function(local) {
    fmri_frame(
      assays = list(bold = matrix(as.double(1:8), nrow = 2)),
      observations = data.frame(.obs_id = c("o1", "o2"), subject_id = c("s1", "s2")),
      space = index_space(4, ids = paste0("f", 1:4), namespace = "exact-study"),
      entities = list(subject = local),
      relations = list(os = key_relation("subject_id", target = "subject"))
    )
  }
  near <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), age = c(20 + 1e-9, 60)),
    key = "subject_id"
  )
  expect_error(
    fmri_study(list(m = local_frame(near)), entities = list(subject = shared)),
    "disagrees",
    class = "fmridataset_error_study"
  )
  # Integer versus double and factor versus character are harmless coercions.
  coerced <- entity_frame(
    data.frame(
      subject_id = c("s1", "s2"), age = c(20L, 60L),
      site = factor(c("x", "y"))
    ),
    key = "subject_id"
  )
  expect_s3_class(
    fmri_study(list(m = local_frame(coerced)), entities = list(subject = shared)),
    "fmri_study"
  )
  # Character never matches numeric.
  text <- entity_frame(
    data.frame(subject_id = c("s1", "s2"), age = c("20", "60")),
    key = "subject_id"
  )
  expect_error(
    fmri_study(list(m = local_frame(text)), entities = list(subject = shared)),
    class = "fmridataset_error_study"
  )
})

test_that("study digests include representation order", {
  parts <- .chain_study_parts()
  other <- parts$frame
  forward <- fmri_study(
    list(a = parts$frame, b = other),
    entities = list(subject = parts$subject, session = parts$session)
  )
  reverse <- fmri_study(
    list(b = other, a = parts$frame),
    entities = list(subject = parts$subject, session = parts$session)
  )
  expect_false(identical(study_digest(forward), study_digest(reverse)))
  expect_identical(
    study_digest(forward),
    study_digest(fmri_study(
      list(a = parts$frame, b = other),
      entities = list(subject = parts$subject, session = parts$session)
    ))
  )
})

test_that("frame links require exactly one type", {
  expect_identical(frame_link("a", "b")$type, "derivation")
  expect_error(
    frame_link("a", "b",
      type = c("alignment", "derivation"),
      source_axis = "feature", target_axis = "feature"
    ),
    class = "fmridataset_error_study"
  )
})

test_that("link composition rejects one-sided and annotated maps", {
  mapped <- frame_link("a", "b",
    type = "correspondence",
    map = data.frame(.source_id = "o1", .target_id = "p1")
  )
  unmapped <- frame_link("b", "c", type = "correspondence")
  expect_error(
    compose_frame_links(mapped, unmapped),
    "unmapped",
    class = "fmridataset_error_study"
  )
  expect_error(
    compose_frame_links(frame_link("z", "a", type = "correspondence"), mapped),
    "unmapped",
    class = "fmridataset_error_study"
  )
  weighted_first <- frame_link("a", "b",
    type = "correspondence",
    map = data.frame(.source_id = "o1", .target_id = "p1", w = 0.5)
  )
  weighted_second <- frame_link("b", "c",
    type = "correspondence",
    map = data.frame(.source_id = "p1", .target_id = "q1", w = 2)
  )
  expect_error(
    compose_frame_links(weighted_first, weighted_second),
    "annotation columns",
    class = "fmridataset_error_study"
  )
  plain <- frame_link("b", "c",
    type = "correspondence",
    map = data.frame(.source_id = "p1", .target_id = "q1")
  )
  expect_identical(compose_frame_links(mapped, plain)$map$.target_id, "q1")
})

test_that("study table accessors validate their input", {
  expect_error(study_tables(list(tables = 1)), class = "fmridataset_error_study")
  expect_error(study_table(list(tables = 1), "events"), class = "fmridataset_error_study")
  expect_false(exists("events", envir = asNamespace("fmridataset"), inherits = FALSE))
})
