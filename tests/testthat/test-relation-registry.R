.frame_with_relations <- function(frame, relations) {
  fmri_frame(
    assays = lapply(assays(frame), `[[`, "source"),
    observations = observation_axis(frame),
    features = feature_axis(frame),
    entities = entities(frame),
    relations = relations,
    tables = frame$tables,
    active_assay = active_assay(frame),
    metadata = frame$metadata,
    provenance = frame$provenance
  )
}

test_that("relation descriptors and registries are serializable", {
  keyed <- key_relation(
    "stimulus_id",
    target = "stimulus",
    source = "observation",
    allow_missing = FALSE,
    metadata = list(role = "presentation")
  )
  edges <- sparse_relation(
    data = tibble::tibble(
      .from_id = c("obs-1", "obs-2"),
      .to_id = c("stim-1", "stim-2"),
      weight = c(.7, .3)
    ),
    from = "observation",
    to = "entity:stimulus",
    weight = "weight"
  )
  registry <- relation_registry(
    observation_stimulus = keyed,
    weighted_stimulus = edges
  )

  expect_s3_class(keyed, "key_relation")
  expect_s3_class(edges, "sparse_relation")
  expect_s3_class(registry, "relation_registry")
  expect_identical(relation_names(registry), c("observation_stimulus", "weighted_stimulus"))
  expect_identical(relation(registry, "observation_stimulus"), keyed)
  expect_identical(relations(registry), registry)
  expect_match(relation_registry_digest(registry), "^[0-9a-f]{64}$")
  expect_false(contains_runtime_state(registry))
  expect_identical(
    relation_registry_digest(unserialize(serialize(registry, NULL))),
    relation_registry_digest(registry)
  )
  expect_error(relation(registry, "missing"), class = "fmridataset_error_relation")
})

test_that("key relations infer unique targets and enforce referential integrity", {
  fixture <- make_frame_fixture()
  frame <- .frame_with_relations(
    fixture$frame,
    list(observation_stimulus = key_relation("stimulus_id"))
  )
  descriptor <- relation(frame, "observation_stimulus")

  expect_identical(descriptor$source, "observation")
  expect_identical(descriptor$target, "entity:stimulus")
  expect_identical(descriptor$key, "stimulus_id")
  expect_invisible(validate_relation_registry(
    relations(frame), observation_axis(frame), feature_axis(frame), entities(frame)
  ))

  bad_observations <- observation_axis(fixture$frame)
  bad_observations$data$stimulus_id[[1L]] <- "unknown"
  expect_error(
    fmri_frame(
      assays = lapply(assays(fixture$frame), `[[`, "source"),
      observations = bad_observations,
      features = feature_axis(fixture$frame),
      entities = entities(fixture$frame),
      relations = list(observation_stimulus = key_relation("stimulus_id"))
    ),
    "unknown target",
    class = "fmridataset_error_relation"
  )

  missing_observations <- observation_axis(fixture$frame)
  missing_observations$data$stimulus_id[[1L]] <- NA_character_
  expect_error(
    fmri_frame(
      assays = lapply(assays(fixture$frame), `[[`, "source"),
      observations = missing_observations,
      features = feature_axis(fixture$frame),
      entities = entities(fixture$frame),
      relations = list(observation_stimulus = key_relation("stimulus_id"))
    ),
    "missing",
    class = "fmridataset_error_relation"
  )
  expect_s3_class(
    fmri_frame(
      assays = lapply(assays(fixture$frame), `[[`, "source"),
      observations = missing_observations,
      features = feature_axis(fixture$frame),
      entities = entities(fixture$frame),
      relations = list(
        observation_stimulus = key_relation("stimulus_id", allow_missing = TRUE)
      )
    ),
    "fmri_frame"
  )
})

test_that("key relations distinguish nested entity domains", {
  fixture <- make_frame_fixture()
  subject <- entity_frame(
    tibble::tibble(subject_id = c("sub-01", "sub-02", "sub-03"), age = c(60, 70, 65)),
    key = "subject_id"
  )
  session <- entity_frame(
    tibble::tibble(
      session_id = c("ses-01", "ses-02", "ses-03"),
      subject_id = c("sub-01", "sub-01", "sub-03")
    ),
    key = "session_id"
  )
  registry <- entity_registry(
    subject = subject,
    session = session,
    stimulus = entity(fixture$frame, "stimulus")
  )
  frame <- fmri_frame(
    assays = lapply(assays(fixture$frame), `[[`, "source"),
    observations = observation_axis(fixture$frame),
    features = feature_axis(fixture$frame),
    entities = registry,
    relations = list(
      session_subject = key_relation(
        "subject_id",
        source = "session", target = "subject"
      )
    )
  )

  expect_identical(relation(frame, "session_subject")$source, "entity:session")
  expect_identical(relation(frame, "session_subject")$target, "entity:subject")

  ambiguous <- entity_registry(
    subject = subject,
    participant = entity_frame(
      tibble::tibble(subject_id = c("sub-01", "sub-02", "sub-03")),
      key = "subject_id"
    )
  )
  expect_error(
    fmri_frame(
      assays = lapply(assays(fixture$frame), `[[`, "source"),
      observations = observation_axis(fixture$frame),
      features = feature_axis(fixture$frame),
      entities = ambiguous,
      relations = list(subject = key_relation("subject_id"))
    ),
    "uniquely infer",
    class = "fmridataset_error_relation"
  )
})

test_that("sparse relations validate edge identities and weights", {
  fixture <- make_frame_fixture()
  edges <- tibble::tibble(
    .from_id = observation_ids(fixture$frame)[c(1L, 2L, 4L)],
    .to_id = c("stim-1", "stim-2", "stim-3"),
    score = c(.8, .5, .9)
  )
  frame <- .frame_with_relations(
    fixture$frame,
    list(
      observation_stimulus = sparse_relation(
        edges,
        from = "observation",
        to = "stimulus",
        weight = "score"
      )
    )
  )
  descriptor <- relation(frame, "observation_stimulus")
  expect_identical(descriptor$from, "observation")
  expect_identical(descriptor$to, "entity:stimulus")
  expect_identical(descriptor$weight, "score")

  unknown <- edges
  unknown$.to_id[[1L]] <- "unknown"
  expect_error(
    .frame_with_relations(
      fixture$frame,
      list(bad = sparse_relation(unknown, "observation", "stimulus"))
    ),
    "unknown",
    class = "fmridataset_error_relation"
  )
  expect_error(
    sparse_relation(
      rbind(edges[1L, ], edges[1L, ]),
      "observation", "stimulus",
      weight = "score"
    ),
    "duplicate",
    class = "fmridataset_error_relation"
  )
})

test_that("frame views restrict sparse observation and feature relations", {
  fixture <- make_frame_fixture()
  edges <- tibble::tibble(
    .from_id = observation_ids(fixture$frame)[1:4],
    .to_id = feature_ids(fixture$frame)[c(1L, 2L, 1L, 3L)]
  )
  frame <- .frame_with_relations(
    fixture$frame,
    list(observation_feature = sparse_relation(edges, "observation", "feature"))
  )
  view <- frame[c(4L, 1L), c(3L, 1L)]
  selected <- relation(view, "observation_feature")$data

  expect_identical(selected$.from_id, observation_ids(frame)[c(1L, 4L)])
  expect_identical(selected$.to_id, feature_ids(frame)[c(1L, 3L)])
  expect_identical(relations(frame)$observation_feature$data, edges)
})

test_that("bind_observations preserves key relations and merges sparse observation edges", {
  fixture <- make_frame_fixture()
  edges <- tibble::tibble(
    .from_id = observation_ids(fixture$frame),
    .to_id = rep("stim-1", nrow(fixture$frame))
  )
  frame <- .frame_with_relations(
    fixture$frame,
    list(
      observation_stimulus = key_relation("stimulus_id"),
      sparse_observation_stimulus = sparse_relation(
        edges, "observation", "stimulus"
      )
    )
  )
  combined <- bind_observations(frame[1:3, ], frame[4:7, ])

  expect_identical(relation_names(combined), relation_names(frame))
  expect_identical(
    relation(combined, "sparse_observation_stimulus")$data,
    edges
  )
  expect_identical(
    relation(combined, "observation_stimulus"),
    relation(frame, "observation_stimulus")
  )
})

test_that("relations persist through FDS and HDF5 without runtime state", {
  fixture <- make_frame_fixture()
  edges <- tibble::tibble(
    .from_id = observation_ids(fixture$frame)[1:3],
    .to_id = c("stim-1", "stim-2", "stim-3")
  )
  frame <- .frame_with_relations(
    fixture$frame,
    list(
      key = key_relation("stimulus_id"),
      sparse = sparse_relation(edges, "observation", "stimulus")
    )
  )
  manifest <- fds_frame_manifest(frame)
  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(frame))

  expect_s3_class(manifest$relations, "relation_registry")
  expect_false(contains_runtime_state(manifest$relations))
  expect_identical(relation_registry_digest(rebuilt), relation_registry_digest(frame))

  invalid <- manifest
  invalid$relations$sparse$data$.to_id[[1L]] <- "unknown"
  expect_error(validate_fds_manifest(invalid), class = "fmridataset_error_schema")

  skip_if_not_installed("fmristore")
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
  path <- tempfile(fileext = ".fds.h5")
  on.exit(unlink(path), add = TRUE)
  write_frame(frame, path)
  reopened <- open_frame(path)
  expect_identical(relation_registry_digest(reopened), relation_registry_digest(frame))
})

test_that("relation constructors reject malformed descriptors", {
  expect_error(key_relation(character()), class = "fmridataset_error_relation")
  expect_error(
    sparse_relation(
      tibble::tibble(.from_id = "a", .to_id = "b", weight = "bad"),
      "observation", "feature",
      weight = "weight"
    ),
    "numeric",
    class = "fmridataset_error_relation"
  )
  expect_error(relation_registry(list(key_relation("id"))), "named")
  expect_error(
    relation_registry(list(bad = list(key = "id"))),
    class = "fmridataset_error_relation"
  )
  reverse_duplicate <- tibble::tibble(
    .from_id = c("a", "b"),
    .to_id = c("b", "a")
  )
  expect_error(
    sparse_relation(
      reverse_duplicate,
      "observation",
      "observation",
      directed = FALSE
    ),
    "duplicate",
    class = "fmridataset_error_relation"
  )
  registry <- relation_registry(
    edge = sparse_relation(
      tibble::tibble(.from_id = "a", .to_id = "b"),
      "observation", "feature"
    )
  )
  registry$edge$data <- rbind(registry$edge$data, registry$edge$data)
  expect_error(
    validate_relation_registry(registry),
    "duplicate",
    class = "fmridataset_error_relation"
  )
})
