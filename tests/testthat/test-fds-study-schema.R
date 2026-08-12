.make_fds_study_frame <- function(prefix, subject_ids, space = NULL) {
  observations_value <- tibble::tibble(
    .obs_id = paste0(prefix, "-", seq_along(subject_ids)),
    subject_id = subject_ids
  )
  if (is.null(space)) {
    space <- index_space(3L, ids = paste0(prefix, "-feature-", 1:3), namespace = prefix)
  }
  fmri_frame(
    assays = list(signal = matrix(
      seq_len(length(subject_ids) * n_features(space)),
      nrow = length(subject_ids)
    )),
    observations = observations_value,
    space = space,
    entities = list(subject = entity_frame(
      tibble::tibble(subject_id = unique(subject_ids)), key = "subject_id"
    )),
    relations = list(observation_subject = key_relation("subject_id", target = "subject"))
  )
}

.make_fds_study_fixture <- function() {
  phenotype <- counting_source(memory_source(matrix(c(.2, .8, .4, .6), nrow = 2L)))
  entities_value <- list(subject = entity_frame(
    tibble::tibble(subject_id = c("sub-01", "sub-02"), age = c(63, 71)),
    key = "subject_id",
    blocks = list(phenotype = axis_block(
      phenotype,
      components = tibble::tibble(.component_id = c("memory", "attention"))
    ))
  ))
  bold <- .make_fds_study_frame("bold", c("sub-01", "sub-02"))
  beta <- .make_fds_study_frame("beta", c("sub-01", "sub-02"))
  native <- fmri_collection(list(
    sub_01 = .make_fds_study_frame(
      "native-01", "sub-01",
      volume_space(c(2L, 2L, 2L), support = 1:3, template = "sub-01")
    ),
    sub_02 = .make_fds_study_frame(
      "native-02", "sub-02",
      volume_space(c(3L, 2L, 2L), support = c(1L, 2L, 4L, 5L), template = "sub-02")
    )
  ))
  study <- fmri_study(
    frames = list(bold = bold, beta = beta, native = native),
    entities = entities_value,
    links = list(beta_from_bold = frame_link(
      "beta", "bold", "derived_from",
      map = tibble::tibble(
        .from_id = observation_ids(beta),
        .to_id = observation_ids(bold)
      )
    )),
    tables = list(events = event_table(tibble::tibble(
      event_id = c("event-1", "event-2"),
      subject_id = c("sub-01", "sub-02"),
      onset = c(0, 2), duration = c(1, 1)
    ))),
    metadata = list(title = "FDS study fixture"),
    provenance = as_provenance_graph(list(step = "test"))
  )
  list(study = study, phenotype = phenotype)
}

test_that("FDS study manifests separate semantic state from numerical bindings", {
  fixture <- .make_fds_study_fixture()
  manifest <- fds_study_manifest(fixture$study)
  bindings <- fds_study_bindings(fixture$study)

  expect_identical(manifest$schema$id, "org.fmridataset.fds-study/v1")
  expect_identical(manifest$schema$version, 1L)
  expect_identical(manifest$object_type, "fmri_study")
  expect_identical(names(manifest$representations), c("bold", "beta", "native"))
  expect_identical(manifest$representations$bold$type, "fmri_frame")
  expect_identical(manifest$representations$native$type, "fmri_collection")
  expect_identical(
    names(manifest$representations$native$members),
    c("sub_01", "sub_02")
  )
  expect_identical(
    names(manifest$arrays),
    "entities/subject/blocks/phenotype"
  )
  expect_identical(names(bindings), names(manifest$arrays))
  expect_s3_class(bindings[[1L]], "counting_source")
  expect_false(contains_runtime_state(manifest))
  expect_invisible(validate_fds_study_manifest(manifest))
  expect_equal(source_counts(fixture$phenotype)$bytes, 0)
})

test_that("FDS study manifests rebuild linked frames collections and shared blocks", {
  fixture <- .make_fds_study_fixture()
  manifest <- fds_study_manifest(fixture$study)
  representations <- study_frames(fixture$study, contextual = FALSE)
  rebuilt <- study_from_fds_manifest(
    manifest,
    representations = representations,
    bindings = fds_study_bindings(fixture$study)
  )

  expect_s3_class(rebuilt, "fmri_study")
  expect_identical(study_ids(rebuilt), study_ids(fixture$study))
  expect_s3_class(study_frame(rebuilt, "native"), "fmri_collection")
  expect_identical(study_links(rebuilt), study_links(fixture$study))
  expect_identical(study_tables(rebuilt), study_tables(fixture$study))
  expect_identical(rebuilt$metadata, fixture$study$metadata)
  expect_identical(rebuilt$provenance, fixture$study$provenance)
  expect_equal(
    source_read(as_array_source(entity_blocks(entity(rebuilt, "subject"))$phenotype$data)),
    source_read(as_array_source(entity_blocks(entity(fixture$study, "subject"))$phenotype$data)),
    tolerance = 0
  )
})

test_that("FDS study manifests reject legacy lineage and hidden alignment", {
  fixture <- .make_fds_study_fixture()
  manifest <- fds_study_manifest(fixture$study)

  malformed <- manifest
  malformed$provenance <- list(step = "legacy")
  expect_error(
    validate_fds_study_manifest(malformed),
    "provenance_graph",
    class = "fmridataset_error_schema"
  )

  malformed <- manifest
  malformed$metadata <- unaligned_record(list(
    per_subject = seq_len(length(entity(fixture$study, "subject")))
  ))
  expect_error(
    validate_fds_study_manifest(malformed),
    "entity:subject-aligned",
    class = "fmridataset_error_schema"
  )

  malformed <- manifest
  malformed$tables$raw <- data.frame(value = 1L)
  expect_error(
    validate_fds_study_manifest(malformed),
    "typed table",
    class = "fmridataset_error_schema"
  )
})

test_that("FDS study validation rejects drift in representations links and bindings", {
  fixture <- .make_fds_study_fixture()
  manifest <- fds_study_manifest(fixture$study)
  representations <- study_frames(fixture$study, contextual = FALSE)
  bindings <- fds_study_bindings(fixture$study)

  future <- manifest
  future$schema$version <- 2L
  expect_error(validate_fds_study_manifest(future), "version", class = "fmridataset_error_schema")

  missing_endpoint <- manifest
  missing_endpoint$links$beta_from_bold$to <- "missing"
  expect_error(validate_fds_study_manifest(missing_endpoint), "endpoint")

  wrong_member <- representations
  wrong_member$native$frames$sub_01 <- wrong_member$native$frames$sub_01[, 1:2]
  expect_error(
    study_from_fds_manifest(manifest, wrong_member, bindings),
    "manifest"
  )

  expect_error(
    study_from_fds_manifest(manifest, representations, list()),
    "exactly match"
  )
})

test_that("filtered studies persist their visible semantic view", {
  study <- filter_entities(.make_fds_study_fixture()$study, subject, age >= 65)
  manifest <- fds_study_manifest(study)
  rebuilt <- study_from_fds_manifest(
    manifest,
    fds_study_representations(study),
    fds_study_bindings(study)
  )

  expect_identical(entity_ids(entity(rebuilt, "subject")), "sub-02")
  expect_identical(nrow(study_frame(rebuilt, "bold")), 1L)
  expect_identical(nrow(event_data(events(rebuilt))), 1L)
  expect_identical(nrow(study_link(rebuilt, "beta_from_bold")$map), 1L)
})

test_that("the installed FDS study schema envelope is machine readable", {
  skip_if_not_installed("jsonlite")
  path <- system.file("schema", "fds-study-v1.schema.json", package = "fmridataset")
  expect_true(file.exists(path))
  schema <- jsonlite::read_json(path, simplifyVector = TRUE)

  expect_identical(schema[["$id"]], "org.fmridataset.fds-study/v1")
  expect_true(all(c(
    "schema", "object_type", "representations", "arrays", "entities",
    "links", "tables", "metadata", "provenance", "extensions"
  ) %in% schema$required))
  expect_identical(schema$properties$schema$properties$version$const, 1L)
})
