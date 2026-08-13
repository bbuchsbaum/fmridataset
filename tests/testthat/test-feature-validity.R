.validity_fixture <- function(instrument = FALSE) {
  values <- matrix(seq_len(30L), nrow = 5L, byrow = TRUE)
  source <- memory_source(values, chunks = c(2L, 3L))
  if (instrument) source <- counting_source(source)
  space_value <- index_space(
    6L, ids = paste0("feature-", 1:6), namespace = "validity-fixture"
  )
  subjects <- entity_frame(
    data.frame(subject_id = c("sub-1", "sub-2", "sub-3")),
    key = "subject_id"
  )
  validity <- entity_feature_validity(
    entity = "subject",
    entity_ids = c("sub-1", "sub-2", "sub-3"),
    masks = rbind(
      c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE),
      c(TRUE, FALSE, FALSE, TRUE, TRUE, TRUE),
      c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE)
    ),
    space = space_value
  )
  frame <- fmri_frame(
    assays = list(signal = source),
    observations = data.frame(
      .obs_id = paste0("obs-", 1:5),
      subject_id = c("sub-1", "sub-1", "sub-2", "sub-3", "sub-3")
    ),
    space = space_value,
    entities = list(subject = subjects),
    relations = list(
      observation_subject = key_relation("subject_id"),
      subject_feature_validity = validity
    )
  )
  list(
    frame = frame, validity = validity, source = source, values = values,
    masks = rbind(
      c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE),
      c(TRUE, FALSE, FALSE, TRUE, TRUE, TRUE),
      c(TRUE, TRUE, FALSE, TRUE, FALSE, TRUE)
    )
  )
}

test_that("mask_bank deduplicates and bit-packs feature masks", {
  fx <- .validity_fixture()
  bank <- validity_mask_bank(fx$validity)

  expect_s3_class(bank, "mask_bank")
  expect_invisible(validate_mask_bank(bank))
  expect_identical(n_masks(bank), 2L)
  expect_identical(dim(mask_values(bank)), c(2L, 6L))
  expect_identical(mask_values(bank)[1L, ], fx$masks[1L, ])
  expect_true(is.raw(bank$bits))
  expect_lt(length(bank$bits), length(fx$masks))
  expect_match(mask_bank_digest(bank), "^[0-9a-f]{64}$")
  expect_identical(
    mask_bank_digest(unserialize(serialize(bank, NULL))),
    mask_bank_digest(bank)
  )
})

test_that("entity_feature_validity retains explicit entity and space identity", {
  fx <- .validity_fixture()
  validity <- fx$validity

  expect_s3_class(validity, "entity_feature_validity")
  expect_s3_class(validity, "fmri_relation")
  expect_invisible(validate_entity_feature_validity(validity))
  expect_identical(validity_entity(validity), "subject")
  expect_identical(validity_entity_ids(validity), c("sub-1", "sub-2", "sub-3"))
  expect_identical(validity_matrix(validity), fx$masks)
  expect_identical(
    space_digest(validity_space(validity)),
    space_digest(space(fx$frame))
  )
  expect_identical(validity$mask_id[c(1L, 3L)], rep(validity$mask_id[[1L]], 2L))
})

test_that("validity contracts reject drift and malformed assignments", {
  fx <- .validity_fixture()
  expect_error(
    mask_bank(fx$masks[, -1L], space(fx$frame)),
    "feature"
  )
  expect_error(
    mask_bank(matrix(logical(), nrow = 0L, ncol = 6L), space(fx$frame)),
    "at least one"
  )
  expect_error(
    entity_feature_validity(
      "subject", c("sub-1", "sub-1"), fx$masks[1:2, ], space(fx$frame)
    ),
    "unique"
  )
  unknown <- fx$validity
  unknown$entity_ids[[1L]] <- "unknown"
  expect_error(
    fmri_frame(
      assays = list(signal = fx$values),
      observations = observation_axis(fx$frame),
      features = feature_axis(fx$frame),
      entities = entities(fx$frame),
      relations = list(validity = unknown)
    ),
    "entity IDs do not match"
  )
})

test_that("frame and view validity access reads zero assay bytes", {
  fx <- .validity_fixture(instrument = TRUE)
  frame <- fx$frame

  expect_equal(source_counts(fx$source)$bytes, 0)
  expect_identical(
    validity_matrix(frame, "subject_feature_validity"),
    fx$masks
  )
  observed <- observation_validity(frame, "subject_feature_validity")
  expect_identical(observed, fx$masks[c(1L, 1L, 2L, 3L, 3L), ])
  expect_equal(source_counts(fx$source)$bytes, 0)

  view <- frame[c(5L, 3L, 1L), c(6L, 2L, 5L)]
  expect_identical(
    validity_matrix(view, "subject_feature_validity"),
    fx$masks[, c(6L, 2L, 5L), drop = FALSE]
  )
  expect_identical(
    observation_validity(view, "subject_feature_validity"),
    fx$masks[c(3L, 2L, 1L), c(6L, 2L, 5L), drop = FALSE]
  )
  expect_equal(source_counts(fx$source)$bytes, 0)

  empty <- frame[1L, integer()]
  expect_identical(dim(validity_matrix(empty, "subject_feature_validity")),
                   c(3L, 0L))
  expect_identical(dim(observation_validity(empty, "subject_feature_validity")),
                   c(1L, 0L))
  expect_identical(validity_coverage(empty, "subject_feature_validity"),
                   setNames(numeric(), character()))
})

test_that("coverage summaries stay policy-free and exact", {
  fx <- .validity_fixture()
  entity_coverage <- validity_coverage(
    fx$frame, "subject_feature_validity", domain = "entity"
  )
  observation_coverage <- validity_coverage(
    fx$frame, "subject_feature_validity", domain = "observation"
  )

  expect_equal(unname(entity_coverage), colMeans(fx$masks))
  expect_equal(
    unname(observation_coverage),
    colMeans(fx$masks[c(1L, 1L, 2L, 3L, 3L), , drop = FALSE])
  )
  expect_identical(names(entity_coverage), feature_ids(fx$frame))
})

test_that("validity_masked_source lazily returns NA outside coverage", {
  fx <- .validity_fixture(instrument = TRUE)
  observed_masks <- fx$masks[c(1L, 1L, 2L, 3L, 3L), ]
  expected <- fx$values
  expected[!observed_masks] <- NA_real_
  source <- validity_masked_source(
    fx$source,
    observation_mask_id = fx$validity$mask_id[c(1L, 1L, 2L, 3L, 3L)],
    bank = validity_mask_bank(fx$validity)
  )

  expect_array_source_conformance(source, expected)
  reset_source_counts(fx$source)
  expect_equal(
    source_read(source, c(5L, 3L), c(6L, 2L)),
    expected[c(5L, 3L), c(6L, 2L), drop = FALSE]
  )
  expect_identical(source_counts(fx$source)$values, 4)

  all_invalid <- mask_bank(matrix(FALSE, nrow = 1L, ncol = 6L),
                           space(fx$frame))
  invalid_source <- validity_masked_source(
    memory_source(fx$values),
    rep(all_invalid$mask_ids, nrow(fx$values)), all_invalid
  )
  expect_true(all(is.na(source_read(invalid_source))))
})

test_that("apply_feature_validity masks selected assays and records provenance", {
  fx <- .validity_fixture(instrument = TRUE)
  masked <- apply_feature_validity(fx$frame, "subject_feature_validity")
  expected <- fx$values
  expected[!fx$masks[c(1L, 1L, 2L, 3L, 3L), ]] <- NA_real_

  expect_s3_class(masked, "fmri_frame")
  expect_identical(observation_ids(masked), observation_ids(fx$frame))
  expect_identical(feature_ids(masked), feature_ids(fx$frame))
  expect_identical(relation_registry_digest(masked),
                   relation_registry_digest(fx$frame))
  expect_equal(collect_assay(masked), expected)
  record <- tail(provenance_records(masked$provenance), 1L)[[1L]]
  expect_identical(record$operation, "apply_feature_validity")
})

test_that("validity relations survive FDS and HDF5 round trips", {
  fx <- .validity_fixture()
  manifest <- fds_frame_manifest(fx$frame)
  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(fx$frame))

  expect_s3_class(relation(rebuilt, "subject_feature_validity"),
                  "entity_feature_validity")
  expect_identical(validity_matrix(rebuilt, "subject_feature_validity"), fx$masks)

  invalid <- manifest
  invalid$relations$subject_feature_validity$bank$space$namespace <- "tampered"
  expect_error(validate_fds_manifest(invalid), class = "fmridataset_error_schema")

  skip_if_not_installed("fmristore")
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
  path <- tempfile(fileext = ".fds.h5")
  on.exit(unlink(path), add = TRUE)
  write_frame(fx$frame, path)
  reopened <- open_frame(path)
  expect_identical(validity_matrix(reopened, "subject_feature_validity"), fx$masks)
})

test_that("feature mapping drops source-domain validity explicitly", {
  fx <- .validity_fixture()
  target <- index_space(2L, ids = c("target-1", "target-2"),
                        namespace = "validity-target")
  map <- feature_map(
    space(fx$frame), target,
    matrix(c(1, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1), nrow = 2L, byrow = TRUE)
  )
  mapped <- map_features(fx$frame, map = map)

  expect_false("subject_feature_validity" %in% relation_names(mapped))
  expect_true("observation_subject" %in% relation_names(mapped))
})

test_that("bind_observations preserves only identical validity semantics", {
  fx <- .validity_fixture()
  split_frame <- function(x, index) {
    fmri_frame(
      assays = lapply(names(assays(x)), function(name) {
        source_view(assay(x, name)$source, observations = index)
      }) |> stats::setNames(names(assays(x))),
      observations = observation_axis(x)[index],
      features = feature_axis(x),
      entities = entities(x), relations = relations(x),
      active_assay = active_assay(x)
    )
  }
  first <- split_frame(fx$frame, 1:2)
  second <- split_frame(fx$frame, 3:5)
  combined <- bind_observations(first, second)

  expect_identical(
    validity_matrix(combined, "subject_feature_validity"), fx$masks
  )
  changed <- second
  changed$relations$subject_feature_validity$mask_id[[1L]] <-
    changed$relations$subject_feature_validity$mask_id[[2L]]
  expect_error(bind_observations(first, changed), "Validity relation")
})
