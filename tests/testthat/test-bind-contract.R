.bind_contract_frame <- function(frame, index, metadata = frame$metadata,
                                 provenance = frame$provenance,
                                 tables = frame$tables,
                                 active = active_assay(frame),
                                 observation_metadata = observation_axis(frame)$metadata) {
  observation <- observation_axis(frame)[index]
  observation$metadata <- observation_metadata
  fmri_frame(
    assays = lapply(names(assays(frame)), function(name) {
      prototype <- assay(frame, name)
      structure(
        list(
          source = source_view(prototype$source, observations = index),
          role = prototype$role, units = prototype$units,
          metadata = prototype$metadata
        ),
        class = "aligned_assay"
      )
    }) |> stats::setNames(names(assays(frame))),
    observations = observation,
    features = feature_axis(frame),
    entities = entities(frame),
    relations = relations(frame),
    tables = tables,
    active_assay = active,
    metadata = metadata,
    provenance = provenance
  )
}

test_that("binding preserves all compatible semantic annotations", {
  fx <- make_frame_fixture()
  frame <- fx$frame
  frame$observations$metadata <- list(grid = "estimate")
  frame$features$metadata <- list(space_role = "analysis")
  frame$assays$beta$role <- "estimate"
  frame$assays$beta$units <- "percent"
  frame$assays$beta$metadata <- list(scale = 100)
  frame$metadata <- unaligned_record(list(title = "fixture"))
  frame$tables <- list(files = auxiliary_table(
    tibble::tibble(file_id = c("f1", "f2"), path = c("a", "b")),
    key = "file_id", role = "files"
  ))
  frame$provenance <- provenance_graph(provenance_record("import"))
  left <- .bind_contract_frame(frame, 1:3)
  right <- .bind_contract_frame(frame, 4:7)

  bound <- bind_observations(left, right)

  expect_identical(observation_axis(bound)$metadata, observation_axis(frame)$metadata)
  expect_identical(feature_axis(bound)$metadata, feature_axis(frame)$metadata)
  expect_identical(assay(bound, "beta")$role, "estimate")
  expect_identical(assay(bound, "beta")$units, "percent")
  expect_identical(assay(bound, "beta")$metadata, list(scale = 100))
  expect_identical(bound$metadata, frame$metadata)
  expect_identical(table_data(bound$tables$files), table_data(frame$tables$files))
  expect_identical(entity_registry_digest(bound), entity_registry_digest(frame))
  expect_identical(relation_registry_digest(bound), relation_registry_digest(frame))
})

test_that("binding frame metadata is identical by default or explicitly merged", {
  frame <- make_frame_fixture()$frame
  left <- .bind_contract_frame(
    frame, 1:3, metadata = list(study = "A", acquisition = list(site = "X"))
  )
  right <- .bind_contract_frame(
    frame, 4:7, metadata = list(study = "A", acquisition = list(scanner = "Prisma"))
  )

  expect_error(
    bind_observations(left, right),
    "metadata",
    class = "fmridataset_error_alignment"
  )
  bound <- bind_observations(left, right, metadata_policy = "merge")
  expect_identical(bound$metadata$study, "A")
  expect_identical(bound$metadata$acquisition$site, "X")
  expect_identical(bound$metadata$acquisition$scanner, "Prisma")

  conflict <- .bind_contract_frame(
    frame, 4:7, metadata = list(study = "B", acquisition = list(site = "Y"))
  )
  expect_error(
    bind_observations(left, conflict, metadata_policy = "merge"),
    "metadata.study",
    class = "fmridataset_error_alignment"
  )
})

test_that("binding active assay differences require an explicit result", {
  frame <- make_frame_fixture()$frame
  left <- .bind_contract_frame(frame, 1:3, active = "beta")
  right <- .bind_contract_frame(frame, 4:7, active = "variance")

  expect_error(
    bind_observations(left, right),
    "active assays",
    class = "fmridataset_error_alignment"
  )
  bound <- bind_observations(left, right, active_assay = "variance")
  expect_identical(active_assay(bound), "variance")
  expect_error(
    bind_observations(left, right, active_assay = "missing"),
    "active_assay",
    class = "fmridataset_error_alignment"
  )
})

test_that("binding typed tables unions declared keys and rejects conflicts", {
  frame <- make_frame_fixture()$frame
  left_table <- auxiliary_table(
    tibble::tibble(file_id = c("f1", "shared"), path = c("a", "same")),
    key = "file_id", role = "files"
  )
  right_table <- auxiliary_table(
    tibble::tibble(file_id = c("shared", "f2"), path = c("same", "b")),
    key = "file_id", role = "files"
  )
  left <- .bind_contract_frame(frame, 1:3, tables = list(files = left_table))
  right <- .bind_contract_frame(frame, 4:7, tables = list(files = right_table))

  bound <- bind_observations(left, right)
  expect_identical(
    table_data(bound$tables$files)$file_id,
    c("f1", "shared", "f2")
  )

  conflicting <- auxiliary_table(
    tibble::tibble(file_id = c("shared", "f2"), path = c("different", "b")),
    key = "file_id", role = "files"
  )
  bad <- .bind_contract_frame(frame, 4:7, tables = list(files = conflicting))
  expect_error(
    bind_observations(left, bad),
    "conflicting rows",
    class = "fmridataset_error_table"
  )

  unkeyed_left <- auxiliary_table(tibble::tibble(note = "left"), role = "notes")
  unkeyed_right <- auxiliary_table(tibble::tibble(note = "right"), role = "notes")
  expect_error(
    bind_observations(
      .bind_contract_frame(frame, 1:3, tables = list(notes = unkeyed_left)),
      .bind_contract_frame(frame, 4:7, tables = list(notes = unkeyed_right))
    ),
    "no declared key",
    class = "fmridataset_error_table"
  )
})

test_that("binding combines lineage under a content-addressed bind node", {
  frame <- make_frame_fixture()$frame
  left_record <- provenance_record("left_import")
  right_record <- provenance_record("right_import")
  left <- .bind_contract_frame(
    frame, 1:3, provenance = provenance_graph(left_record)
  )
  right <- .bind_contract_frame(
    frame, 4:7, provenance = provenance_graph(right_record)
  )

  bound <- bind_observations(left, right)
  records <- provenance_records(bound$provenance)
  tip <- records[[provenance_tips(bound$provenance)]]
  expect_identical(tip$operation, "bind_observations")
  expect_setequal(tip$parents, c(left_record$id, right_record$id))
  expect_identical(tip$inputs$observation_ids, list(
    observation_ids(left), observation_ids(right)
  ))
})

test_that("binding is lazy, order-preserving, and associative in data semantics", {
  fx <- make_frame_fixture(instrument = TRUE)
  a <- fx$frame[c(7L, 2L), ]
  b <- fx$frame[c(4L, 5L), ]
  c <- fx$frame[c(1L, 3L, 6L), ]

  expect_true(all(vapply(
    assays(fx$frame), function(value) source_counts(value$source)$reads == 0,
    logical(1)
  )))

  left <- bind_observations(bind_observations(a, b), c)
  right <- bind_observations(a, bind_observations(b, c))

  expect_identical(observation_ids(left), observation_ids(right))
  expect_identical(frame_schema(left), frame_schema(right))
  expect_identical(
    source_fingerprint(assay(left)$source),
    source_fingerprint(assay(right)$source)
  )
  expect_equal(collect_assay(left), collect_assay(right), tolerance = 0)
  expect_gt(source_counts(assay(fx$frame, "beta")$source)$reads, 0)
  expect_identical(source_counts(assay(fx$frame, "variance")$source)$reads, 0)
})

test_that("binding handles empty and single operands without semantic loss", {
  frame <- make_frame_fixture()$frame
  empty <- frame[integer(), ]
  nonempty <- frame[c(3L, 1L), ]

  combined <- bind_observations(empty, nonempty)
  expect_identical(observation_ids(combined), observation_ids(nonempty))
  expect_equal(collect_assay(combined), collect_assay(nonempty), tolerance = 0)

  second_empty <- frame[integer(), ]
  second_empty$observations$data$.obs_id <- character()
  second_empty$observations$ids <- character()
  all_empty <- bind_observations(empty, second_empty)
  expect_identical(dim(all_empty), c(0L, ncol(frame)))
  expect_identical(dim(collect_assay(all_empty)), c(0L, ncol(frame)))

  single <- bind_observations(nonempty)
  expect_identical(observation_ids(single), observation_ids(nonempty))
  expect_equal(collect_assay(single), collect_assay(nonempty), tolerance = 0)
})
