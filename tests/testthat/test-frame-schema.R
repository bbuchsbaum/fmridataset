test_that("canonical frame schema is complete and performs zero numerical reads", {
  frame <- make_frame_fixture(instrument = TRUE)$frame
  schema <- frame_schema(frame)

  expect_s3_class(schema, "fmri_frame_schema")
  expect_identical(names(schema$assays), c("beta", "variance"))
  expect_identical(schema$assays$beta$dtype, "float64")
  expect_identical(schema$observation$columns$Fac1$levels, c("A", "B"))
  expect_identical(
    schema$observation$blocks$motion$component_ids,
    c("translation", "rotation")
  )
  expect_identical(schema$observation$blocks$motion$trailing_shape, 2L)
  expect_identical(schema$feature$space$digest, space_digest(space(frame)))
  expect_identical(schema$entities$stimulus$key, "stimulus_id")
  expect_identical(schema$active_assay, list(policy = "named", name = "beta"))
  expect_true(all(vapply(
    assays(frame), function(value) source_counts(value$source)$bytes, numeric(1)
  ) == 0))
})

test_that("same, collection, and bind compatibility have distinct laws", {
  frame <- make_frame_fixture()$frame
  subset <- frame[1:3, ]

  expect_false(compare_frame_schema(subset, frame, mode = "same")$compatible)
  expect_true(compare_frame_schema(subset, frame, mode = "collection")$compatible)
  expect_true(compare_frame_schema(subset, frame, mode = "bind")$compatible)

  other_space <- volume_space(c(3L, 2L, 1L), support = 1:6, template = "native-2")
  native <- fmri_frame(
    assays = lapply(assays(frame), function(value) value$source),
    observations = observation_axis(frame),
    features = feature_axis(features(frame), space = other_space),
    entities = entities(frame), active_assay = active_assay(frame)
  )
  expect_true(compare_frame_schema(native, frame, mode = "collection")$compatible)
  bind_report <- compare_frame_schema(native, frame, mode = "bind")
  expect_false(bind_report$compatible)
  expect_identical(bind_report$path, "schema.feature.space.digest")
})

test_that("schema mismatches report their first semantic path", {
  frame <- make_frame_fixture()$frame
  changed <- frame
  changed$observations$data$Fac1 <- factor(
    changed$observations$data$Fac1, levels = c("B", "A")
  )
  report <- compare_frame_schema(changed, frame)
  expect_false(report$compatible)
  expect_identical(report$path, "schema.observation.columns.Fac1.levels")
  expect_error(
    validate_against_schema(changed, frame),
    "schema.observation.columns.Fac1.levels",
    class = "fmridataset_error_schema"
  )

  changed_block <- frame
  changed_block$observations$blocks$motion$components$.component_id <- c("x", "y")
  expect_identical(
    compare_frame_schema(changed_block, frame)$path,
    "schema.observation.blocks.motion.component_ids"
  )
})

test_that("collection and binding use canonical schema diagnostics", {
  frame <- make_frame_fixture()$frame
  changed <- frame
  changed$assays$beta$units <- "percent"

  expect_error(
    fmri_collection(list(reference = frame, changed = changed)),
    "schema.assays.beta.units",
    class = "fmridataset_error_collection"
  )
  expect_error(
    bind_observations(frame[1:3, ], changed[4:7, ]),
    "schema.assays.beta.units",
    class = "fmridataset_error_schema"
  )
})

test_that("binding preserves assay annotations accepted by the schema", {
  frame <- make_frame_fixture()$frame
  frame$assays$beta$role <- "estimate"
  frame$assays$beta$units <- "percent"
  frame$assays$beta$metadata <- list(scale = 100)

  bound <- bind_observations(frame[1:3, ], frame[4:7, ])
  expect_identical(assay(bound, "beta")$role, "estimate")
  expect_identical(assay(bound, "beta")$units, "percent")
  expect_identical(assay(bound, "beta")$metadata, list(scale = 100))
  expect_true(compare_frame_schema(bound, frame, mode = "bind")$compatible)
})

test_that("FDS manifests and live frames derive the identical canonical schema", {
  frame <- make_frame_fixture()$frame
  manifest <- fds_frame_manifest(frame)
  from_manifest <- fmridataset:::.frame_schema_from_manifest(manifest)

  expect_identical(from_manifest, frame_schema(frame))
  expect_identical(frame_schema_digest(from_manifest), frame_schema_digest(frame))
  expect_invisible(validate_fds_manifest(manifest))
})

test_that("explain reports the canonical schema digest without reads", {
  frame <- make_frame_fixture(instrument = TRUE)$frame
  plan <- explain(frame)

  expect_identical(plan$digests$schema, frame_schema_digest(frame))
  expect_true(all(vapply(
    assays(frame), function(value) source_counts(value$source)$bytes, numeric(1)
  ) == 0))
})
