test_that("FDS v1 frame manifests are explicit and backend-neutral", {
  fx <- make_frame_fixture()
  manifest <- fds_frame_manifest(fx$frame)

  expect_identical(fds_schema_version(), 1L)
  expect_identical(fds_schema()$id, "org.fmridataset.fds/v1")
  expect_identical(manifest$schema, fds_schema())
  expect_identical(manifest$object_type, "fmri_frame")
  expect_identical(manifest$shape, c(7L, 6L))
  expect_identical(manifest$axes$observation$ids, observation_ids(fx$frame))
  expect_identical(manifest$axes$feature$ids, feature_ids(fx$frame))
  expect_identical(
    space_digest(manifest$axes$feature$space),
    space_digest(space(fx$frame))
  )
  expect_identical(
    unname(vapply(manifest$assays, `[[`, character(1), "name")),
    c("beta", "variance")
  )
  expect_identical(
    names(manifest$arrays),
    c("assays/beta", "assays/variance", "axis/observation/blocks/motion")
  )
  expect_false(any(vapply(
    manifest$assays,
    function(value) any(c("source", "uri", "dataset", "chunks") %in% names(value)),
    logical(1)
  )))
  expect_false(any(vapply(manifest$arrays, inherits, logical(1), "array_source")))
  expect_false(contains_runtime_state(manifest))
  expect_invisible(validate_fds_manifest(manifest))
  expect_identical(fds_manifest_digest(manifest), fds_manifest_digest(manifest))
})

test_that("FDS manifests do not change with physical source layout", {
  fx <- make_frame_fixture()
  rechunked <- fmri_frame(
    assays = list(
      beta = memory_source(fx$beta, chunks = c(1L, 2L)),
      variance = memory_source(fx$variance, chunks = c(2L, 3L))
    ),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = fx$frame$entities,
    relations = fx$frame$relations,
    tables = fx$frame$tables,
    active_assay = active_assay(fx$frame),
    metadata = fx$frame$metadata,
    provenance = fx$frame$provenance
  )

  expect_identical(fds_frame_manifest(rechunked), fds_frame_manifest(fx$frame))
})

test_that("FDS array declarations retain higher-dimensional block axes", {
  fx <- make_frame_fixture()
  tensor <- array(seq_len(7L * 2L * 3L), dim = c(7L, 2L, 3L))
  tensor_block <- axis_block(
    tensor,
    components = data.frame(.component_id = c("channel-a", "channel-b"))
  )
  observation <- observation_axis(fx$frame)
  observation <- axis_frame(
    observation$data,
    blocks = c(observation$blocks, list(tensor = tensor_block)),
    id = observation_ids(fx$frame),
    axis = "observation",
    id_col = observation$id_col,
    metadata = observation$metadata
  )
  x <- fmri_frame(
    assays = lapply(assays(fx$frame), `[[`, "source"),
    observations = observation,
    features = feature_axis(fx$frame),
    entities = fx$frame$entities,
    active_assay = "beta"
  )
  manifest <- fds_frame_manifest(x)
  declaration <- manifest$arrays[["axis/observation/blocks/tensor"]]

  expect_identical(declaration$shape, c(7L, 2L, 3L))
  expect_identical(
    declaration$axes,
    c(
      "observation",
      "component:axis/observation/blocks/tensor",
      "dimension:axis/observation/blocks/tensor:3"
    )
  )
  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(x))
  expect_identical(axis_block_data(obs_blocks(rebuilt)$tensor), tensor)
})

test_that("FDS manifests rebuild frames from separately bound sources", {
  fx <- make_frame_fixture()
  manifest <- fds_frame_manifest(fx$frame)
  bindings <- fds_frame_bindings(fx$frame)
  rebuilt <- frame_from_fds_manifest(manifest, bindings)

  expect_identical(observations(rebuilt), observations(fx$frame))
  expect_identical(feature_ids(rebuilt), feature_ids(fx$frame))
  expect_identical(space_digest(space(rebuilt)), space_digest(space(fx$frame)))
  expect_identical(names(obs_blocks(rebuilt)), names(obs_blocks(fx$frame)))
  expect_identical(rebuilt$entities, fx$frame$entities)
  expect_identical(rebuilt$relations, fx$frame$relations)
  expect_identical(rebuilt$tables, fx$frame$tables)
  expect_identical(active_assay(rebuilt), active_assay(fx$frame))
  expect_equal(collect_assay(rebuilt, "beta"), fx$beta)
  expect_equal(collect_assay(rebuilt, "variance"), fx$variance)
})

test_that("FDS v1 validation rejects semantic drift", {
  manifest <- fds_frame_manifest(make_frame_fixture()$frame)

  future <- manifest
  future$schema$version <- 2L
  expect_error(validate_fds_manifest(future), "version", class = "fmridataset_error_schema")

  wrong_shape <- manifest
  wrong_shape$shape <- c(8L, 6L)
  expect_error(validate_fds_manifest(wrong_shape), "shape", class = "fmridataset_error_schema")

  duplicate_ids <- manifest
  duplicate_ids$axes$observation$ids[[2L]] <- duplicate_ids$axes$observation$ids[[1L]]
  expect_error(validate_fds_manifest(duplicate_ids), "IDs", class = "fmridataset_error_schema")

  wrong_digest <- manifest
  wrong_digest$assays$beta$feature_digest <- "not-the-feature-axis"
  expect_error(validate_fds_manifest(wrong_digest), "digest", class = "fmridataset_error_schema")

  runtime <- manifest
  runtime$metadata$loader <- function() NULL
  expect_error(validate_fds_manifest(runtime), "runtime", class = "fmridataset_error_schema")
})

test_that("frame reconstruction rejects incompatible physical bindings", {
  fx <- make_frame_fixture()
  manifest <- fds_frame_manifest(fx$frame)
  bindings <- fds_frame_bindings(fx$frame)

  missing <- bindings["assays/beta"]
  expect_error(frame_from_fds_manifest(manifest, missing), "exactly match")

  wrong_shape <- bindings
  wrong_shape[["assays/beta"]] <- memory_source(matrix(1:48, 8, 6))
  expect_error(frame_from_fds_manifest(manifest, wrong_shape), "shape")

  wrong_dtype <- bindings
  wrong_dtype[["assays/beta"]] <- memory_source(fx$beta, dtype = "float32")
  expect_error(frame_from_fds_manifest(manifest, wrong_dtype), "dtype")
})

test_that("the installed FDS v1 schema envelope is machine readable", {
  skip_if_not_installed("jsonlite")
  path <- system.file("schema", "fds-v1.schema.json", package = "fmridataset")
  expect_true(file.exists(path))
  schema <- jsonlite::read_json(path, simplifyVector = TRUE)

  expect_identical(schema[["$id"]], "org.fmridataset.fds/v1")
  expect_true(all(c("schema", "object_type", "shape", "axes", "arrays", "assays") %in% schema$required))
  expect_identical(schema$properties$schema$properties$version$const, 1L)
})
