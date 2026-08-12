test_that("container metadata is a typed unaligned record", {
  fx <- make_frame_fixture(instrument = TRUE)
  graph <- provenance_graph(provenance_record("import"))
  frame <- fmri_frame(
    assays = lapply(assays(fx$frame), function(value) value$source),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = entities(fx$frame),
    relations = relations(fx$frame),
    metadata = list(title = "fixture", acquisition = list(site = "A")),
    provenance = graph
  )

  expect_s3_class(frame$metadata, "unaligned_record")
  expect_identical(frame$metadata$title, "fixture")
  expect_s3_class(frame$metadata$acquisition, "unaligned_record")
  expect_identical(frame$provenance, graph)
  expect_true(all(vapply(
    assays(frame), function(value) source_counts(value$source)$bytes == 0,
    logical(1)
  )))
})

test_that("container metadata rejects hidden alignment and result diagnostics", {
  fx <- make_frame_fixture(instrument = TRUE)
  rebuild <- function(metadata) fmri_frame(
    assays = lapply(assays(fx$frame), function(value) value$source),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = entities(fx$frame),
    relations = relations(fx$frame),
    metadata = metadata
  )

  expect_error(
    rebuild(list(per_observation = seq_len(nrow(fx$frame)))),
    "observation-aligned",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    rebuild(list(nested = list(per_feature = seq_len(ncol(fx$frame))))),
    "feature-aligned",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    rebuild(list(stimulus_score = seq_len(length(entity(fx$frame, "stimulus"))))),
    "entity:stimulus-aligned",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    rebuild(list(convergence = "success")),
    "diagnostic",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    rebuild(list(fit = list(log_likelihood = -12.5))),
    "diagnostic",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    rebuild(list(table = data.frame(value = 1:2))),
    "typed table",
    class = "fmridataset_error_metadata"
  )
  expect_true(all(vapply(
    assays(fx$frame), function(value) source_counts(value$source)$bytes == 0,
    logical(1)
  )))
})

test_that("auxiliary tables are typed and frame table registries reject data frames", {
  fx <- make_frame_fixture()
  files <- auxiliary_table(
    tibble::tibble(file_id = c("f1", "f2"), path = c("a.nii", "b.nii")),
    key = "file_id",
    role = "files",
    metadata = list(source = "BIDS")
  )
  frame <- fmri_frame(
    assays = lapply(assays(fx$frame), function(value) value$source),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = entities(fx$frame),
    relations = relations(fx$frame),
    tables = list(files = files)
  )

  expect_s3_class(frame$tables$files, "fmri_auxiliary_table")
  expect_identical(table_key(frame$tables$files), "file_id")
  expect_identical(table_role(frame$tables$files), "files")
  expect_identical(table_data(frame$tables$files), files$data)
  expect_error(
    fmri_frame(
      assays = list(signal = matrix(1:6, nrow = 2L)),
      observations = data.frame(.obs_id = c("o1", "o2")),
      space = index_space(3L),
      tables = list(files = data.frame(file_id = "f1"))
    ),
    "typed table",
    class = "fmridataset_error_table"
  )
})

test_that("lineage requires a provenance graph and legacy lists migrate explicitly", {
  fx <- make_frame_fixture()
  args <- list(
    assays = lapply(assays(fx$frame), function(value) value$source),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = entities(fx$frame),
    relations = relations(fx$frame)
  )

  expect_error(
    do.call(fmri_frame, c(args, list(provenance = list(step = "legacy")))),
    "as_provenance_graph",
    class = "fmridataset_error_provenance"
  )
  graph <- as_provenance_graph(list(step = "legacy"))
  frame <- do.call(fmri_frame, c(args, list(provenance = graph)))
  expect_s3_class(frame$provenance, "provenance_graph")
  record <- provenance_records(frame$provenance)[[1L]]
  expect_identical(record$operation, "legacy_provenance")
  expect_identical(record$inputs$value, list(step = "legacy"))

  collection <- fmri_collection(list(one = frame), provenance = graph)
  study <- fmri_study(
    list(beta = frame), entities = entities(frame), provenance = graph
  )
  expect_identical(collection$provenance, graph)
  expect_identical(study$provenance, graph)
  expect_error(
    fmri_collection(list(one = frame), provenance = list(step = "legacy")),
    class = "fmridataset_error_provenance"
  )
  expect_error(
    fmri_study(
      list(beta = frame), entities = entities(frame),
      provenance = list(step = "legacy")
    ),
    class = "fmridataset_error_provenance"
  )
  expect_error(
    fmri_collection(
      list(one = frame), metadata = list(per_observation = seq_len(nrow(frame)))
    ),
    "observation-aligned",
    class = "fmridataset_error_metadata"
  )
  expect_error(
    fmri_study(
      list(beta = frame), entities = entities(frame),
      metadata = list(per_feature = seq_len(ncol(frame)))
    ),
    "feature-aligned",
    class = "fmridataset_error_metadata"
  )
})

test_that("views and FDS preserve typed metadata tables and lineage", {
  fx <- make_frame_fixture()
  graph <- provenance_graph(provenance_record("import"))
  files <- auxiliary_table(
    tibble::tibble(file_id = "f1", path = "beta.h5"),
    key = "file_id", role = "files"
  )
  frame <- fmri_frame(
    assays = lapply(assays(fx$frame), function(value) value$source),
    observations = observation_axis(fx$frame),
    features = feature_axis(fx$frame),
    entities = entities(fx$frame),
    relations = relations(fx$frame),
    tables = list(files = files),
    metadata = list(title = "typed"),
    provenance = graph
  )
  view <- frame[1:2, 1:3]
  manifest <- fds_frame_manifest(view)
  rebuilt <- frame_from_fds_manifest(manifest, fds_frame_bindings(view))

  expect_s3_class(rebuilt$metadata, "unaligned_record")
  expect_s3_class(rebuilt$tables$files, "fmri_auxiliary_table")
  expect_identical(rebuilt$provenance, graph)

  malformed <- manifest
  malformed$metadata <- unaligned_record(list(per_observation = 1:2))
  expect_error(validate_fds_manifest(malformed), "observation-aligned")
  malformed <- manifest
  malformed$provenance <- list(step = "legacy")
  expect_error(validate_fds_manifest(malformed), "provenance_graph")
  malformed <- manifest
  malformed$tables$files <- data.frame(file_id = "f1")
  expect_error(validate_fds_manifest(malformed), "typed table")
})
