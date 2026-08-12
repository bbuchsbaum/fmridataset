test_that("as_fmri_frame preserves canonical frames without realization", {
  fixture <- make_frame_fixture(instrument = TRUE)
  frame <- fixture$frame

  converted <- as_fmri_frame(frame)

  expect_identical(converted, frame)
  expect_true(all(vapply(
    assays(frame),
    function(value) source_counts(value$source)$bytes == 0,
    logical(1)
  )))
  expect_identical(class(frame), "fmri_frame")
})

test_that("as_fmri_frame has an explicit unsupported-object failure", {
  expect_error(
    as_fmri_frame(matrix(1:4, 2L)),
    "No as_fmri_frame method"
  )
})

test_that("canonical frames and views cannot fall through legacy dispatch", {
  frame <- make_frame_fixture()$frame
  view <- frame[1:2, 1:3]

  expect_false(inherits(frame, "fmri_dataset"))
  expect_false(inherits(view, "fmri_dataset"))
  expect_identical(class(view), c("fmri_view", "fmri_frame"))
  namespace <- asNamespace("fmridataset")
  expect_false(exists("get_TR", envir = namespace, inherits = FALSE))
  expect_false(exists("fmri_series", envir = namespace, inherits = FALSE))
})

test_that("provisional canonical frames migrate without assay reads", {
  fixture <- make_frame_fixture(instrument = TRUE)
  provisional <- fixture$frame
  class(provisional) <- c("fmri_frame", "fmri_dataset")

  expect_warning(
    upgraded <- upgrade_dataset(provisional),
    "provisional canonical frame"
  )
  expect_identical(class(upgraded), "fmri_frame")
  expect_identical(observation_ids(upgraded), observation_ids(fixture$frame))
  expect_true(all(vapply(
    assays(upgraded),
    function(value) source_counts(value$source)$bytes == 0,
    logical(1)
  )))

  path <- tempfile(fileext = ".rds")
  saveRDS(provisional, path)
  reopened <- readRDS(path)
  expect_warning(reopened <- upgrade_dataset(reopened), "provisional canonical frame")
  expect_identical(class(reopened), "fmri_frame")

  future <- provisional
  future$schema_version <- 2L
  expect_error(
    upgrade_dataset(future, warn = FALSE),
    "Unsupported provisional fmri_frame schema version",
    class = "fmridataset_error_schema"
  )
})

test_that("matrix_dataset migration preserves values and temporal structure", {
  values <- matrix(seq_len(24), nrow = 6L, ncol = 4L)
  legacy <- legacy_matrix_dataset(values, TR = 1.5, run_length = c(2L, 4L))

  expect_warning(frame <- upgrade_dataset(legacy), "matrix_dataset")
  expect_identical(class(frame), "fmri_frame")
  expect_identical(names(assays(frame)), "signal")
  expect_equal(collect_assay(frame), values)
  expect_identical(observations(frame)$run_id, c("run-001", "run-001", rep("run-002", 4L)))
  expect_identical(observations(frame)$run_timepoint, c(1L, 2L, 1L, 2L, 3L, 4L))
  expect_equal(observations(frame)$time, c(0, 1.5, 0, 1.5, 3, 4.5))
  expect_identical(frame$metadata$migration$source_class, "matrix_dataset")
  expect_s3_class(frame$tables$events, "fmri_auxiliary_table")
  expect_identical(table_data(frame$tables$events), tibble::as_tibble(legacy$event_table))

  again <- as_fmri_frame(legacy)
  expect_identical(observation_ids(again), observation_ids(frame))
  expect_identical(feature_ids(again), feature_ids(frame))

  same_shape <- legacy_matrix_dataset(values + 1, TR = 1.5, run_length = c(2L, 4L))
  expect_false(compatible_space(space(frame), space(as_fmri_frame(same_shape)))$compatible)
})

test_that("in-memory fmri_series migration preserves aligned metadata", {
  series <- structure(
    list(
      data = matrix(seq_len(9), nrow = 3L),
      voxel_info = data.frame(voxel = c(8L, 3L, 1L)),
      temporal_info = data.frame(run_id = c(1L, 1L, 2L), timepoint = 1:3),
      selection_info = list(selector = c(8L, 3L, 1L)),
      dataset_info = list(backend_type = "fixture")
    ),
    class = "fmri_series"
  )

  frame <- as_fmri_frame(series)
  expect_identical(class(frame), "fmri_frame")
  expect_equal(collect_assay(frame), series$data)
  expect_identical(observations(frame)$run_id, series$temporal_info$run_id)
  expect_identical(features(frame)$voxel, series$voxel_info$voxel)
  migrated <- provenance_records(frame$provenance)[[1L]]$inputs$value
  expect_identical(migrated$selection_info, series$selection_info)
})

test_that("legacy golden inputs have explicit migration outcomes", {
  golden <- file.path("golden")
  matrix_legacy <- readRDS(file.path(golden, "matrix_dataset.rds"))
  series_envelope <- readRDS(file.path(golden, "fmri_series.rds"))
  sampling <- readRDS(file.path(golden, "sampling_frame.rds"))
  neurovec <- readRDS(file.path(golden, "mock_neurvec.rds"))

  matrix_frame <- upgrade_dataset(matrix_legacy, warn = FALSE)
  series_frame <- upgrade_dataset(series_envelope, warn = FALSE)
  neurovec_frame <- upgrade_dataset(neurovec, TR = 2, warn = FALSE)

  expect_equal(collect_assay(matrix_frame), matrix_legacy$datamat)
  expect_equal(collect_assay(series_frame), series_envelope$data)
  expect_identical(dim(neurovec_frame), c(50L, 100L))
  expect_equal(collect_assay(neurovec_frame), t(matrix(as.numeric(neurovec), nrow = 100L)))
  expect_s3_class(space(neurovec_frame), "volume_space")
  expect_error(upgrade_dataset(sampling), "not assay-bearing")
})

test_that("ambiguous legacy datasets fail before numerical reads", {
  legacy <- structure(
    list(backend = fault_source(memory_source(matrix(1:4, 2L)), stage = "read")),
    class = c("fmri_file_dataset", "fmri_dataset", "list")
  )
  expect_error(
    as_fmri_frame(legacy),
    "no supported self-contained migration",
    class = "fmridataset_error_schema"
  )
})
