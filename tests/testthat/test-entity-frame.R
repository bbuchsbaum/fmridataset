test_that("entity_frame owns stable scalar keys and aligned blocks", {
  embedding <- axis_block(
    matrix(as.double(1:12), 3, 4),
    components = tibble::tibble(.component_id = paste0("E", 1:4)),
    role = "embedding"
  )
  x <- entity_frame(
    key = "stimulus_id",
    data = tibble::tibble(
      stimulus_id = factor(c("stim-1", "stim-2", "stim-3")),
      category = factor(c("face", "scene", "object"))
    ),
    blocks = list(semantic = embedding),
    entity_type = "stimulus",
    metadata = list(source = "fixture")
  )

  expect_s3_class(x, "entity_frame")
  expect_s3_class(x, "axis_frame")
  expect_identical(entity_key(x), "stimulus_id")
  expect_identical(entity_ids(x), c("stim-1", "stim-2", "stim-3"))
  expect_identical(entity_data(x)$stimulus_id, entity_ids(x))
  expect_true(is.factor(entity_data(x)$category))
  expect_identical(names(entity_blocks(x)), "semantic")
  expect_identical(block_component_ids(entity_blocks(x)$semantic), paste0("E", 1:4))
  expect_identical(x$entity_type, "stimulus")
  expect_false(contains_runtime_state(x))
})

test_that("entity_frame rejects ambiguous scalar tables and misaligned blocks", {
  expect_error(
    entity_frame(tibble::tibble(label = c("a", "b")), key = "subject_id"),
    class = "fmridataset_error_entity"
  )
  expect_error(
    entity_frame(
      tibble::tibble(subject_id = c("sub-1", "sub-1")),
      key = "subject_id"
    ),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    entity_frame(
      tibble::tibble(subject_id = c("sub-1", NA_character_)),
      key = "subject_id"
    ),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    entity_frame(
      tibble::tibble(
        subject_id = c("sub-1", "sub-2"),
        opaque = list(1:2, 3:4)
      ),
      key = "subject_id"
    ),
    "scalar"
  )
  expect_error(
    entity_frame(
      tibble::tibble(subject_id = c("sub-1", "sub-2")),
      key = "subject_id",
      blocks = list(scores = axis_block(matrix(1:3, 3, 1)))
    ),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    entity_frame(
      tibble::tibble(subject_id = "sub-1"),
      key = "subject_id",
      metadata = list(loader = function() NULL)
    ),
    class = "fmridataset_error_entity"
  )
})

test_that("entity_frame subsetting synchronizes data and blocks", {
  x <- entity_frame(
    tibble::tibble(stimulus_id = paste0("stim-", 1:4), kind = letters[1:4]),
    key = "stimulus_id",
    blocks = list(scores = axis_block(matrix(as.double(1:12), 4, 3)))
  )
  selected <- x[c(4L, 2L)]

  expect_s3_class(selected, "entity_frame")
  expect_identical(entity_key(selected), "stimulus_id")
  expect_identical(entity_ids(selected), c("stim-4", "stim-2"))
  expect_identical(entity_data(selected)$kind, c("d", "b"))
  expect_equal(
    axis_block_data(entity_blocks(selected)$scores),
    matrix(as.double(1:12), 4, 3)[c(4L, 2L), , drop = FALSE],
    tolerance = 0
  )
})

test_that("entity registries are named validated semantic collections", {
  subjects <- entity_frame(
    tibble::tibble(subject_id = c("sub-1", "sub-2"), age = c(63, 71)),
    key = "subject_id",
    entity_type = "subject"
  )
  stimuli <- entity_frame(
    tibble::tibble(stimulus_id = c("stim-1", "stim-2")),
    key = "stimulus_id"
  )
  registry <- entity_registry(subject = subjects, stimulus = stimuli)

  expect_s3_class(registry, "entity_registry")
  expect_identical(entity_names(registry), c("subject", "stimulus"))
  expect_identical(entity(registry, "subject"), subjects)
  expect_identical(entities(registry), registry)
  expect_invisible(validate_entity_registry(registry))
  expect_false(contains_runtime_state(registry))
  expect_match(entity_registry_digest(registry), "^[0-9a-f]{64}$")
  expect_error(entity(registry, "run"), class = "fmridataset_error_entity")

  expect_error(entity_registry(list(subjects)), "named")
  # The pre-1.0 list-entry form is gone: entries must be entity frames, even
  # when a conventional `<name>_id` column could have supplied the key.
  expect_error(
    entity_registry(list(subject = list(data = tibble::tibble(subject_id = "sub-1")))),
    "must be an entity_frame",
    class = "fmridataset_error_entity"
  )
  # An entity_frame passed positionally is rejected with a clear message
  # instead of being read as a list of registry entries.
  expect_error(
    entity_registry(subjects),
    "by name",
    class = "fmridataset_error_entity"
  )
  unsafe <- unclass(registry)
  unsafe$bad <- list(data = tibble::tibble(bad_id = "bad"), key = "bad_id")
  class(unsafe) <- c("entity_registry", "list")
  expect_error(validate_entity_registry(unsafe), class = "fmridataset_error_entity")

  misaligned <- registry
  misaligned$stimulus$blocks <- list(
    bad = axis_block(matrix(1:3, nrow = 1L))
  )
  expect_error(
    validate_entity_registry(misaligned),
    "not aligned",
    class = "fmridataset_error_entity"
  )
})

test_that("fmri_frame normalizes and exposes entity registries", {
  fixture <- make_frame_fixture()
  registry <- entities(fixture$frame)

  expect_s3_class(registry, "entity_registry")
  expect_s3_class(entity(fixture$frame, "stimulus"), "entity_frame")
  expect_identical(entity_names(fixture$frame), "stimulus")
  expect_identical(entities(fixture$frame[1:2, 1:2]), registry)

  no_entities <- fmri_frame(
    assays = list(signal = matrix(1:4, 2, 2)),
    observations = tibble::tibble(.obs_id = c("o1", "o2")),
    space = index_space(
      2,
      namespace = "entity-frame", id_policy = "deterministic"
    )
  )
  expect_s3_class(entities(no_entities), "entity_registry")
  expect_length(entities(no_entities), 0L)
})

test_that("bind_observations requires identical entity registries", {
  fixture <- make_frame_fixture()
  left <- fixture$frame[1:2, ]
  right_base <- fixture$frame
  changed <- entity_frame(
    tibble::tibble(
      stimulus_id = c("stim-1", "stim-2", "stim-3"),
      category = c("changed", "scene", "object")
    ),
    key = "stimulus_id"
  )
  right_base$entities <- entity_registry(stimulus = changed)
  right <- right_base[3:4, ]

  expect_error(
    bind_observations(left, right),
    "entity registries",
    class = "fmridataset_error_entity"
  )
})

test_that("FDS declares entity blocks as source-free named-axis arrays", {
  fixture <- make_frame_fixture()
  manifest <- fds_frame_manifest(fixture$frame)
  key <- "entities/stimulus/blocks/visual_pca"

  expect_true(key %in% names(manifest$arrays))
  expect_identical(
    manifest$arrays[[key]]$axes[1:2],
    c("entity:stimulus", paste0("component:", key))
  )
  expect_false(inherits(manifest$entities$stimulus, "entity_frame"))
  expect_false("data" %in% names(manifest$entities$stimulus$blocks$visual_pca))
  expect_false(contains_runtime_state(manifest))

  duplicate <- manifest
  duplicate$entities$stimulus$ids[[2L]] <- duplicate$entities$stimulus$ids[[1L]]
  expect_error(validate_fds_manifest(duplicate), class = "fmridataset_error_schema")
  missing_array <- manifest
  missing_array$entities$stimulus$blocks$visual_pca$array <- "missing"
  expect_error(validate_fds_manifest(missing_array), class = "fmridataset_error_schema")

  bindings <- fds_frame_bindings(fixture$frame)
  expect_true(inherits(bindings[[key]], "array_source"))
  rebuilt <- frame_from_fds_manifest(manifest, bindings)
  expect_identical(entity_names(rebuilt), "stimulus")
  expect_identical(
    entity_data(entity(rebuilt, "stimulus")),
    entity_data(entity(fixture$frame, "stimulus"))
  )
  expect_equal(
    source_read(as_array_source(axis_block_data(
      entity_blocks(entity(rebuilt, "stimulus"))$visual_pca
    ))),
    source_read(as_array_source(axis_block_data(
      entity_blocks(entity(fixture$frame, "stimulus"))$visual_pca
    ))),
    tolerance = 0
  )
})

test_that("entity registries survive HDF5 round trips", {
  skip_if_not_installed("fmristore")
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
  fixture <- make_frame_fixture()
  path <- tempfile(fileext = ".fds.h5")
  on.exit(unlink(path), add = TRUE)

  write_frame(fixture$frame, path)
  rebuilt <- open_frame(path)
  expect_identical(entity_names(rebuilt), entity_names(fixture$frame))
  expect_identical(
    entity_data(entity(rebuilt, "stimulus")),
    entity_data(entity(fixture$frame, "stimulus"))
  )
  expect_equal(
    source_read(as_array_source(axis_block_data(
      entity_blocks(entity(rebuilt, "stimulus"))$visual_pca
    ))),
    axis_block_data(entity_blocks(entity(fixture$frame, "stimulus"))$visual_pca),
    tolerance = 0
  )
})

test_that("numeric entity and foreign keys use exact whole-number identity", {
  subjects <- entity_frame(
    data.frame(subject_id = c(100000, 200000), age = c(20, 60)),
    key = "subject_id"
  )
  expect_identical(entity_ids(subjects), c("100000", "200000"))
  expect_identical(
    entity_ids(entity_frame(data.frame(k = c(1e15, -0, 7L)), key = "k")),
    c("1000000000000000", "0", "7")
  )
  expect_error(
    entity_frame(data.frame(k = c(0.1 + 0.2, 0.3)), key = "k"),
    "whole numbers",
    class = "fmridataset_error_entity"
  )
  expect_error(
    entity_frame(data.frame(k = c(1, Inf)), key = "k"),
    class = "fmridataset_error_entity"
  )
  expect_error(
    entity_frame(data.frame(k = c(TRUE, FALSE)), key = "k"),
    class = "fmridataset_error_entity"
  )

  make <- function(foreign) {
    fmri_frame(
      assays = list(bold = matrix(as.double(1:8), nrow = 2)),
      observations = data.frame(.obs_id = c("o1", "o2"), subject_id = foreign),
      space = index_space(4, ids = paste0("f", 1:4), namespace = "numeric-keys"),
      entities = list(subject = subjects),
      relations = list(os = key_relation("subject_id", target = "subject"))
    )
  }
  # Integer and double foreign keys both match the double primary keys.
  for (foreign in list(c(100000L, 200000L), c(100000, 200000), c("100000", "200000"))) {
    resolved <- observations(make(foreign), resolve = TRUE)
    expect_identical(resolved$subject.age, c(20, 60))
  }
  expect_error(make(c(100000.5, 200000)), class = "fmridataset_error_relation")
  expect_invisible(validate_entity_registry(entity_registry(subject = subjects)))
})
