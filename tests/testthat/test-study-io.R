.skip_without_study_store <- function() {
  skip_if_not(
    requireNamespace("fmristore", quietly = TRUE) &&
      "write_frame_h5" %in% getNamespaceExports("fmristore") &&
      requireNamespace("hdf5r", quietly = TRUE),
    "fmristore with frame support is not installed"
  )
}

.study_io_fixture <- function() {
  set.seed(11)
  subjects <- entity_frame(
    data.frame(subject_id = c("sub-01", "sub-02"), group = c("patient", NA), age = c(34, 29)),
    key = "subject_id", entity_type = "subject"
  )
  embedding <- axis_block(
    matrix(as.double(1:8), 4),
    components = data.frame(.component_id = c("d1", "d2"))
  )
  stimuli <- entity_frame(
    data.frame(stimulus_id = c("face-1", "face-2", "house-1", "house-2")),
    key = "stimulus_id", entity_type = "stimulus",
    blocks = list(embedding = embedding)
  )
  regions <- index_space(3, ids = c("FFA", "PPA", "V1"))
  trials <- data.frame(
    .obs_id = sprintf("%s-trial-%d", rep(c("sub-01", "sub-02"), each = 4), 1:4),
    subject_id = rep(c("sub-01", "sub-02"), each = 4),
    stimulus_id = rep(c("face-1", "house-1", "face-2", "house-2"), 2),
    onset = rep(c(0, 8, 16, 24), 2)
  )
  betas <- fmri_frame(
    assays = list(beta = matrix(round(rnorm(24), 2), 8, 3)),
    observations = trials, space = regions,
    entities = list(subject = subjects, stimulus = stimuli),
    relations = list(
      trial_subject = key_relation("subject_id", target = "subject"),
      trial_stimulus = key_relation("stimulus_id", target = "stimulus")
    )
  )
  volumes <- data.frame(
    .obs_id = sprintf("%s-vol-%02d", rep(c("sub-01", "sub-02"), each = 16), 1:16),
    subject_id = rep(c("sub-01", "sub-02"), each = 16),
    run_id = rep(c("r1", "r2"), each = 16), TR = 2
  )
  bold <- fmri_frame(
    assays = list(bold = matrix(round(rnorm(96), 2), 32, 3)),
    observations = volumes, space = regions,
    entities = list(subject = subjects),
    relations = list(volume_subject = key_relation("subject_id", target = "subject"))
  )
  native <- lapply(c("sub-01", "sub-02"), function(subject) {
    fmri_frame(
      assays = list(bold = matrix(round(rnorm(8), 2), 4, 2)),
      observations = data.frame(.obs_id = sprintf("%s-v%d", subject, 1:4), run_id = "r1", TR = 2),
      space = volume_space(c(2, 1, 1), template = paste0(subject, "-native"))
    )
  })
  names(native) <- c("sub-01", "sub-02")
  link <- frame_link(
    "betas", "bold", type = "correspondence",
    map = data.frame(
      .source_id = trials$.obs_id,
      .target_id = sprintf("%s-vol-%02d", trials$subject_id, trials$onset / 2 + 1)
    )
  )
  fmri_study(
    list(native = fmri_collection(native), betas = betas, bold = bold),
    entities = list(subject = subjects, stimulus = stimuli),
    links = list(trial_onset = link),
    metadata = list(study = "toy")
  )
}

test_that("a study round-trips through write_study() and open_study()", {
  .skip_without_study_store()
  study <- .study_io_fixture()
  path <- file.path(withr::local_tempdir(), "toy-study")

  expect_identical(write_study(study, path), normalizePath(path, winslash = "/"))
  reopened <- open_study(path)

  expect_s3_class(reopened, "fmri_study")
  expect_identical(
    fds_study_manifest_digest(fds_study_manifest(reopened)),
    fds_study_manifest_digest(fds_study_manifest(study))
  )
  expect_identical(study_ids(reopened), study_ids(study))
  expect_equal(
    collect_assay(study_frame(reopened, "betas")),
    collect_assay(study_frame(study, "betas"))
  )
  expect_s3_class(assay(study_frame(reopened, "bold"))$source, "h5_array_source")
  expect_identical(collection_ids(study_frame(reopened, "native")), c("sub-01", "sub-02"))
  expect_identical(study_link(reopened, "trial_onset")$map, study_link(study, "trial_onset")$map)
  expect_identical(
    source_read(entity_blocks(entity(reopened, "stimulus"))$embedding$data),
    matrix(as.double(1:8), 4)
  )
  # Reading the reopened study's entity tables needs no frame I/O.
  expect_identical(entity_data(entity(reopened, "subject")), entity_data(entity(study, "subject")))
})

test_that("a saved study is a BIDS derivative dataset", {
  .skip_without_study_store()
  path <- file.path(withr::local_tempdir(), "toy-study")
  write_study(
    .study_io_fixture(), path,
    name = "Toy analysis",
    source_datasets = list(list(URL = "https://example.org/raw", Version = "1.0")),
    dataset_links = list(raw = "../raw")
  )

  expect_setequal(
    list.files(path, all.files = TRUE, no.. = TRUE),
    c(".bidsignore", "arrays.h5", "dataset_description.json", "frames",
      "participants.tsv", "storage.json", "study.json")
  )
  description <- jsonlite::read_json(file.path(path, "dataset_description.json"))
  expect_identical(description$Name, "Toy analysis")
  expect_identical(description$DatasetType, "derivative")
  expect_identical(description$GeneratedBy[[1]]$Name, "fmridataset")
  expect_identical(description$SourceDatasets[[1]]$URL, "https://example.org/raw")
  expect_identical(description$DatasetLinks, list(raw = "../raw"))
  expect_identical(
    readLines(file.path(path, ".bidsignore")),
    c("study.json", "storage.json", "frames/", "arrays.h5")
  )
  participants <- utils::read.delim(file.path(path, "participants.tsv"), na.strings = "n/a")
  expect_identical(participants$participant_id, c("sub-01", "sub-02"))
  expect_identical(names(participants), c("participant_id", "group", "age"))
  expect_true(is.na(participants$group[[2]]))
  expect_match(readLines(file.path(path, "study.json"))[[2]], "org.fmridataset.fds-json/v1", fixed = TRUE)
})

test_that("participants.tsv is skipped when subject IDs are not BIDS labels", {
  .skip_without_study_store()
  subjects <- entity_frame(
    data.frame(subject_id = c("pt 1", "pt 2")), key = "subject_id", entity_type = "subject"
  )
  frame <- fmri_frame(
    list(b = matrix(0, 2, 1)),
    data.frame(.obs_id = c("a", "b"), subject_id = c("pt 1", "pt 2")),
    space = index_space(1, ids = "f"),
    entities = list(subject = subjects),
    relations = list(r = key_relation("subject_id", target = "subject"))
  )
  path <- file.path(withr::local_tempdir(), "odd-labels")
  write_study(fmri_study(list(main = frame), entities = list(subject = subjects)), path)
  expect_false(file.exists(file.path(path, "participants.tsv")))
  expect_s3_class(open_study(path), "fmri_study")
})

test_that("write_study() refuses before writing and leaves nothing behind on failure", {
  .skip_without_study_store()
  dir <- withr::local_tempdir()
  path <- file.path(dir, "study")

  expect_error(write_study(.study_io_fixture()$frames$betas, path), "saves an fmri_study")
  expect_error(
    write_study(.study_io_fixture(), path, dataset_links = list("../raw")),
    "dataset_links must be a named list"
  )
  scratch <- fmri_frame(
    list(b = matrix(0, 2, 1)),
    axis_frame(data.frame(x = 1:2), id_policy = "ephemeral"),
    space = index_space(1, ids = "f")
  )
  expect_error(write_study(fmri_study(list(main = scratch)), path), class = "fmridataset_error_identity")
  expect_length(list.files(dir, all.files = TRUE, no.. = TRUE), 0L)

  testthat::local_mocked_bindings(
    write_frame = function(x, path, ...) stop("disk full"),
    .package = "fmridataset"
  )
  expect_error(write_study(.study_io_fixture(), path), "disk full")
  expect_length(list.files(dir, all.files = TRUE, no.. = TRUE), 0L)
})

test_that("an existing study is replaced only on request", {
  .skip_without_study_store()
  study <- .study_io_fixture()
  path <- file.path(withr::local_tempdir(), "study")
  write_study(study, path)
  expect_error(write_study(study, path), "overwrite = TRUE", class = "fmridataset_error_backend_io")
  expect_identical(write_study(study, path, overwrite = TRUE), normalizePath(path, winslash = "/"))
  expect_length(list.files(dirname(path), all.files = TRUE, no.. = TRUE), 1L)
  expect_s3_class(open_study(path), "fmri_study")
})

test_that("open_study() detects replaced frames, edited manifests, and missing files", {
  .skip_without_study_store()
  study <- .study_io_fixture()
  dir <- withr::local_tempdir()
  path <- file.path(dir, "study")
  write_study(study, path)
  storage <- fmridataset:::.fds_json_read(file.path(path, "storage.json"))$value

  # Swap two frame files: each is a valid frame, but not the one recorded.
  betas <- file.path(path, storage$representations$betas$file)
  bold <- file.path(path, storage$representations$bold$file)
  file.copy(betas, file.path(dir, "betas.h5"))
  file.copy(bold, betas, overwrite = TRUE)
  expect_error(open_study(path), "not the frame this study was saved with", class = "fmridataset_error_source_stale")
  file.copy(file.path(dir, "betas.h5"), betas, overwrite = TRUE)
  expect_s3_class(open_study(path), "fmri_study")

  # Edit the study's metadata by hand.
  study_json <- file.path(path, "study.json")
  original <- readLines(study_json)
  edited <- sub('"toy"', '"edited"', original, fixed = TRUE)
  expect_false(identical(edited, original))
  writeLines(edited, study_json)
  expect_error(open_study(path), "does not match the digest", class = "fmridataset_error_source_stale")
  writeLines(original, study_json)

  unlink(bold)
  expect_error(open_study(path), "missing the frame file", class = "fmridataset_error_backend_io")
  expect_error(open_study(file.path(dir, "nowhere")), "not a saved study", class = "fmridataset_error_backend_io")
})
