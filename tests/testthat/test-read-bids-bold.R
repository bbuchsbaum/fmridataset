test_that("read_bids_bold constructs a deterministic lazy frame", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  frame <- .with_bids_frame_bindings(fixture, {
    read_bids_bold(
      fixture$root,
      subject = "sub-01",
      task = "memory",
      space = "MNI152NLin6Asym"
    )
  })

  expect_s3_class(frame, "fmri_frame")
  expect_identical(dim(frame), c(6L, 3L))
  expect_identical(
    observation_ids(frame),
    unlist(lapply(sub("\\.nii$", "", fs::path_rel(fixture$bold, fixture$root)),
                  function(scan) sprintf("%s::volume-%06d", scan, 0:2)),
           use.names = FALSE)
  )
  expect_identical(features(frame)$.feature_id, paste0("voxel-", fixture$support))
  expect_identical(entity_names(entities(frame)), c("subject", "run"))
  expect_identical(entity_ids(entity(entities(frame), "subject")), "sub-01")
  expect_identical(nrow(entity_data(entity(entities(frame), "run"))), 2L)
  expect_identical(names(relations(frame)), c("observation_run", "run_subject"))
  expect_s3_class(frame$tables$events, "fmri_event_table")
  expect_identical(event_data(frame$tables$events)$scan_id,
                   entity_ids(entity(entities(frame), "run")))
  expect_false(any(grepl(fixture$root, event_data(frame$tables$events)$file,
                         fixed = TRUE)))
  expect_false(any(grepl(fixture$root, space(frame)$metadata$mask_files,
                         fixed = TRUE)))
  expect_false(contains_runtime_state(frame))
})

test_that("BIDS import is invariant to discovery order and dataset relocation", {
  first <- .make_bids_frame_fixture()
  second <- .make_bids_frame_fixture()
  on.exit(unlink(c(first$root, second$root), recursive = TRUE), add = TRUE)

  frame_a <- .with_bids_frame_bindings(first, {
    read_bids_bold(first$root, subject = "01", task = "memory",
                   space = "MNI152NLin6Asym", events = FALSE)
  }, paths = rev(first$bold), masks = rev(first$masks))
  frame_b <- .with_bids_frame_bindings(second, {
    read_bids_bold(second$root, subject = "01", task = "memory",
                   space = "MNI152NLin6Asym", events = FALSE)
  })

  expect_identical(observation_ids(frame_a), observation_ids(frame_b))
  expect_identical(observations(frame_a)$scan_id, observations(frame_b)$scan_id)
  expect_identical(space_digest(space(frame_a)), space_digest(space(frame_b)))
  expect_identical(frame_a$provenance, frame_b$provenance)
})

test_that("metadata inspection does not read BOLD values", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)
  reads <- 0L

  frame <- testthat::with_mocked_bindings(
    .nifti_read_vec = function(...) {
      reads <<- reads + 1L
      stop("unexpected BOLD read")
    },
    .package = "fmridataset",
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", task = "memory",
                     space = "MNI152NLin6Asym", events = FALSE)
    })
  )

  expect_identical(reads, 0L)
  expect_silent(dim(frame))
  expect_silent(observations(frame))
  expect_output(print(frame), "<fmri_frame>")
  expect_identical(reads, 0L)
})

test_that("BIDS import reads only selected values on collection", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  frame <- .with_bids_frame_bindings(fixture, {
    read_bids_bold(fixture$root, subject = "01", task = "memory",
                   space = "MNI152NLin6Asym", events = FALSE)
  })
  selected <- collect_assay(frame[c(1L, 4L), c(1L, 3L)])

  expect_identical(dim(selected), c(2L, 2L))
  reference <- source_read(assay(frame)$source, c(1L, 4L), c(1L, 3L))
  expect_equal(selected, reference, tolerance = 0)
})

test_that("BIDS import rejects ambiguous or unsupported domains", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  mixed_space_entities <- function(paths) {
    value <- fixture$entities(paths)
    value$space[seq_len(min(1L, nrow(value)))] <- "OtherSpace"
    value
  }
  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", events = FALSE)
    }, entities = mixed_space_entities),
    "multiple spaces",
    class = "fmridataset_error_bids_import"
  )

  echo_entities <- function(paths) {
    value <- fixture$entities(paths)
    value$echo <- 1L
    value
  }
  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", events = FALSE)
    }, entities = echo_entities),
    "multi-echo",
    class = "fmridataset_error_bids_import"
  )

  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", mask = "union", events = FALSE)
    }),
    "mask",
    class = "fmridataset_error_bids_import"
  )

  native_entities <- function(paths) {
    value <- fixture$entities(paths)
    value$space <- NA_character_
    value
  }
  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01",
                     space = "MNI152NLin6Asym", events = FALSE)
    }, entities = native_entities),
    "no space entity",
    class = "fmridataset_error_bids_import"
  )

  mixed_resolution_entities <- function(paths) {
    value <- fixture$entities(paths)
    value$res <- c("1", "2")[seq_len(nrow(value))]
    value
  }
  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", events = FALSE)
    }, entities = mixed_resolution_entities),
    "multiple resolutions",
    class = "fmridataset_error_bids_import"
  )
})

test_that("BIDS importer reports the development bidser dependency", {
  skip_if(packageVersion("bidser") >= "0.5.0")
  expect_error(
    read_bids_bold(tempdir(), subject = "01"),
    "bidser.*0.5.0",
    class = "fmridataset_error_bids_import"
  )
})

test_that("BIDS selectors are exact even when labels contain regex metacharacters", {
  expect_identical(.bids_exact_pattern("task.a+b", "task"), "^task\\.a\\+b$")
  expect_true(grepl(.bids_exact_pattern("task.a+b", "task"), "task.a+b"))
  expect_false(grepl(.bids_exact_pattern("task.a+b", "task"), "taskXaab"))
})

test_that("BIDS project discovery disables on-disk indexing", {
  seen <- NULL
  testthat::with_mocked_bindings(
    .bids_export = function(name) {
      expect_identical(name, "bids_project")
      function(path, fmriprep, index) {
        seen <<- list(path = path, fmriprep = fmriprep, index = index)
        structure(list(), class = "bids_project")
      }
    },
    .package = "fmridataset",
    .bids_open_project("/tmp/study", "fmriprep")
  )
  expect_identical(seen$index, "none")
  expect_true(seen$fmriprep)
})

test_that("native-space mask discovery does not request a named template", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)
  seen_space <- "unset"
  native_entities <- function(paths) {
    value <- fixture$entities(paths)
    value$space <- NA_character_
    value
  }
  frame <- .with_bids_frame_bindings(fixture, {
    read_bids_bold(fixture$root, subject = "01", events = FALSE)
  }, entities = native_entities, discover_masks = function(project, subject, session, space) {
      seen_space <<- space
      fixture$masks
  })
  expect_null(seen_space)
  expect_identical(frame$metadata$space, "native")
})

test_that("intersection masking ignores non-brain tissue masks", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)
  tissue_path <- sub("desc-brain_mask", "label-GM_mask", fixture$masks[[1L]])
  expect_true(file.copy(fixture$masks[[1L]], tissue_path))
  candidates <- c(fixture$masks, tissue_path)

  frame <- .with_bids_frame_bindings(fixture, {
    read_bids_bold(fixture$root, subject = "01", task = "memory",
                   space = "MNI152NLin6Asym", events = FALSE)
  }, masks = candidates)

  expect_identical(feature_ids(frame), paste0("voxel-", fixture$support))
})

test_that("explicit volume spaces cannot contradict selected BIDS space", {
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)
  header <- neuroim2::read_header(fixture$bold[[1L]])
  affine <- fmridataset:::.nifti_header_affine(header)
  unlabeled <- volume_space(c(2, 2, 2), affine = affine, support = fixture$support)
  labeled <- volume_space(c(2, 2, 2), affine = affine, support = fixture$support,
                          template = "OtherSpace")

  frame <- .with_bids_frame_bindings(fixture, {
    read_bids_bold(fixture$root, subject = "01", task = "memory",
                   space = "MNI152NLin6Asym", mask = unlabeled, events = FALSE)
  })
  expect_identical(space(frame)$template, "MNI152NLin6Asym")

  expect_error(
    .with_bids_frame_bindings(fixture, {
      read_bids_bold(fixture$root, subject = "01", task = "memory",
                     space = "MNI152NLin6Asym", mask = labeled, events = FALSE)
    }),
    "template disagrees",
    class = "fmridataset_error_bids_import"
  )
})

test_that("read_bids_bold integrates with the released bidser contract", {
  skip_if_not_installed("bidser", minimum_version = "0.5.0")
  fixture <- .make_bids_frame_fixture()
  on.exit(unlink(fixture$root, recursive = TRUE), add = TRUE)

  frame <- read_bids_bold(
    fixture$root,
    subject = "01",
    task = "memory",
    space = "MNI152NLin6Asym"
  )

  expect_identical(dim(frame), c(6L, 3L))
  expect_identical(observation_ids(frame)[c(1L, 4L)], sprintf(
    "%s::volume-000000",
    sub("\\.nii$", "", fs::path_rel(fixture$bold, fixture$root))
  ))
  expect_identical(nrow(event_data(frame$tables$events)), 2L)
})
