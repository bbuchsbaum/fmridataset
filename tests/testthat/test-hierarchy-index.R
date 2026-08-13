.make_hierarchy_frame <- function(instrument = FALSE, missing_run = FALSE,
                                  ambiguous = FALSE) {
  subject <- entity_frame(
    tibble::tibble(
      subject_id = c("sub-02", "sub-01"),
      age = c(71, 63)
    ),
    key = "subject_id"
  )
  session <- entity_frame(
    tibble::tibble(
      session_id = c("ses-01b", "ses-02a", "ses-01a"),
      subject_id = c("sub-01", "sub-02", "sub-01")
    ),
    key = "session_id"
  )
  run <- entity_frame(
    tibble::tibble(
      run_id = c("run-02a", "run-01b", "run-01a"),
      session_id = c("ses-02a", "ses-01b", "ses-01a")
    ),
    key = "run_id"
  )
  stimulus <- entity_frame(
    tibble::tibble(stimulus_id = c("stim-1", "stim-2")),
    key = "stimulus_id"
  )
  run_id <- c("run-01a", "run-01b", "run-02a", "run-01a", "run-02a")
  if (missing_run) run_id[[2L]] <- NA_character_
  observations <- tibble::tibble(
    .obs_id = paste0("obs-", seq_along(run_id)),
    run_id = run_id,
    stimulus_id = c("stim-1", "stim-2", "stim-1", "stim-2", "stim-1")
  )
  source <- memory_source(matrix(seq_len(15L), nrow = 5L, ncol = 3L))
  if (instrument) source <- counting_source(source)
  relation_values <- list(
    observation_run = key_relation(
      "run_id",
      target = "run", allow_missing = missing_run
    ),
    run_session = key_relation(
      "session_id",
      source = "run", target = "session"
    ),
    session_subject = key_relation(
      "subject_id",
      source = "session", target = "subject"
    ),
    observation_stimulus = key_relation(
      "stimulus_id",
      target = "stimulus"
    )
  )
  if (ambiguous) {
    relation_values$observation_run_alias <- key_relation(
      "run_id",
      target = "run", allow_missing = missing_run
    )
  }
  fmri_frame(
    assays = list(beta = source),
    observations = observations,
    entities = list(
      subject = subject,
      session = session,
      run = run,
      stimulus = stimulus
    ),
    relations = relation_values
  )
}

test_that("hierarchy indices resolve a strict root-to-leaf containment path", {
  frame <- .make_hierarchy_frame(instrument = TRUE)
  index <- hierarchy_index(frame, c("subject", "session", "run"))

  expect_s3_class(index, "fmri_hierarchy_index")
  expect_identical(hierarchy_levels(index), c("subject", "session", "run"))
  expect_identical(
    hierarchy_relations(index),
    c(
      subject = "session_subject",
      session = "run_session",
      run = "observation_run"
    )
  )
  expect_identical(
    hierarchy_ids(index),
    tibble::tibble(
      .obs_id = paste0("obs-", 1:5),
      subject = c("sub-01", "sub-01", "sub-02", "sub-01", "sub-02"),
      session = c("ses-01a", "ses-01b", "ses-02a", "ses-01a", "ses-02a"),
      run = c("run-01a", "run-01b", "run-02a", "run-01a", "run-02a")
    )
  )
  expect_identical(
    hierarchy_groups(index),
    tibble::tibble(
      .obs_id = paste0("obs-", 1:5),
      subject = c(2L, 2L, 1L, 2L, 1L),
      session = c(3L, 1L, 2L, 3L, 2L),
      run = c(3L, 2L, 1L, 3L, 1L)
    )
  )
  expect_true(all(hierarchy_complete(index)))
  expect_match(hierarchy_digest(index), "^[0-9a-f]{64}$")
  expect_false(contains_runtime_state(index))
  expect_equal(source_counts(assay(frame)$source)$bytes, 0)
})

test_that("hierarchy group codes are stable across lazy views and reordering", {
  frame <- .make_hierarchy_frame()
  full <- hierarchy_index(frame, c("subject", "session", "run"))
  view <- frame[c(5L, 2L, 1L), ]
  selected <- hierarchy_index(view, c("subject", "session", "run"))

  expect_identical(
    hierarchy_ids(selected),
    hierarchy_ids(full)[c(5L, 2L, 1L), ]
  )
  expect_identical(
    hierarchy_groups(selected),
    hierarchy_groups(full)[c(5L, 2L, 1L), ]
  )
  expect_identical(observation_ids(view), hierarchy_ids(selected)$.obs_id)
})

test_that("hierarchy paths exclude crossed relations and reject invalid levels", {
  frame <- .make_hierarchy_frame()

  expect_error(
    hierarchy_index(frame, c("subject", "stimulus")),
    "strict containment path",
    class = "fmridataset_error_hierarchy"
  )
  expect_error(
    hierarchy_index(frame, c("subject", "subject")),
    "unique",
    class = "fmridataset_error_hierarchy"
  )
  expect_error(
    hierarchy_index(frame, "unknown"),
    "Unknown hierarchy level",
    class = "fmridataset_error_hierarchy"
  )
  expect_error(
    hierarchy_index(frame, character()),
    "at least one",
    class = "fmridataset_error_hierarchy"
  )
})

test_that("explicit relation selection resolves ambiguity and validates edges", {
  frame <- .make_hierarchy_frame(ambiguous = TRUE)

  expect_error(
    hierarchy_index(frame, c("subject", "session", "run")),
    "multiple key relations",
    class = "fmridataset_error_hierarchy"
  )
  selected <- hierarchy_index(
    frame,
    c("subject", "session", "run"),
    relations = c(
      run = "observation_run_alias",
      session = "run_session",
      subject = "session_subject"
    )
  )
  expect_identical(hierarchy_relations(selected)[["run"]], "observation_run_alias")
  expect_error(
    hierarchy_index(
      frame,
      c("subject", "session", "run"),
      relations = c(
        run = "observation_stimulus",
        session = "run_session",
        subject = "session_subject"
      )
    ),
    "does not connect",
    class = "fmridataset_error_hierarchy"
  )
  expect_error(
    hierarchy_index(
      frame,
      c("subject", "session", "run"),
      relations = c(run = "observation_run")
    ),
    "exactly once",
    class = "fmridataset_error_hierarchy"
  )
})

test_that("permitted missing descendants propagate through the ancestry", {
  frame <- .make_hierarchy_frame(missing_run = TRUE)
  index <- hierarchy_index(frame, c("subject", "session", "run"))

  expect_false(hierarchy_complete(index)[[2L]])
  expect_true(all(hierarchy_complete(index)[-2L]))
  expect_true(all(is.na(hierarchy_ids(index)[2L, c("subject", "session", "run")])))
  expect_true(all(is.na(hierarchy_groups(index)[2L, c("subject", "session", "run")])))
})

test_that("hierarchy indices serialize deterministically and print compactly", {
  index <- hierarchy_index(
    .make_hierarchy_frame(),
    c("subject", "session", "run")
  )
  restored <- unserialize(serialize(index, NULL))

  expect_identical(hierarchy_digest(restored), hierarchy_digest(index))
  expect_output(print(index), "5 observations")
  expect_output(print(index), "subject > session > run")
  expect_error(hierarchy_ids(list()), class = "fmridataset_error_hierarchy")

  malformed <- index
  malformed$groups$run[[1L]] <- 0L
  expect_error(
    hierarchy_groups(malformed),
    class = "fmridataset_error_hierarchy"
  )
})

test_that("single-level and empty-view hierarchies retain two-dimensional tables", {
  frame <- .make_hierarchy_frame()
  run_only <- hierarchy_index(frame, "run")
  empty <- hierarchy_index(frame[integer(), ], c("subject", "session", "run"))

  expect_identical(names(hierarchy_ids(run_only)), c(".obs_id", "run"))
  expect_identical(hierarchy_relations(run_only), c(run = "observation_run"))
  expect_identical(dim(hierarchy_ids(empty)), c(0L, 4L))
  expect_identical(dim(hierarchy_groups(empty)), c(0L, 4L))
  expect_identical(hierarchy_complete(empty), logical())
})
