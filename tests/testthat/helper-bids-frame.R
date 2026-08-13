.make_bids_frame_fixture <- function(root = tempfile("bids-frame-")) {
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  func_dir <- file.path(root, "derivatives", "fmriprep", "sub-01", "func")
  dir.create(func_dir, recursive = TRUE, showWarnings = FALSE)
  raw_func_dir <- file.path(root, "sub-01", "func")
  dir.create(raw_func_dir, recursive = TRUE, showWarnings = FALSE)

  writeLines(
    '{"Name":"BIDS frame fixture","BIDSVersion":"1.10.0"}',
    file.path(root, "dataset_description.json")
  )
  utils::write.table(
    data.frame(participant_id = "sub-01"),
    file.path(root, "participants.tsv"),
    sep = "\t", row.names = FALSE, quote = FALSE
  )
  writeLines(
    paste0(
      '{"Name":"fMRIPrep fixture","BIDSVersion":"1.10.0",',
      '"DatasetType":"derivative","GeneratedBy":[{"Name":"fMRIPrep"}]}'
    ),
    file.path(root, "derivatives", "fmriprep", "dataset_description.json")
  )

  bold <- file.path(func_dir, sprintf(
    "sub-01_task-memory_run-%02d_space-MNI152NLin6Asym_desc-preproc_bold.nii",
    1:2
  ))
  masks <- file.path(func_dir, sprintf(
    "sub-01_task-memory_run-%02d_space-MNI152NLin6Asym_desc-brain_mask.nii",
    1:2
  ))

  spatial <- c(2L, 2L, 2L)
  for (i in seq_along(bold)) {
    values <- array(
      seq_len(prod(c(spatial, 3L))) + 100L * (i - 1L),
      c(spatial, 3L)
    )
    neuroim2::write_vec(
      neuroim2::NeuroVec(values, neuroim2::NeuroSpace(c(spatial, 3L))),
      bold[[i]]
    )
    writeLines(
      '{"RepetitionTime":2}',
      sub("\\.nii$", ".json", bold[[i]])
    )
  }

  mask_values <- list(
    array(c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE), spatial),
    array(c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE), spatial)
  )
  for (i in seq_along(masks)) {
    neuroim2::write_vol(
      neuroim2::LogicalNeuroVol(mask_values[[i]], neuroim2::NeuroSpace(spatial)),
      masks[[i]]
    )
  }

  entities <- function(paths) {
    base <- basename(paths)
    capture <- function(pattern) {
      value <- sub(pattern, "\\1", base)
      value[value == base] <- NA_character_
      value
    }
    tibble::tibble(
      .path = paths,
      subid = capture(".*sub-([A-Za-z0-9]+).*"),
      session = NA_character_,
      task = capture(".*task-([A-Za-z0-9]+).*"),
      run = as.integer(capture(".*run-([0-9]+).*")),
      echo = NA_integer_,
      space = capture(".*space-([A-Za-z0-9]+).*"),
      res = NA_character_,
      label = capture(".*label-([A-Za-z0-9]+).*"),
      desc = capture(".*desc-([A-Za-z0-9]+).*"),
      kind = ifelse(grepl("_bold\\.nii", base), "bold", "mask")
    )
  }

  event_paths <- file.path(
    raw_func_dir,
    sprintf("sub-01_task-memory_run-%02d_events.tsv", 1:2)
  )
  for (i in seq_along(event_paths)) {
    utils::write.table(
      data.frame(
        onset = 2 * (i - 1L),
        duration = 1,
        trial_type = c("old", "new")[[i]]
      ),
      event_paths[[i]],
      sep = "\t", row.names = FALSE, quote = FALSE
    )
  }

  events <- tibble::tibble(
    .subid = c("01", "01"),
    .session = c(NA_character_, NA_character_),
    .task = c("memory", "memory"),
    .run = c("01", "02"),
    file = event_paths,
    onset = c(0, 2),
    duration = c(1, 1),
    trial_type = c("old", "new")
  )

  list(
    root = root,
    bold = bold,
    masks = masks,
    entities = entities,
    events = events,
    support = c(2L, 3L, 4L)
  )
}

.with_bids_frame_bindings <- function(fixture, code, paths = fixture$bold,
                                      masks = fixture$masks,
                                      events = fixture$events,
                                      entities = fixture$entities,
                                      discover_masks = NULL) {
  if (is.null(discover_masks)) {
    discover_masks <- function(project, subject, session, space) masks
  }
  testthat::with_mocked_bindings(
    .bids_require = function() invisible("0.5.0"),
    .bids_open_project = function(path, derivative) {
      structure(list(path = path, derivative = derivative), class = "mock_bids_project")
    },
    .bids_discover_bold = function(project, subject, task, session, run, space) paths,
    .bids_parse_entities = entities,
    .bids_n_volumes = function(paths) stats::setNames(rep.int(3L, length(paths)), paths),
    .bids_infer_tr = function(path) 2,
    .bids_discover_masks = discover_masks,
    .bids_read_events = function(project, subject, task, session, run) events,
    .package = "fmridataset",
    code
  )
}
