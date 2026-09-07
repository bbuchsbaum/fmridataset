.bids_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_bids_import", ...)
}

.bids_required_exports <- c(
  "bids_project", "preproc_scans", "bids_entities", "n_volumes",
  "infer_tr", "mask_files", "load_all_events"
)

.bids_require <- function() {
  minimum <- package_version("0.5.0")
  if (!requireNamespace("bidser", quietly = TRUE)) {
    .bids_abort(
      "read_bids_bold() requires the optional package 'bidser' (>= 0.5.0).",
      dependency = "bidser", minimum_version = as.character(minimum)
    )
  }
  installed <- utils::packageVersion("bidser")
  namespace <- asNamespace("bidser")
  missing <- .bids_required_exports[!vapply(
    .bids_required_exports,
    exists,
    logical(1),
    envir = namespace,
    inherits = FALSE
  )]
  if (installed < minimum || length(missing)) {
    details <- if (length(missing)) {
      paste0(" Missing exports: ", paste(missing, collapse = ", "), ".")
    } else {
      ""
    }
    .bids_abort(
      paste0(
        "read_bids_bold() requires bidser >= 0.5.0; installed version is ",
        as.character(installed), ".", details
      ),
      dependency = "bidser", minimum_version = as.character(minimum),
      installed_version = as.character(installed), missing_exports = missing
    )
  }
  invisible(as.character(installed))
}

.bids_export <- function(name) getExportedValue("bidser", name)

.bids_open_project <- function(path, derivative) {
  .bids_export("bids_project")(
    path,
    fmriprep = identical(derivative, "fmriprep"),
    index = "none"
  )
}

.bids_exact_pattern <- function(value, field, required = FALSE) {
  if (is.null(value)) {
    if (required) .bids_abort(sprintf("%s must be supplied.", field), field = field)
    return(".*")
  }
  if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
    .bids_abort(sprintf("%s must be one non-empty string or NULL.", field), field = field)
  }
  value <- sub(paste0("^", field, "-"), "", value)
  chars <- strsplit(value, "", fixed = TRUE)[[1L]]
  special <- chars %in% strsplit("\\.^$|()[]{}*+?", "", fixed = TRUE)[[1L]]
  chars[special] <- paste0("\\", chars[special])
  escaped <- paste0(chars, collapse = "")
  paste0("^", escaped, "$")
}

.bids_discover_bold <- function(project, subject, task, session, run, space) {
  .bids_export("preproc_scans")(
    project,
    subid = .bids_exact_pattern(subject, "sub", required = TRUE),
    task = .bids_exact_pattern(task, "task"),
    session = .bids_exact_pattern(session, "ses"),
    run = .bids_exact_pattern(run, "run"),
    space = .bids_exact_pattern(space, "space"),
    modality = "bold",
    kind = "bold",
    full_path = TRUE
  )
}

.bids_parse_entities <- function(paths) .bids_export("bids_entities")(paths)
.bids_n_volumes <- function(paths) .bids_export("n_volumes")(paths)
.bids_infer_tr <- function(path) .bids_export("infer_tr")(path)

.bids_discover_masks <- function(project, subject, session, space) {
  .bids_export("mask_files")(
    project,
    subid = .bids_exact_pattern(subject, "sub", required = TRUE),
    session = .bids_exact_pattern(session, "ses"),
    space = .bids_exact_pattern(space, "space"),
    full_path = TRUE
  )
}

.bids_read_events <- function(project, subject, task, session, run) {
  suppressMessages(.bids_export("load_all_events")(
    project,
    subid = .bids_exact_pattern(subject, "sub", required = TRUE),
    task = .bids_exact_pattern(task, "task"),
    session = .bids_exact_pattern(session, "ses"),
    run = .bids_exact_pattern(run, "run"),
    full_path = TRUE
  ))
}

.bids_scalar <- function(value, field) {
  if (is.null(value)) {
    return(NULL)
  }
  if (length(value) != 1L || is.na(value) || !nzchar(as.character(value))) {
    .bids_abort(sprintf("%s must be one non-empty value or NULL.", field), field = field)
  }
  as.character(value)
}

.bids_entity_value <- function(data, name, default = NA_character_) {
  if (!name %in% names(data)) {
    return(rep(default, nrow(data)))
  }
  as.character(data[[name]])
}

.bids_relative_path <- function(paths, root) {
  root <- normalizePath(root, winslash = "/", mustWork = TRUE)
  existing <- file.exists(paths)
  absolute <- fs::path_abs(paths)
  absolute[existing] <- normalizePath(
    paths[existing],
    winslash = "/", mustWork = TRUE
  )
  values <- fs::path_rel(absolute, start = root)
  outside <- values == ".." | startsWith(values, paste0("..", .Platform$file.sep))
  if (any(outside)) {
    .bids_abort(
      "Discovered BIDS files must be inside the supplied BIDS root so stable relative IDs can be minted.",
      files = paths[outside]
    )
  }
  gsub("\\\\", "/", values)
}

.bids_domain_label <- function(value, missing = "native") {
  absent <- is.na(value) | !nzchar(value)
  present <- unique(value[!absent])
  if (all(absent)) {
    return(missing)
  }
  if (any(absent)) {
    return(c(present, missing))
  }
  present
}

.bids_bold_manifest <- function(project, root, subject, task = NULL,
                                session = NULL, run = NULL, space = NULL) {
  paths <- .bids_discover_bold(project, subject, task, session, run, space)
  if (is.null(paths) || !length(paths)) {
    .bids_abort("No matching fMRIPrep preprocessed BOLD files were found.")
  }
  paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
  parsed <- tibble::as_tibble(.bids_parse_entities(paths))
  if (nrow(parsed) != length(paths)) {
    .bids_abort("bidser returned entity metadata that is not aligned with the discovered scans.")
  }
  parsed$.path <- paths
  parsed$relative_path <- .bids_relative_path(paths, root)
  parsed$scan_id <- sub("\\.nii(\\.gz)?$", "", parsed$relative_path, ignore.case = TRUE)
  if (anyDuplicated(parsed$scan_id)) {
    .bids_abort("BOLD discovery produced duplicate stable scan IDs.", scan_ids = parsed$scan_id)
  }

  parsed$subject_id <- paste0("sub-", .bids_entity_value(parsed, "subid"))
  requested_subject <- paste0("sub-", sub("^sub-", "", subject))
  if (anyNA(parsed$subject_id) || any(parsed$subject_id != requested_subject)) {
    .bids_abort("BOLD discovery escaped the requested subject selector.")
  }
  session_value <- .bids_entity_value(parsed, "session")
  task_value <- .bids_entity_value(parsed, "task")
  run_value <- .bids_entity_value(parsed, "run")
  parsed$session_id <- ifelse(
    is.na(session_value) | !nzchar(session_value),
    "ses-none",
    paste0("ses-", session_value)
  )
  parsed$task_id <- ifelse(
    is.na(task_value) | !nzchar(task_value),
    "task-none",
    paste0("task-", task_value)
  )
  parsed$run_id <- ifelse(
    is.na(run_value) | !nzchar(run_value),
    "run-none",
    paste0("run-", run_value)
  )

  spaces <- .bids_domain_label(.bids_entity_value(parsed, "space"))
  if (length(spaces) != 1L) {
    .bids_abort(
      paste0(
        "Selection contains multiple spaces: ", paste(spaces, collapse = ", "),
        ". Supply space = to select one."
      ),
      spaces = spaces
    )
  }
  observed_space <- .bids_entity_value(parsed, "space")
  observed_space <- observed_space[!is.na(observed_space) & nzchar(observed_space)]
  if (!is.null(space) && !length(observed_space)) {
    .bids_abort(
      "The selected BOLD files have no space entity and cannot satisfy an explicit space selector.",
      requested_space = space
    )
  }
  echoes <- .bids_entity_value(parsed, "echo")
  if (any(!is.na(echoes) & nzchar(echoes))) {
    .bids_abort("multi-echo BOLD import is not supported; select or combine echoes explicitly.")
  }
  resolutions <- .bids_domain_label(.bids_entity_value(parsed, "res"), missing = "unspecified")
  if (length(resolutions) != 1L) {
    .bids_abort(
      paste0(
        "Selection contains multiple resolutions: ",
        paste(resolutions, collapse = ", "),
        ". Select one resolution before import."
      ),
      resolutions = resolutions
    )
  }

  parsed$n_volume <- as.integer(unname(.bids_n_volumes(paths)))
  if (length(parsed$n_volume) != nrow(parsed) || anyNA(parsed$n_volume) ||
    any(parsed$n_volume <= 0L)) {
    .bids_abort("Every selected BOLD file must have a positive volume count.")
  }
  parsed$TR <- vapply(paths, function(path) as.numeric(.bids_infer_tr(path))[[1L]], numeric(1))
  if (anyNA(parsed$TR) || any(!is.finite(parsed$TR)) || any(parsed$TR <= 0)) {
    .bids_abort("Every selected BOLD file must have a finite positive TR.")
  }

  parsed <- parsed[order(parsed$scan_id), , drop = FALSE]
  attr(parsed, "selected_space") <- spaces[[1L]]
  parsed
}

.bids_entity_match <- function(scan, candidates, keys) {
  keep <- rep(TRUE, nrow(candidates))
  score <- integer(nrow(candidates))
  for (key in keys) {
    scan_value <- .bids_entity_value(scan, key)[[1L]]
    candidate_value <- .bids_entity_value(candidates, key)
    if (key %in% c("run", "echo")) {
      scan_value <- if (!is.na(scan_value) && grepl("^[0-9]+$", scan_value)) {
        as.character(as.integer(scan_value))
      } else {
        scan_value
      }
      numeric_candidate <- !is.na(candidate_value) & grepl("^[0-9]+$", candidate_value)
      candidate_value[numeric_candidate] <- as.character(
        as.integer(candidate_value[numeric_candidate])
      )
    }
    comparable <- !is.na(scan_value) & nzchar(scan_value) &
      !is.na(candidate_value) & nzchar(candidate_value)
    keep <- keep & (!comparable | candidate_value == scan_value)
    score <- score + as.integer(comparable & candidate_value == scan_value)
  }
  list(keep = keep, score = score)
}

.bids_match_run_masks <- function(manifest, candidates) {
  keys <- c("subid", "session", "task", "run", "space", "res", "acq", "dir")
  vapply(seq_len(nrow(manifest)), function(i) {
    matched <- .bids_entity_match(manifest[i, , drop = FALSE], candidates, keys)
    positions <- which(matched$keep)
    if (!length(positions)) {
      .bids_abort("No compatible fMRIPrep mask was found for a selected BOLD run.",
        scan_id = manifest$scan_id[[i]]
      )
    }
    best <- positions[matched$score[positions] == max(matched$score[positions])]
    if (length(best) != 1L) {
      .bids_abort(
        "Multiple equally specific masks match a selected BOLD run; supply mask = explicitly.",
        scan_id = manifest$scan_id[[i]], masks = candidates$.path[best]
      )
    }
    candidates$.path[[best]]
  }, character(1))
}

.bids_validate_mask_geometry <- function(paths, expected_dim, expected_affine) {
  headers <- lapply(paths, neuroim2::read_header)
  dims <- lapply(headers, function(header) as.integer(methods::slot(header, "dims")[1:3]))
  affines <- lapply(headers, .nifti_header_affine)
  compatible <- vapply(seq_along(paths), function(i) {
    identical(dims[[i]], expected_dim) &&
      isTRUE(all.equal(affines[[i]], expected_affine,
        tolerance = 1e-7,
        check.attributes = FALSE
      ))
  }, logical(1))
  if (any(!compatible)) {
    .bids_abort("One or more masks are not spatially compatible with the selected BOLD grid.",
      masks = paths[!compatible]
    )
  }
  invisible(TRUE)
}

.bids_resolve_space <- function(project, root, manifest, mask, subject, session,
                                selected_space) {
  bold_header <- neuroim2::read_header(manifest$.path[[1L]])
  expected_dim <- as.integer(methods::slot(bold_header, "dims")[1:3])
  expected_affine <- .nifti_header_affine(bold_header)

  if (inherits(mask, "volume_space")) {
    if (!identical(as.integer(mask$dim), expected_dim) ||
      !isTRUE(all.equal(mask$affine, expected_affine,
        tolerance = 1e-7,
        check.attributes = FALSE
      ))) {
      .bids_abort("The supplied volume_space mask is incompatible with the BOLD grid.")
    }
    if (!is.null(mask$template) && !identical(mask$template, selected_space)) {
      .bids_abort(
        "The supplied volume_space template disagrees with the selected BIDS space.",
        mask_template = mask$template, selected_space = selected_space
      )
    }
    resolved_space <- volume_space(
      dim = mask$dim,
      affine = mask$affine,
      support = mask$support,
      template = selected_space,
      units = mask$units,
      metadata = mask$metadata
    )
    return(list(space = resolved_space, mask_paths = character(), policy = "explicit-space"))
  }
  if (is.character(mask) && length(mask) == 1L && !is.na(mask) &&
    nzchar(mask) && !identical(mask, "intersection")) {
    if (!file.exists(mask)) {
      .bids_abort("The explicit mask path does not exist.", mask = mask)
    }
    path <- normalizePath(mask, winslash = "/", mustWork = TRUE)
    .bids_validate_mask_geometry(path, expected_dim, expected_affine)
    support <- which(as.logical(as.vector(suppressWarnings(neuroim2::read_vol(path)))))
    if (!length(support)) .bids_abort("The explicit mask contains no active features.")
    return(list(
      space = volume_space(expected_dim,
        affine = expected_affine, support = support,
        template = selected_space
      ),
      mask_paths = path,
      policy = "explicit-path"
    ))
  }
  if (!is.character(mask) || length(mask) != 1L || !identical(mask, "intersection")) {
    .bids_abort("mask must be 'intersection', an existing NIfTI path, or a volume_space.")
  }

  mask_space <- if (identical(selected_space, "native")) NULL else selected_space
  paths <- .bids_discover_masks(project, subject, session, mask_space)
  if (is.null(paths) || !length(paths)) {
    .bids_abort("No fMRIPrep masks were found; supply mask = explicitly.")
  }
  paths <- normalizePath(paths, winslash = "/", mustWork = TRUE)
  candidates <- tibble::as_tibble(.bids_parse_entities(paths))
  candidates$.path <- paths
  candidate_space <- .bids_entity_value(candidates, "space")
  if (identical(selected_space, "native")) {
    candidates <- candidates[is.na(candidate_space) | !nzchar(candidate_space), , drop = FALSE]
  }
  description <- tolower(.bids_entity_value(candidates, "desc"))
  kind <- tolower(.bids_entity_value(candidates, "kind"))
  label <- .bids_entity_value(candidates, "label")
  candidate_names <- basename(candidates$.path)
  no_description <- is.na(description) | !nzchar(description)
  is_brain <- description %in% c("brain", "brainmask") |
    kind %in% c("brainmask") |
    (grepl("_(brain_)?mask\\.nii(\\.gz)?$", candidate_names, ignore.case = TRUE) &
      no_description &
      (is.na(label) | !nzchar(label)))
  candidates <- candidates[is_brain, , drop = FALSE]
  if (!nrow(candidates)) {
    .bids_abort("No brain masks were found among the matching fMRIPrep masks; supply mask = explicitly.")
  }
  selected <- unique(.bids_match_run_masks(manifest, candidates))
  .bids_validate_mask_geometry(selected, expected_dim, expected_affine)
  values <- lapply(selected, function(path) {
    as.logical(as.vector(suppressWarnings(neuroim2::read_vol(path))))
  })
  support <- which(Reduce(`&`, values))
  if (!length(support)) .bids_abort("The selected run masks have an empty intersection.")
  relative <- .bids_relative_path(selected, root)
  list(
    space = volume_space(
      expected_dim,
      affine = expected_affine,
      support = support,
      template = selected_space,
      metadata = list(mask_policy = "intersection", mask_files = sort(relative))
    ),
    mask_paths = selected,
    policy = "intersection"
  )
}

.bids_expand_observations <- function(manifest) {
  values <- lapply(seq_len(nrow(manifest)), function(i) {
    index <- seq_len(manifest$n_volume[[i]])
    n <- length(index)
    data.frame(
      .obs_id = sprintf("%s::volume-%06d", manifest$scan_id[[i]], index - 1L),
      scan_id = rep(manifest$scan_id[[i]], n),
      subject_id = rep(manifest$subject_id[[i]], n),
      session_id = rep(manifest$session_id[[i]], n),
      task_id = rep(manifest$task_id[[i]], n),
      run_id = rep(manifest$run_id[[i]], n),
      volume_index = index,
      run_time = (index - 1L) * manifest$TR[[i]],
      TR = rep(manifest$TR[[i]], n),
      stringsAsFactors = FALSE
    )
  })
  tibble::as_tibble(do.call(rbind, values))
}

.bids_event_scan_ids <- function(events, manifest) {
  event_entities <- tibble::tibble(
    subid = .bids_entity_value(events, ".subid"),
    session = .bids_entity_value(events, ".session"),
    task = .bids_entity_value(events, ".task"),
    run = .bids_entity_value(events, ".run")
  )
  vapply(seq_len(nrow(events)), function(i) {
    matched <- .bids_entity_match(
      event_entities[i, , drop = FALSE], manifest,
      c("subid", "session", "task", "run")
    )
    positions <- which(matched$keep)
    if (length(positions) != 1L) {
      .bids_abort("An event row could not be associated with exactly one selected BOLD run.",
        event_row = i
      )
    }
    manifest$scan_id[[positions]]
  }, character(1))
}

.bids_event_table <- function(project, root, manifest, subject, task, session, run) {
  events <- tibble::as_tibble(.bids_read_events(project, subject, task, session, run))
  if (!nrow(events)) {
    return(NULL)
  }
  events$scan_id <- .bids_event_scan_ids(events, manifest)
  file_column <- if ("file" %in% names(events)) "file" else if (".file" %in% names(events)) ".file" else NULL
  if (is.null(file_column)) {
    event_source <- paste0(events$scan_id, "::events")
  } else {
    raw <- as.character(events[[file_column]])
    event_source <- vapply(raw, function(path) {
      if (!is.na(path) && nzchar(path) && file.exists(path)) {
        .bids_relative_path(path, root)
      } else {
        basename(path)
      }
    }, character(1))
    events[[file_column]] <- event_source
  }
  within_file <- stats::ave(seq_len(nrow(events)), event_source, FUN = seq_along)
  events$event_id <- sprintf("%s::event-%06d", event_source, within_file - 1L)
  event_table(events, key = "event_id", metadata = list(source = "BIDS events.tsv"))
}

#' Read one subject's preprocessed BIDS BOLD data as an fmri_frame
#'
#' `read_bids_bold()` is the narrow BIDS on-ramp to the canonical frame API. It
#' discovers fMRIPrep `desc-preproc` BOLD runs, creates deterministic volume
#' IDs, resolves one common volume space, and returns a lazy NIfTI-backed frame.
#' BOLD values are not read until an assay or spatial map is explicitly
#' collected.
#'
#' @param path Path to a BIDS dataset containing fMRIPrep derivatives.
#' @param subject One exact subject label, with or without the `sub-` prefix.
#' @param task,session,run,space Optional exact BIDS entity selectors.
#' @param derivative Currently only `"fmriprep"`.
#' @param mask `"intersection"` (default), an explicit NIfTI mask path, or a
#'   compatible `volume_space`.
#' @param events Whether to attach matching BIDS events as a keyed event table.
#' @param chunks Optional observation-by-feature chunk hint passed to
#'   [nifti_array_source()].
#' @return A lazy `fmri_frame` with a `signal` assay.
#' @export
#' @examples
#' \dontrun{
#' bold <- read_bids_bold(
#'   "/data/my-study",
#'   subject = "01",
#'   task = "memory",
#'   space = "MNI152NLin2009cAsym"
#' )
#' first_run <- filter_obs(bold, run_id == "run-1")
#' map <- spatial_map(bold, observation = 1)
#' }
read_bids_bold <- function(path, subject, task = NULL, session = NULL,
                           run = NULL, space = NULL,
                           derivative = "fmriprep",
                           mask = "intersection", events = TRUE,
                           chunks = NULL) {
  bidser_version <- .bids_require()
  if (!is.character(path) || length(path) != 1L || is.na(path) ||
    !nzchar(path) || !dir.exists(path)) {
    .bids_abort("path must name an existing BIDS directory.", field = "path")
  }
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  subject <- .bids_scalar(subject, "subject")
  task <- .bids_scalar(task, "task")
  session <- .bids_scalar(session, "session")
  run <- .bids_scalar(run, "run")
  space <- .bids_scalar(space, "space")
  if (!identical(derivative, "fmriprep")) {
    .bids_abort("derivative must currently be 'fmriprep'.", field = "derivative")
  }
  if (!is.logical(events) || length(events) != 1L || is.na(events)) {
    .bids_abort("events must be TRUE or FALSE.", field = "events")
  }

  project <- .bids_open_project(path, derivative)
  manifest <- .bids_bold_manifest(project, path, subject, task, session, run, space)
  selected_space <- attr(manifest, "selected_space", exact = TRUE)
  resolved <- .bids_resolve_space(
    project, path, manifest, mask, subject, session, selected_space
  )
  source <- nifti_array_source(manifest$.path, resolved$space, chunks = chunks)
  observation_data <- .bids_expand_observations(manifest)

  subject_data <- unique(manifest["subject_id"])
  run_columns <- c(
    "scan_id", "subject_id", "session_id", "task_id", "run_id",
    "relative_path", "n_volume", "TR"
  )
  run_data <- manifest[run_columns]
  table_values <- list()
  if (events) {
    event_value <- .bids_event_table(project, path, manifest, subject, task, session, run)
    if (!is.null(event_value)) table_values$events <- event_value
  }
  provenance <- provenance_graph(provenance_record(
    "read_bids_bold",
    inputs = list(scans = manifest$relative_path),
    parameters = list(
      subject = sub("^sub-", "", subject), task = task, session = session,
      run = run, space = selected_space, derivative = derivative,
      mask_policy = resolved$policy, events = events
    ),
    outputs = list(
      n_observation = nrow(observation_data),
      n_feature = n_features(resolved$space)
    ),
    software = list(
      fmridataset = as.character(utils::packageVersion("fmridataset")),
      bidser = bidser_version
    )
  ))

  fmri_frame(
    assays = list(signal = source),
    observations = axis_frame(
      observation_data,
      id = observation_data$.obs_id,
      axis = "observation"
    ),
    space = resolved$space,
    entities = list(
      subject = entity_frame(subject_data, key = "subject_id", entity_type = "subject"),
      run = entity_frame(run_data, key = "scan_id", entity_type = "run")
    ),
    relations = list(
      observation_run = key_relation("scan_id", target = "run"),
      run_subject = key_relation("subject_id", source = "run", target = "subject")
    ),
    tables = table_values,
    active_assay = "signal",
    metadata = list(
      source_format = "BIDS",
      derivative = derivative,
      subject = sub("^sub-", "", subject),
      space = selected_space,
      mask_policy = resolved$policy
    ),
    provenance = provenance
  )
}
