#' Elegant BIDS Interface System for fmridataset
#'
#' This file implements an advanced but loosely coupled BIDS integration system
#' that allows pluggable backends, elegant discovery interfaces, and sophisticated
#' configuration without tight coupling to specific BIDS libraries.
#'
#' @name bids_interface
NULL

# ============================================================================
# BIDS Backend Interface (Pluggable Architecture)
# ============================================================================

#' Create BIDS Backend Interface
#'
#' Creates a pluggable BIDS backend that can work with different BIDS libraries.
#' This provides loose coupling between fmridataset and specific BIDS implementations.
#'
#' @param backend_type Character string specifying backend: "bidser", "pybids", "custom"
#' @param backend_config List of backend-specific configuration options
#' @return A BIDS backend object with standardized interface
#' @export
#' @examples
#' \dontrun{
#' # Using bidser backend
#' backend <- bids_backend("bidser")
#'
#' # Configure bidser backend to prefer preprocessed images
#' backend <- bids_backend("bidser", backend_config = list(prefer_preproc = TRUE))
#'
#' # Using custom backend with configuration
#' backend <- bids_backend("custom",
#'   backend_config = list(
#'     find_scans = my_scan_function,
#'     read_metadata = my_metadata_function,
#'     get_run_info = my_run_info
#'   ))
#' }
bids_backend <- function(backend_type = "bidser", backend_config = list()) {

  if (!is.list(backend_config)) {
    stop("backend_config must be a list")
  }
  
  # Validate backend type
  supported_backends <- c("bidser", "pybids", "custom")
  if (!backend_type %in% supported_backends) {
    stop("Unsupported backend_type: ", backend_type, 
         ". Supported: ", paste(supported_backends, collapse = ", "))
  }
  
  # Create backend object
  backend <- list(
    type = backend_type,
    config = backend_config,
    
    # Standard interface methods (to be populated by backend-specific code)
    find_scans = NULL,
    read_metadata = NULL,
    get_run_info = NULL,
    find_derivatives = NULL,
    validate_bids = NULL
  )
  
  # Initialize backend-specific methods
  backend <- switch(backend_type,
    "bidser" = initialize_bidser_backend(backend),
    "pybids" = initialize_pybids_backend(backend),
    "custom" = initialize_custom_backend(backend, backend_config),
    stop("Unknown backend type: ", backend_type)
  )
  
  class(backend) <- c("bids_backend", paste0("bids_backend_", backend_type))
  return(backend)
}

#' Initialize Bidser Backend
#' @param backend Backend object to populate
#' @return Populated backend object
#' @keywords internal
#' @noRd
initialize_bidser_backend <- function(backend, config) {

  # Check bidser availability
  check_package_available("bidser", "bidser backend", error = TRUE)
  

  backend$config <- config

  backend$find_scans <- function(bids_root, filters) {
    bidser_find_scans(bids_root, filters, config)
  }

  backend$read_metadata <- function(scan_path) {
    bidser_read_metadata(scan_path)
  }

  backend$get_run_info <- function(scan_paths) {
    bidser_get_run_info(scan_paths)
  }

  backend$find_derivatives <- function(bids_root, filters) {
    bidser_find_derivatives(bids_root, filters, config)
  }

  backend$validate_bids <- function(bids_root) {
    bidser_validate_bids(bids_root)
  }

  return(backend)
}

#' Initialize PyBIDS Backend (Future Implementation)
#' @param backend Backend object to populate
#' @return Populated backend object
#' @keywords internal
#' @noRd
initialize_pybids_backend <- function(backend) {
  stop("PyBIDS backend not yet implemented. Use 'bidser' or 'custom' backend.")
}

#' Initialize Custom Backend
#' @param backend Backend object to populate
#' @param config Configuration list with user-provided functions
#' @return Populated backend object
#' @keywords internal
#' @noRd
initialize_custom_backend <- function(backend, config) {
  
  # Validate required functions in config
  required_functions <- c("find_scans", "read_metadata", "get_run_info")
  missing_functions <- setdiff(required_functions, names(config))
  
  if (length(missing_functions) > 0) {
    stop("Custom backend requires these functions in backend_config: ",
         paste(missing_functions, collapse = ", "))
  }
  
  # Assign user-provided functions
  backend$find_scans <- config$find_scans
  backend$read_metadata <- config$read_metadata
  backend$get_run_info <- config$get_run_info
  backend$find_derivatives <- config$find_derivatives %||% function(...) NULL
  backend$validate_bids <- config$validate_bids %||% function(...) TRUE
  
  return(backend)
}

# ============================================================================
# BIDS Query Interface (Elegant Discovery)
# ============================================================================

#' BIDS Query Builder
#'
#' Elegant interface for building complex BIDS queries with method chaining.
#' Provides fluent API for discovering and filtering BIDS datasets.
#'
#' @param bids_root Path to BIDS dataset root
#' @param backend BIDS backend object (optional, defaults to auto-detect)
#' @return A BIDS query object with chainable methods
#' @export
#' @examples
#' \dontrun{
#' # Elegant query building with method chaining
#' query <- bids_query("/path/to/bids") %>%
#'   subject("01", "02") %>%
#'   task("rest", "task") %>%
#'   session("1") %>%
#'   derivatives("fmriprep") %>%
#'   space("MNI152NLin2009cAsym")
#' 
#' # Execute query
#' scans <- query %>% find_scans()
#' metadata <- query %>% get_metadata()
#' 
#' # Direct dataset creation
#' dataset <- query %>% as_fmri_dataset(subject_id = "01")
#' }
bids_query <- function(bids_root, backend = NULL) {
  
  # Auto-detect backend if not provided
  if (is.null(backend)) {
    backend <- auto_detect_bids_backend(bids_root)
  }
  
  # Create query object
  query <- list(
    bids_root = bids_root,
    backend = backend,
    
    # Filter criteria (cumulative)
    filters = list(
      subjects = NULL,
      sessions = NULL,
      tasks = NULL,
      runs = NULL,
      derivatives = NULL,
      spaces = NULL,
      suffixes = NULL,
      extensions = NULL
    ),
    
    # Query options
    options = list(
      validate = TRUE,
      recursive = TRUE,
      include_derivatives = TRUE
    )
  )
  
  class(query) <- "bids_query"
  return(query)
}

#' Add Subject Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Subject IDs to include
#' @return Modified query object (for chaining)
#' @export
subject.bids_query <- function(query, ...) {
  subjects <- c(...)

  if (length(subjects) == 0) {
    return(query)
  }
  if (!is.character(subjects)) {
    warning("subject IDs must be character; coercing")
    subjects <- as.character(subjects)
  }
  if (anyNA(subjects)) {
    warning("NA subject IDs removed")
    subjects <- subjects[!is.na(subjects)]
  }

  query$filters$subjects <- union(query$filters$subjects, subjects)
  return(query)
}

#' Add Task Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Task names to include
#' @return Modified query object (for chaining)
#' @export
task.bids_query <- function(query, ...) {
  tasks <- c(...)

  if (length(tasks) == 0) {
    return(query)
  }
  if (!is.character(tasks)) {
    warning("task names must be character; coercing")
    tasks <- as.character(tasks)
  }
  if (anyNA(tasks)) {
    warning("NA task names removed")
    tasks <- tasks[!is.na(tasks)]
  }

  query$filters$tasks <- union(query$filters$tasks, tasks)
  return(query)
}

#' Add Session Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Session IDs to include
#' @return Modified query object (for chaining)
#' @export
session.bids_query <- function(query, ...) {
  sessions <- c(...)

  if (length(sessions) == 0) {
    return(query)
  }
  if (!is.character(sessions)) {
    warning("session IDs must be character; coercing")
    sessions <- as.character(sessions)
  }
  if (anyNA(sessions)) {
    warning("NA session IDs removed")
    sessions <- sessions[!is.na(sessions)]
  }

  query$filters$sessions <- union(query$filters$sessions, sessions)
  return(query)
}

#' Add Run Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Run numbers to include
#' @return Modified query object (for chaining)
#' @export
run.bids_query <- function(query, ...) {
  runs <- c(...)

  if (length(runs) == 0) {
    return(query)
  }
  if (!is.character(runs)) {
    warning("run numbers must be character; coercing")
    runs <- as.character(runs)
  }
  if (anyNA(runs)) {
    warning("NA run numbers removed")
    runs <- runs[!is.na(runs)]
  }

  query$filters$runs <- union(query$filters$runs, runs)
  return(query)
}

#' Add Derivatives Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Derivative pipeline names to include
#' @return Modified query object (for chaining)
#' @export
derivatives.bids_query <- function(query, ...) {
  derivatives <- c(...)

  if (length(derivatives) == 0) {
    return(query)
  }
  if (!is.character(derivatives)) {
    warning("derivative names must be character; coercing")
    derivatives <- as.character(derivatives)
  }
  if (anyNA(derivatives)) {
    warning("NA derivative names removed")
    derivatives <- derivatives[!is.na(derivatives)]
  }

  query$filters$derivatives <- union(query$filters$derivatives, derivatives)
  return(query)
}

#' Add Space Filter to BIDS Query
#' @param query BIDS query object
#' @param ... Space names to include (for derivatives)
#' @return Modified query object (for chaining)
#' @export
space.bids_query <- function(query, ...) {
  spaces <- c(...)

  if (length(spaces) == 0) {
    return(query)
  }
  if (!is.character(spaces)) {
    warning("space names must be character; coercing")
    spaces <- as.character(spaces)
  }
  if (anyNA(spaces)) {
    warning("NA space names removed")
    spaces <- spaces[!is.na(spaces)]
  }

  query$filters$spaces <- union(query$filters$spaces, spaces)
  return(query)
}

#' Find Scans for a BIDS Query
#'
#' Executes the query using the associated backend and returns the
#' paths to matching scan files.
#'
#' @param x A `bids_query` object
#' @param ... Additional arguments passed to the backend
#' @return Character vector of scan paths
#' @export
find_scans.bids_query <- function(x, ...) {
  x$backend$find_scans(x$bids_root, x$filters)
}

#' Retrieve Metadata for Scans in a Query
#'
#' Reads metadata for each scan returned by `find_scans()`.
#'
#' @param x A `bids_query` object
#' @param ... Additional arguments passed to the backend
#' @return List of metadata entries, one per scan
#' @export
get_metadata.bids_query <- function(x, ...) {
  scans <- find_scans(x, ...)
  lapply(scans, x$backend$read_metadata)
}

#' Get Run Information for a Query
#'
#' Retrieves run-level information (e.g., run lengths) for the scans
#' matched by the query.
#'
#' @param x A `bids_query` object
#' @param ... Additional arguments passed to the backend
#' @return Backend-specific run information
#' @export
get_run_info.bids_query <- function(x, ...) {
  scans <- find_scans(x, ...)
  x$backend$get_run_info(scans)
}

# ============================================================================
# BIDS Discovery Interface
# ============================================================================

#' Discover Available BIDS Entities
#'
#' Elegant interface for exploring what's available in a BIDS dataset.
#' Returns structured information about subjects, tasks, sessions, etc.
#'
#' @param bids_root Path to BIDS dataset root
#' @param backend BIDS backend object (optional)
#' @return List with discovered entities
#' @export
#' @examples
#' \dontrun{
#' # Discover what's available
#' discovery <- bids_discover("/path/to/bids")
#'
#' # View structure
#' print(discovery)
#' 
#' # Access specific entities
#' discovery$subjects
#' discovery$tasks
#' discovery$derivatives$pipelines
#'
#' # Individual helper functions
#' subjects <- discover_subjects(discovery$backend, "/path/to/bids")
#' tasks <- discover_tasks(discovery$backend, "/path/to/bids")
#' }
bids_discover <- function(bids_root, backend = NULL) {
  
  if (is.null(backend)) {
    backend <- auto_detect_bids_backend(bids_root)
  }
  
  # Validate BIDS dataset
  if (!backend$validate_bids(bids_root)) {
    warning("BIDS validation failed for: ", bids_root)
  }
  
  # Discover entities
  discovery <- list(
    bids_root = bids_root,
    
    # Core entities
    subjects = discover_subjects(backend, bids_root),
    sessions = discover_sessions(backend, bids_root),
    tasks = discover_tasks(backend, bids_root),
    runs = discover_runs(backend, bids_root),
    
    # Data types
    datatypes = discover_datatypes(backend, bids_root),
    
    # Derivatives
    derivatives = discover_derivatives(backend, bids_root),
    
    # Summary statistics
    summary = create_discovery_summary(backend, bids_root)
  )
  
  class(discovery) <- "bids_discovery"
  return(discovery)
}

# ============================================================================
# BIDS Configuration System
# ============================================================================

#' BIDS Configuration Builder
#'
#' Create sophisticated BIDS configurations for dataset creation with
#' elegant defaults and advanced customization options.
#'
#' @param image_selection List specifying image selection strategy
#' @param preprocessing List specifying preprocessing options
#' @param quality_control List specifying QC options
#' @param metadata_extraction List specifying metadata options
#' @return BIDS configuration object
#' @export
#' @examples
#' \dontrun{
#' # Elegant configuration
#' config <- bids_config(
#'   image_selection = list(
#'     strategy = "prefer_derivatives",
#'     preferred_pipelines = c("fmriprep", "nilearn"),
#'     required_space = "MNI152NLin2009cAsym"
#'   ),
#'   quality_control = list(
#'     censoring_threshold = 0.5
#'   )
#' )
#'
#' # Use configuration
#' dataset <- bids_query("/path/to/bids") %>%
#'   subject("01") %>%
#'   task("rest") %>%
#'   as_fmri_dataset(config = config)
#' }
bids_config <- function(image_selection = NULL,
                       preprocessing = NULL,
                       quality_control = NULL,
                       metadata_extraction = NULL) {
  
  # Default sophisticated configuration
  config <- list(
    image_selection = list(
      strategy = "auto",  # "auto", "raw", "derivatives", "prefer_derivatives"
      preferred_pipelines = c("fmriprep", "nilearn", "afni", "spm"),
      required_space = NULL,
      required_resolution = NULL,
      fallback_to_raw = TRUE
    ),
    
    preprocessing = list(
      auto_detect_applied = TRUE,
      validate_preprocessing = TRUE,
      merge_pipeline_metadata = TRUE
    ),
    
    quality_control = list(
      check_completeness = TRUE,
      validate_headers = TRUE,
      check_temporal_alignment = TRUE,
      censoring_threshold = NULL
    ),
    
    metadata_extraction = list(
      include_all_sidecars = TRUE,
      merge_inheritance = TRUE,
      extract_physio = FALSE,
      extract_motion = TRUE
    ),
    
    # Advanced options
    advanced = list(
      lazy_validation = FALSE,
      cache_metadata = TRUE,
      parallel_discovery = TRUE
    )
  )
  
  # Override defaults with user inputs
  if (!is.null(image_selection)) {
    config$image_selection <- modifyList(config$image_selection, image_selection)
  }
  if (!is.null(preprocessing)) {
    config$preprocessing <- modifyList(config$preprocessing, preprocessing)
  }
  if (!is.null(quality_control)) {
    config$quality_control <- modifyList(config$quality_control, quality_control)
  }
  if (!is.null(metadata_extraction)) {
    config$metadata_extraction <- modifyList(config$metadata_extraction, metadata_extraction)
  }
  
  class(config) <- "bids_config"
  return(config)
}

# ============================================================================
# Enhanced as.fmri_dataset.bids_query Method
# ============================================================================

#' Convert BIDS Query to fmri_dataset
#'
#' Elegant method for creating fmri_dataset objects from BIDS queries.
#' This provides the most sophisticated and user-friendly interface.
#'
#' @param x BIDS query object
#' @param subject_id Subject ID to extract (required)
#' @param config BIDS configuration object (optional)
#' @param transformation_pipeline Transformation pipeline (optional)
#' @param ... Additional arguments passed to fmri_dataset_create
#' @return fmri_dataset object
#' @export
as.fmri_dataset.bids_query <- function(x, subject_id, 
                                      config = NULL,
                                      transformation_pipeline = NULL,
                                      ...) {
  
  if (missing(subject_id)) {
    stop("subject_id is required for BIDS query conversion")
  }
  
  # Use default config if not provided
  if (is.null(config)) {
    config <- bids_config()
  }
  
  # Execute sophisticated BIDS extraction
  extraction_result <- execute_bids_extraction(x, subject_id, config)
  
  # Create fmri_dataset with extracted information
  fmri_dataset_create(
    images = extraction_result$images,
    mask = extraction_result$mask,
    TR = extraction_result$TR,
    run_lengths = extraction_result$run_lengths,
    event_table = extraction_result$events,
    censor_vector = extraction_result$censoring,
    transformation_pipeline = transformation_pipeline,
    metadata = extraction_result$metadata,
    ...
  )
}

# ============================================================================
# Helper Functions (Elegant Implementations)
# ============================================================================

#' Auto-detect Best BIDS Backend
#' @param bids_root BIDS dataset path
#' @return BIDS backend object
#' @keywords internal
#' @noRd
auto_detect_bids_backend <- function(bids_root) {

  # Currently only bidser backend is supported
  check_package_available("bidser", "BIDS backend", error = TRUE)
  bids_backend("bidser")
}

#' Execute Sophisticated BIDS Extraction
#' @param query BIDS query object
#' @param subject_id Subject ID
#' @param config BIDS configuration
#' @return List with extracted components
#' @keywords internal
#' @noRd
execute_bids_extraction <- function(query, subject_id, config) {
  
  # Add subject filter to query
  query <- subject(query, subject_id)
  
  # Execute sophisticated extraction based on config
  # This would implement the advanced logic for:
  # - Intelligent image selection
  # - Preprocessing detection
  # - Quality control
  # - Metadata extraction
  
  # For now, delegate to existing implementation
  # This is where the sophisticated logic would go
  
  stop("Sophisticated BIDS extraction not yet implemented.\n",
       "This is the placeholder for the advanced BIDS interface.")
}

# Placeholder implementations for backend-specific functions
bidser_find_scans <- function(bids_root, filters, config = list()) {

  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(character(0))
  }

  # Allow passing either a path or a bids_project object
  proj <- if (inherits(bids_root, "bids_project")) {
    bids_root
  } else if (inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }

  subid <- filters$subjects
  task <- filters$tasks
  session <- filters$sessions
  run <- filters$runs

  if (!is.null(filters$derivatives) || isTRUE(config$prefer_preproc)) {
    pipeline <- NULL
    if (!is.null(filters$derivatives)) pipeline <- filters$derivatives[1]
    if (is.null(pipeline)) pipeline <- config$pipeline

    space <- NULL
    if (!is.null(filters$spaces)) space <- filters$spaces[1]

    scans <- tryCatch(
      bidser::preproc_scans(proj, subid = subid, task = task,
                            run = run, session = session,
                            variant = pipeline, space = space,
                            full_path = TRUE),
      error = function(e) character(0)
    )
    if (length(scans) > 0) {
      return(scans)
    }
  }

  tryCatch(
    bidser::func_scans(proj, subid = subid, task = task,
                       run = run, session = session,
                       full_path = TRUE),
    error = function(e) character(0)
##>>>>>>> main
  )
}

bidser_read_metadata <- function(scan_path) {
##<<<<<<< codex/audit-unused-function-arguments-in-r/bids_interface.r
  sidecar <- sub("\\.nii(\\.gz)?$", ".json", scan_path, ignore.case = TRUE)
  if (file.exists(sidecar)) {
    return(jsonlite::read_json(sidecar, simplifyVector = TRUE))
  }
  list()
}

bidser_get_run_info <- function(scan_paths) {
  if (!check_package_available("neuroim2", "reading NIfTI headers", error = FALSE)) {
    stop("neuroim2 package is required to determine run information")
  }
  vapply(scan_paths, function(p) {
    dims <- dim(neuroim2::read_vol(p))
    if (length(dims) < 4) {
      stop("Image has no time dimension: ", p)
    }
    dims[4]
  }, integer(1))
}

bidser_find_derivatives <- function(bids_root, filters) {
  check_package_available("bidser", "BIDS derivative discovery", error = TRUE)
  proj <- bidser::bids_project(bids_root)
  bidser::preproc_scans(
    proj,
    subid = filters$subjects,
    session = filters$sessions,
    task = filters$tasks,
    run = filters$runs,
    space = filters$spaces,
    variant = filters$derivatives,
    full_path = TRUE
##=======
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    return(list())
  }
  sidecar <- sub("\\.nii(\\.gz)?$", ".json", scan_path)
  if (file.exists(sidecar)) {
    tryCatch(jsonlite::read_json(sidecar, simplifyVector = TRUE),
             error = function(e) list())
  } else {
    list()
  }
}

bidser_get_run_info <- function(scan_paths) {
  lengths <- rep(NA_integer_, length(scan_paths))
  if (requireNamespace("neuroim2", quietly = TRUE)) {
    lengths <- vapply(scan_paths, function(p) {
      tryCatch(dim(neuroim2::read_vol(p))[4], error = function(e) NA_integer_)
    }, integer(1))
  }
  data.frame(path = scan_paths, run_length = lengths,
             stringsAsFactors = FALSE)
}

bidser_find_derivatives <- function(bids_root, filters, config = list()) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(character(0))
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }

  pipeline <- filters$derivatives
  if (length(pipeline) == 0) pipeline <- config$pipeline

  tryCatch(
    bidser::preproc_scans(proj, subid = filters$subjects,
                          task = filters$tasks, run = filters$runs,
                          session = filters$sessions, variant = pipeline,
                          space = filters$spaces, full_path = TRUE),
    error = function(e) character(0)
##>>>>>>> main
  )
}

bidser_validate_bids <- function(bids_root) {
##<<<<<<< codex/audit-unused-function-arguments-in-r/bids_interface.r
  if (!check_package_available("bidser", "BIDS validation", error = FALSE)) {
    return(FALSE)
  }
  proj <- bidser::bids_project(bids_root)
  res <- tryCatch(bidser::bids_check_compliance(proj), error = function(e) e)
  !inherits(res, "error")
}

discover_subjects <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    return(bidser::participants(proj)$participant_id)
  }
  stop("discover_subjects not implemented for backend type: ", backend$type)
}

discover_sessions <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    return(bidser::sessions(proj))
  }
  stop("discover_sessions not implemented for backend type: ", backend$type)
}

discover_tasks <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    return(bidser::tasks(proj))
  }
  stop("discover_tasks not implemented for backend type: ", backend$type)
}

discover_runs <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    scans <- bidser::func_scans(proj, full_path = FALSE)
    info <- bidser::encode(scans)
    return(unique(info$run))
  }
  stop("discover_runs not implemented for backend type: ", backend$type)
}

discover_datatypes <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    summary <- bidser::bids_summary(proj)
    return(names(summary$datatype_counts))
  }
  stop("discover_datatypes not implemented for backend type: ", backend$type)
}

discover_derivatives <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    derivs <- bidser::preproc_scans(proj, full_path = FALSE)
    info <- bidser::encode(derivs)
    return(unique(info$variant))
  }
  stop("discover_derivatives not implemented for backend type: ", backend$type)
}

create_discovery_summary <- function(backend, bids_root) {
  if (backend$type == "bidser") {
    check_package_available("bidser", "BIDS discovery", error = TRUE)
    proj <- bidser::bids_project(bids_root)
    return(bidser::bids_summary(proj))
  }
  stop("create_discovery_summary not implemented for backend type: ", backend$type)
}
=======
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(FALSE)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  tryCatch({
    bidser::bids_check_compliance(proj)
    TRUE
  }, error = function(e) FALSE)
}

discover_subjects <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  tryCatch(bidser::participants(proj), error = function(e) NULL)
}

discover_sessions <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  tryCatch(bidser::sessions(proj), error = function(e) NULL)
}

discover_tasks <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  tryCatch(bidser::tasks(proj), error = function(e) NULL)
}

discover_runs <- function(backend, bids_root) {
  scans <- backend$find_scans(bids_root, list())
  matches <- regmatches(scans, regexpr("run-[0-9]+", scans))
  unique(sub("run-", "", matches))
}

discover_datatypes <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  fl <- tryCatch(bidser::flat_list(proj, full_path = FALSE),
                 error = function(e) NULL)
  if (is.null(fl) || !"datatype" %in% names(fl)) return(NULL)
  unique(fl$datatype)
}

discover_derivatives <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  summary <- tryCatch(bidser::bids_summary(proj), error = function(e) NULL)
  if (!is.null(summary) && "pipelines" %in% names(summary)) {
    list(pipelines = summary$pipelines)
  } else {
    list(pipelines = NULL)
  }
}

create_discovery_summary <- function(backend, bids_root) {
  if (!requireNamespace("bidser", quietly = TRUE)) {
    return(NULL)
  }
  proj <- if (inherits(bids_root, "bids_project") ||
                inherits(bids_root, "mock_bids_project")) {
    bids_root
  } else {
    bidser::bids_project(bids_root)
  }
  tryCatch(bidser::bids_summary(proj), error = function(e) NULL)
}



