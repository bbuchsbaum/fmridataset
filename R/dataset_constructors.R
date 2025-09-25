#' @importFrom assertthat assert_that
#' @importFrom fs is_absolute_path
#' @importFrom purrr map_lgl
#' @importFrom tibble as_tibble
#' @importFrom lifecycle deprecate_warn
NULL

#' Matrix Dataset Constructor
#'
#' This function creates a matrix dataset object, which is a list containing
#' information about the data matrix, TR, number of runs, event table,
#' sampling frame, and mask.
#'
#' @param datamat A matrix where each column is a voxel time-series.
#' @param TR Repetition time (TR) of the fMRI acquisition.
#' @param run_length A numeric vector specifying the length of each run in the dataset.
#' @param event_table An optional data frame containing event information. Default is an empty data frame.
#'
#' @return A matrix dataset object of class c("matrix_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' # A matrix with 100 rows and 100 columns (voxels)
#' X <- matrix(rnorm(100 * 100), 100, 100)
#' dset <- matrix_dataset(X, TR = 2, run_length = 100)
matrix_dataset <- function(datamat, TR, run_length, event_table = data.frame()) {
  if (is.vector(datamat)) {
    datamat <- as.matrix(datamat)
  }
  assert_that(sum(run_length) == nrow(datamat))

  frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)

  # For backward compatibility, keep the original structure
  # but could optionally add backend support in the future
  ret <- list(
    datamat = datamat,
    TR = TR,
    nruns = length(run_length),
    event_table = event_table,
    sampling_frame = frame,
    mask = rep(1, ncol(datamat))
  )

  class(ret) <- c("matrix_dataset", "fmri_dataset", "list")
  ret
}

#' Create an fMRI Memory Dataset Object
#'
#' This function creates an fMRI memory dataset object, which is a list containing information about the scans, mask, TR, number of runs, event table, base path, sampling frame, and censor.
#'
#' @param scans A list of objects of class \code{NeuroVec} from the neuroim2 package.
#' @param mask A binary mask of class \code{NeuroVol} from the neuroim2 package indicating the set of voxels to include in analyses.
#' @param TR Repetition time (TR) of the fMRI acquisition.
#' @param run_length A numeric vector specifying the length of each run in the dataset. Default is the length of the scans.
#' @param event_table An optional data frame containing event information. Default is an empty data frame.
#' @param base_path Base directory for relative file names. Absolute paths are used as-is.
#' @param censor An optional numeric vector specifying which time points to censor. Default is NULL.
#'
#' @return An fMRI memory dataset object of class c("fmri_mem_dataset", "volumetric_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' # Create a NeuroVec object
#' d <- c(10, 10, 10, 10)
#' nvec <- neuroim2::NeuroVec(array(rnorm(prod(d)), d), space = neuroim2::NeuroSpace(d))
#'
#' # Create a NeuroVol mask
#' mask <- neuroim2::NeuroVol(array(rnorm(10 * 10 * 10), d[1:3]), space = neuroim2::NeuroSpace(d[1:3]))
#' mask[mask < .5] <- 0
#'
#' # Create an fmri_mem_dataset
#' dset <- fmri_mem_dataset(list(nvec), mask, TR = 2)
fmri_mem_dataset <- function(scans, mask, TR,
                             run_length = sapply(scans, function(x) dim(x)[4]),
                             event_table = data.frame(),
                             base_path = ".",
                             censor = NULL) {
  assert_that(all(map_lgl(scans, function(x) inherits(x, "NeuroVec"))))
  assert_that(inherits(mask, "NeuroVol"))
  assert_that(all(dim(mask) == dim(scans[[1]][1:3])))

  ntotscans <- sum(sapply(scans, function(x) dim(x)[4]))
  # run_length <- map_dbl(scans, ~ dim(.)[4])
  assert_that(sum(run_length) == ntotscans)

  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }

  frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)

  ret <- list(
    scans = scans,
    mask = mask,
    nruns = length(run_length),
    event_table = event_table,
    base_path = base_path,
    sampling_frame = frame,
    censor = censor
  )

  class(ret) <- c("fmri_mem_dataset", "volumetric_dataset", "fmri_dataset", "list")
  ret
}


#' Create an fMRI Dataset Object from LatentNeuroVec Files or Objects
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' This function is deprecated. Please use `latent_dataset()` instead,
#' which provides a proper interface for latent space data.
#'
#' @param latent_files Source files or objects
#' @param mask_source Ignored
#' @param TR The repetition time in seconds
#' @param run_length Vector of run lengths
#' @param event_table Event table
#' @param base_path Base path for files
#' @param censor Censor vector
#' @param preload Whether to preload data
#'
#' @return A latent_dataset object
#' @export
#'
#' @examples
#' \dontrun{
#' # Use latent_dataset() instead:
#' dset <- latent_dataset(
#'   source = c("run1.lv.h5", "run2.lv.h5", "run3.lv.h5"),
#'   TR = 2,
#'   run_length = c(150, 150, 150)
#' )
#' }
#'
#' @seealso \code{\link{latent_dataset}}
fmri_latent_dataset <- function(latent_files, mask_source = NULL, TR,
                                run_length,
                                event_table = data.frame(),
                                base_path = ".",
                                censor = NULL,
                                preload = FALSE) {
  lifecycle::deprecate_warn(
    "0.9.0",
    "fmri_latent_dataset()",
    "latent_dataset()",
    details = "The new interface provides proper handling of latent space data.",
    always = TRUE
  )

  # Forward to new function
  latent_dataset(
    source = latent_files,
    TR = TR,
    run_length = run_length,
    event_table = event_table,
    base_path = base_path,
    censor = censor,
    preload = preload
  )
}

#' Create an fMRI Dataset Object from a Set of Scans
#'
#' This function creates an fMRI dataset object from a set of scans, design information, and other data.
#' The new implementation uses a pluggable backend architecture.
#'
#' @param scans A vector of one or more file names of the images comprising the dataset,
#'   or a pre-created storage backend object.
#' @param mask Name of the binary mask file indicating the voxels to include in the analysis.
#'   Ignored if scans is a backend object.
#' @param TR The repetition time in seconds of the scan-to-scan interval.
#' @param run_length A vector of one or more integers indicating the number of scans in each run.
#' @param event_table A data.frame containing the event onsets and experimental variables. Default is an empty data.frame.
#' @param base_path Base directory for relative file names. Absolute paths are used as-is.
#' @param censor A binary vector indicating which scans to remove. Default is NULL.
#' @param preload Read image scans eagerly rather than on first access. Default is FALSE.
#' @param mode The type of storage mode ('normal', 'bigvec', 'mmap', filebacked'). Default is 'normal'.
#'   Ignored if scans is a backend object.
#' @param backend Deprecated. Use scans parameter to pass a backend object.
#' @param dummy_mode Logical, if TRUE allows non-existent files (for testing purposes only). Default is FALSE.
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' \dontrun{
#' # Create an fMRI dataset with 3 scans and a mask
#' dset <- fmri_dataset(c("scan1.nii", "scan2.nii", "scan3.nii"),
#'   mask = "mask.nii", TR = 2, run_length = rep(300, 3),
#'   event_table = data.frame(
#'     onsets = c(3, 20, 99, 3, 20, 99, 3, 20, 99),
#'     run = c(1, 1, 1, 2, 2, 2, 3, 3, 3)
#'   )
#' )
#'
#' # Create an fMRI dataset with 1 scan and a mask
#' dset <- fmri_dataset("scan1.nii",
#'   mask = "mask.nii", TR = 2,
#'   run_length = 300,
#'   event_table = data.frame(onsets = c(3, 20, 99), run = rep(1, 3))
#' )
#'
#' # Create an fMRI dataset with a backend
#' backend <- nifti_backend(c("scan1.nii", "scan2.nii"), mask_source = "mask.nii")
#' dset <- fmri_dataset(backend, TR = 2, run_length = c(150, 150))
#'
#' # Create a dummy dataset for testing (files don't need to exist)
#' dset_dummy <- fmri_dataset(
#'   scans = c("dummy1.nii", "dummy2.nii"),
#'   mask = "dummy_mask.nii",
#'   TR = 2,
#'   run_length = c(100, 100),
#'   dummy_mode = TRUE # Enable dummy mode for testing
#' )
#' }
fmri_dataset <- function(scans, mask = NULL, TR,
                         run_length,
                         event_table = data.frame(),
                         base_path = ".",
                         censor = NULL,
                         preload = FALSE,
                         mode = c("normal", "bigvec", "mmap", "filebacked"),
                         backend = NULL,
                         dummy_mode = FALSE) {
  # Check if scans is actually a backend object
  if (inherits(scans, "storage_backend")) {
    backend <- scans
  } else if (!is.null(backend)) {
    warning("backend parameter is deprecated. Pass backend as first argument.")
  } else {
    # Legacy path: create a NiftiBackend from file paths
    assert_that(is.character(mask) && length(mask) == 1, msg = "'mask' should be the file name of the binary mask file")
    mode <- match.arg(mode)

    # Handle paths
    abs_mask <- fs::is_absolute_path(mask)
    maskfile <- if (length(abs_mask) == 1 && abs_mask) {
      mask
    } else {
      file.path(base_path, mask)
    }

    # For scan files, handle each one
    abs_scans <- fs::is_absolute_path(scans)
    scan_files <- character(length(scans))
    for (i in seq_along(scans)) {
      scan_files[i] <- if (length(abs_scans) >= i && abs_scans[i]) {
        scans[i]
      } else {
        file.path(base_path, scans[i])
      }
    }

    # Use registry to create backend for future extensibility
    backend <- create_backend("nifti",
      source = scan_files,
      mask_source = maskfile,
      preload = preload,
      mode = mode,
      dummy_mode = dummy_mode
    )
  }

  # Store run_length in backend for dummy mode (must be before validation)
  if (inherits(backend, "nifti_backend") && isTRUE(backend$dummy_mode)) {
    backend$run_length <- run_length
  }

  # Validate backend
  validate_backend(backend)

  # Open backend to initialize resources
  backend <- backend_open(backend)

  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }

  frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)

  # Get dimensions to validate run_length
  dims <- backend_get_dims(backend)
  assert_that(sum(run_length) == dims$time,
    msg = sprintf(
      "Sum of run_length (%d) must equal total time points (%d)",
      sum(run_length), dims$time
    )
  )

  ret <- list(
    backend = backend,
    nruns = length(run_length),
    event_table = suppressMessages(tibble::as_tibble(event_table, .name_repair = "check_unique")),
    sampling_frame = frame,
    censor = censor
  )

  class(ret) <- c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list")
  ret
}

#' Create an fMRI Dataset Object from H5 Files
#'
#' This function creates an fMRI dataset object specifically from H5 files using the fmristore package.
#' Each scan is stored as an H5 file that loads to an H5NeuroVec object.
#'
#' @param h5_files A vector of one or more file paths to H5 files containing the fMRI data.
#' @param mask_source File path to H5 mask file, regular mask file, or in-memory NeuroVol object.
#' @param TR The repetition time in seconds of the scan-to-scan interval.
#' @param run_length A vector of one or more integers indicating the number of scans in each run.
#' @param event_table A data.frame containing the event onsets and experimental variables. Default is an empty data.frame.
#' @param base_path Base directory for relative file names. Absolute paths are used as-is.
#' @param censor A binary vector indicating which scans to remove. Default is NULL.
#' @param preload Read H5NeuroVec objects eagerly rather than on first access. Default is FALSE.
#' @param mask_dataset Character string specifying the dataset path within H5 file for mask (default: "data/elements").
#' @param data_dataset Character string specifying the dataset path within H5 files for data (default: "data").
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' \dontrun{
#' # Create an fMRI dataset with H5NeuroVec files (standard fmristore format)
#' dset <- fmri_h5_dataset(
#'   h5_files = c("scan1.h5", "scan2.h5", "scan3.h5"),
#'   mask_source = "mask.h5",
#'   TR = 2,
#'   run_length = c(150, 150, 150)
#' )
#'
#' # Create an fMRI dataset with H5 files and NIfTI mask
#' dset <- fmri_h5_dataset(
#'   h5_files = "single_scan.h5",
#'   mask_source = "mask.nii",
#'   TR = 2,
#'   run_length = 300
#' )
#'
#' # Custom dataset paths (if using non-standard H5 structure)
#' dset <- fmri_h5_dataset(
#'   h5_files = "custom_scan.h5",
#'   mask_source = "custom_mask.h5",
#'   TR = 2,
#'   run_length = 200,
#'   data_dataset = "my_data_path",
#'   mask_dataset = "my_mask_path"
#' )
#' }
fmri_h5_dataset <- function(h5_files, mask_source, TR,
                            run_length,
                            event_table = data.frame(),
                            base_path = ".",
                            censor = NULL,
                            preload = FALSE,
                            mask_dataset = "data/elements",
                            data_dataset = "data") {
  # Prepare file paths
  h5_file_paths <- ifelse(
    fs::is_absolute_path(h5_files),
    h5_files,
    file.path(base_path, h5_files)
  )

  mask_file_path <- if (is.character(mask_source)) {
    ifelse(fs::is_absolute_path(mask_source),
      mask_source,
      file.path(base_path, mask_source)
    )
  } else {
    mask_source
  }

  # Create H5 backend using registry
  backend <- create_backend("h5",
    source = h5_file_paths,
    mask_source = mask_file_path,
    mask_dataset = mask_dataset,
    data_dataset = data_dataset,
    preload = preload
  )

  # Use the generic fmri_dataset constructor with the H5 backend
  fmri_dataset(
    scans = backend,
    TR = TR,
    run_length = run_length,
    event_table = event_table,
    censor = censor
  )
}

#' Create an fmri_study_dataset
#'
#' High level constructor that combines multiple `fmri_dataset` objects
#' into a single study-level dataset using `study_backend`.
#'
#' @param datasets A list of `fmri_dataset` objects
#' @param subject_ids Optional vector of subject identifiers
#' @return An object of class `fmri_study_dataset`
#' @export
fmri_study_dataset <- function(datasets, subject_ids = NULL) {
  if (!is.list(datasets) || length(datasets) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      "datasets must be a non-empty list"
    )
  }

  lapply(datasets, function(d) {
    if (!inherits(d, "fmri_dataset")) {
      stop_fmridataset(
        fmridataset_error_config,
        "all elements of datasets must inherit from 'fmri_dataset'"
      )
    }
  })

  if (is.null(subject_ids)) {
    subject_ids <- seq_along(datasets)
  }

  if (length(subject_ids) != length(datasets)) {
    stop_fmridataset(
      fmridataset_error_config,
      "subject_ids must match length of datasets"
    )
  }

  trs <- vapply(datasets, function(d) get_TR(d$sampling_frame), numeric(1))
  if (!all(vapply(trs[-1], function(tr) isTRUE(all.equal(tr, trs[1])), logical(1)))) {
    stop_fmridataset(
      fmridataset_error_config,
      "All datasets must have equal TR"
    )
  }

  DelayedArray::setAutoBlockSize(64 * 1024^2)

  backends <- lapply(datasets, function(d) {
    if (inherits(d, "matrix_dataset") && !is.null(d$datamat)) {
      # Convert legacy matrix_dataset to matrix_backend using registry
      mask_logical <- as.logical(d$mask)
      create_backend("matrix", data_matrix = d$datamat, mask = mask_logical)
    } else if (!is.null(d$backend)) {
      # New-style dataset with backend
      d$backend
    } else {
      # This should not happen but return the dataset for study_backend to handle
      d
    }
  })
  sb <- create_backend("study", backends = backends, subject_ids = subject_ids)

  events <- Map(function(d, sid) {
    et <- tibble::as_tibble(d$event_table)
    if (nrow(et) > 0) {
      et$subject_id <- sid
      if (!"run_id" %in% names(et)) {
        et$run_id <- rep(seq_len(d$nruns), length.out = nrow(et))
      }
    }
    et
  }, datasets, subject_ids)
  combined_events <- do.call(rbind, events)

  run_lengths <- unlist(lapply(datasets, function(d) d$sampling_frame$blocklens))
  frame <- fmrihrf::sampling_frame(blocklens = run_lengths, TR = trs[1])

  ret <- list(
    backend = sb,
    event_table = combined_events,
    sampling_frame = frame,
    subject_ids = subject_ids
  )

  class(ret) <- c("fmri_study_dataset", "fmri_dataset", "list")
  ret
}

#' Attach rowData metadata to a DelayedMatrix
#'
#' Helper for reattaching metadata after DelayedMatrixStats operations.
#'
#' @param x A DelayedMatrix
#' @param rowData A data.frame of row-wise metadata
#' @return `x` with `rowData` attribute set
#' @export
with_rowData <- function(x, rowData) {
  attr(x, "rowData") <- rowData
  x
}
