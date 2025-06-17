#' @importFrom assertthat assert_that
#' @importFrom purrr map_lgl
#' @importFrom tibble as_tibble
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

#' Create a Latent Dataset Object
#'
#' This function creates a latent dataset object, which encapsulates a dimension-reduced
#' subspace of "latent variables". The dataset is a list containing information about the latent
#' neuroimaging vector, TR, number of runs, event table, base path, sampling frame, and censor.
#'
#' @param lvec An instance of class \code{LatentNeuroVec}. (Typically, a \code{LatentNeuroVec} is
#'   created using the \code{fmristore} package.)
#' @param TR Repetition time (TR) of the fMRI acquisition.
#' @param run_length A numeric vector specifying the length of each run in the dataset.
#' @param event_table An optional data frame containing event information. Default is an empty data frame.
#'
#' @return A latent dataset object of class \code{c("latent_dataset", "matrix_dataset", "fmri_dataset", "list")}.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Create a matrix with 100 rows and 1000 columns (voxels)
#' X <- matrix(rnorm(100 * 1000), 100, 1000)
#' pres <- prcomp(X)
#' basis <- pres$x[, 1:25]
#' loadings <- pres$rotation[, 1:25]
#' offset <- colMeans(X)
#'
#' # Create a LatentNeuroVec object (requires the fmristore package)
#' lvec <- fmristore::LatentNeuroVec(basis, loadings,
#'   neuroim2::NeuroSpace(c(10, 10, 10, 100)),
#'   mask = rep(TRUE, 1000), offset = offset
#' )
#'
#' # Create a latent_dataset
#' dset <- latent_dataset(lvec, TR = 2, run_length = 100)
#' }
latent_dataset <- function(lvec, TR, run_length, event_table = data.frame()) {
  # Lazy check: make sure fmristore is installed (fmristore is not a hard dependency)
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop("The 'fmristore' package is required to create a latent_dataset. Please install fmristore.",
      call. = FALSE
    )
  }

  # Ensure the total run length matches the number of time points in lvec
  assertthat::assert_that(
    sum(run_length) == dim(lvec)[4],
    msg = "Sum of run lengths must equal the 4th dimension of lvec"
  )

  frame <- fmrihrf::sampling_frame(blocklens = run_length, TR = TR)

  ret <- list(
    lvec = lvec,
    datamat = lvec@basis,
    TR = TR,
    nruns = length(run_length),
    event_table = event_table,
    sampling_frame = frame,
    mask = rep(1, ncol(lvec@basis))
  )

  class(ret) <- c("latent_dataset", "matrix_dataset", "fmri_dataset", "list")
  ret
}

#' Create an fMRI Dataset Object from LatentNeuroVec Files or Objects
#'
#' This function creates an fMRI dataset object from LatentNeuroVec files (.lv.h5) or objects
#' using the new backend architecture. LatentNeuroVec represents data in a compressed latent
#' space using basis functions and spatial loadings.
#'
#' @param latent_files A character vector of file paths to LatentNeuroVec HDF5 files (.lv.h5),
#'   or a list of LatentNeuroVec objects, or a pre-created latent_backend object.
#' @param mask_source Optional mask source. If NULL, the mask will be extracted from
#'   the first LatentNeuroVec object.
#' @param TR The repetition time in seconds of the scan-to-scan interval.
#' @param run_length A vector of one or more integers indicating the number of scans in each run.
#' @param event_table A data.frame containing the event onsets and experimental variables. Default is an empty data.frame.
#' @param base_path Base directory for relative file names. Absolute paths are used as-is.
#' @param censor A binary vector indicating which scans to remove. Default is NULL.
#' @param preload Read LatentNeuroVec objects eagerly rather than on first access. Default is FALSE.
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list").
#'
#' @details
#' This function uses the latent_backend to handle LatentNeuroVec data efficiently.
#' LatentNeuroVec objects store fMRI data in a compressed format using:
#' - Basis functions (temporal components)
#' - Spatial loadings (voxel weights)
#' - Optional offset terms
#'
#' This is particularly efficient for data that can be well-represented by a
#' lower-dimensional basis (e.g., from PCA, ICA, or dictionary learning).
#'
#' **CRITICAL: Data Access in Latent Space**
#' Unlike standard fMRI datasets that return voxel-wise data, this dataset returns
#' **latent scores** (temporal basis components) rather than reconstructed voxel data.
#' The data matrix dimensions are (time × components), not (time × voxels). This is because:
#'
#' - Time-series analyses should be performed in the efficient latent space
#' - The latent scores capture temporal dynamics in the compressed representation
#' - Reconstructing to full voxel space defeats the compression benefits
#' - Most analysis workflows (GLM, connectivity, etc.) work directly with these temporal patterns
#'
#' Use this dataset when you want to analyze temporal dynamics in the latent space.
#' If you need full voxel reconstruction, use the reconstruction methods from fmristore directly.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Create an fMRI dataset from LatentNeuroVec HDF5 files
#' dset <- fmri_latent_dataset(
#'   latent_files = c("run1.lv.h5", "run2.lv.h5", "run3.lv.h5"),
#'   TR = 2,
#'   run_length = c(150, 150, 150)
#' )
#'
#' # Create from pre-loaded LatentNeuroVec objects
#' lvec1 <- fmristore::read_vec("run1.lv.h5")
#' lvec2 <- fmristore::read_vec("run2.lv.h5")
#' dset <- fmri_latent_dataset(
#'   latent_files = list(lvec1, lvec2),
#'   TR = 2,
#'   run_length = c(100, 100)
#' )
#'
#' # Create from a latent_backend
#' backend <- latent_backend(c("run1.lv.h5", "run2.lv.h5"))
#' dset <- fmri_latent_dataset(backend, TR = 2, run_length = c(100, 100))
#' }
#'
#' @seealso
#' \code{\link{latent_backend}}, \code{\link{latent_dataset}}, \code{\link{fmri_h5_dataset}}
fmri_latent_dataset <- function(latent_files, mask_source = NULL, TR,
                                run_length,
                                event_table = data.frame(),
                                base_path = ".",
                                censor = NULL,
                                preload = FALSE) {
  # Check if latent_files is actually a backend object
  if (inherits(latent_files, "latent_backend")) {
    backend <- latent_files
  } else {
    # Create a latent_backend from the input
    if (is.character(latent_files)) {
      # File paths - prepend base_path if needed for relative paths
      latent_files <- ifelse(
        grepl("^(/|[A-Za-z]:)", latent_files), # Check if absolute path
        latent_files,
        file.path(base_path, latent_files)
      )
    }

    backend <- latent_backend(
      source = latent_files,
      mask_source = mask_source,
      preload = preload
    )
  }

  # Validate backend
  validate_backend(backend)

  # Open backend to initialize resources
  backend <- backend_open(backend)

  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }

  frame <- sampling_frame(run_length, TR)

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
#' }
fmri_dataset <- function(scans, mask = NULL, TR,
                         run_length,
                         event_table = data.frame(),
                         base_path = ".",
                         censor = NULL,
                         preload = FALSE,
                         mode = c("normal", "bigvec", "mmap", "filebacked"),
                         backend = NULL) {
  # Check if scans is actually a backend object
  if (inherits(scans, "storage_backend")) {
    backend <- scans
  } else if (!is.null(backend)) {
    warning("backend parameter is deprecated. Pass backend as first argument.")
  } else {
    # Legacy path: create a NiftiBackend from file paths
    assert_that(is.character(mask), msg = "'mask' should be the file name of the binary mask file")
    mode <- match.arg(mode)

    maskfile <- ifelse(
      is_absolute_path(mask),
      mask,
      file.path(base_path, mask)
    )
    scan_files <- ifelse(
      is_absolute_path(scans),
      scans,
      file.path(base_path, scans)
    )

    backend <- nifti_backend(
      source = scan_files,
      mask_source = maskfile,
      preload = preload,
      mode = mode
    )
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
    is_absolute_path(h5_files),
    h5_files,
    file.path(base_path, h5_files)
  )

  mask_file_path <- if (is.character(mask_source)) {
    ifelse(is_absolute_path(mask_source),
           mask_source,
           file.path(base_path, mask_source))
  } else {
    mask_source
  }

  # Create H5 backend
  backend <- h5_backend(
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

  backends <- lapply(datasets, function(d) d$backend)
  sb <- study_backend(backends, subject_ids = subject_ids)

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
