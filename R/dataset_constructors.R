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
#' X <- matrix(rnorm(100*100), 100, 100)
#' dset <- matrix_dataset(X, TR=2, run_length=100)
matrix_dataset <- function(datamat, TR, run_length, event_table=data.frame()) {
  if (is.vector(datamat)) {
    datamat <- as.matrix(datamat)
  }
  assert_that(sum(run_length) == nrow(datamat))
  
  frame <- sampling_frame(run_length, TR)
  
  # For backward compatibility, keep the original structure
  # but could optionally add backend support in the future
  ret <- list(
    datamat=datamat,
    TR=TR,
    nruns=length(run_length),
    event_table=event_table,
    sampling_frame=frame,
    mask=rep(1,ncol(datamat))
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
#' @param base_path An optional base path for the dataset. Default is "." (current directory).
#' @param censor An optional numeric vector specifying which time points to censor. Default is NULL.
#'
#' @return An fMRI memory dataset object of class c("fmri_mem_dataset", "volumetric_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' # Create a NeuroVec object
#' d <- c(10, 10, 10, 10)
#' nvec <- neuroim2::NeuroVec(array(rnorm(prod(d)), d), space=neuroim2::NeuroSpace(d))
#'
#' # Create a NeuroVol mask
#' mask <- neuroim2::NeuroVol(array(rnorm(10*10*10), d[1:3]), space=neuroim2::NeuroSpace(d[1:3]))
#' mask[mask < .5] <- 0
#'
#' # Create an fmri_mem_dataset
#' dset <- fmri_mem_dataset(list(nvec), mask, TR=2)
fmri_mem_dataset <- function(scans, mask, TR, 
                             run_length=sapply(scans, function(x) dim(x)[4]),
                             event_table=data.frame(), 
                             base_path=".",
                             censor=NULL) {
  
  
  
  assert_that(all(map_lgl(scans, function(x) inherits(x, "NeuroVec"))))
  assert_that(inherits(mask, "NeuroVol"))
  assert_that(all(dim(mask) == dim(scans[[1]][1:3])))
  
  ntotscans <- sum(sapply(scans, function(x) dim(x)[4]))
  #run_length <- map_dbl(scans, ~ dim(.)[4])
  assert_that(sum(run_length) == ntotscans)
  
  if (is.null(censor)) {
    censor <- rep(0, sum(run_length))
  }

  frame <- sampling_frame(run_length, TR)
  
  ret <- list(
    scans=scans,
    mask=mask,
    nruns=length(run_length),
    event_table=event_table,
    base_path=base_path,
    sampling_frame=frame,
    censor=censor
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
#'             neuroim2::NeuroSpace(c(10, 10, 10, 100)),
#'             mask = rep(TRUE, 1000), offset = offset)
#'
#' # Create a latent_dataset
#' dset <- latent_dataset(lvec, TR = 2, run_length = 100)
#' }
latent_dataset <- function(lvec, TR, run_length, event_table = data.frame()) {
  # Lazy check: make sure fmristore is installed (fmristore is not a hard dependency)
  if (!requireNamespace("fmristore", quietly = TRUE)) {
    stop("The 'fmristore' package is required to create a latent_dataset. Please install fmristore.",
         call. = FALSE)
  }
  
  # Ensure the total run length matches the number of time points in lvec
  assertthat::assert_that(
    sum(run_length) == dim(lvec)[4],
    msg = "Sum of run lengths must equal the 4th dimension of lvec"
  )
  
  frame <- sampling_frame(run_length, TR)
  
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
#' @param base_path The file path to be prepended to relative file names. Default is "." (current directory).
#'   Ignored if scans is a backend object.
#' @param censor A binary vector indicating which scans to remove. Default is NULL.
#' @param preload Read image scans eagerly rather than on first access. Default is FALSE.
#'   Ignored if scans is a backend object.
#' @param mode The type of storage mode ('normal', 'bigvec', 'mmap', filebacked'). Default is 'normal'.
#'   Ignored if scans is a backend object.
#' @param backend Deprecated. Use scans parameter to pass a backend object.
#'
#' @return An fMRI dataset object of class c("fmri_file_dataset", "volumetric_dataset", "fmri_dataset", "list").
#' @export
#'
#' @examples
#' # Create an fMRI dataset with 3 scans and a mask
#' dset <- fmri_dataset(c("scan1.nii", "scan2.nii", "scan3.nii"), 
#'   mask="mask.nii", TR=2, run_length=rep(300, 3), 
#'   event_table=data.frame(onsets=c(3, 20, 99, 3, 20, 99, 3, 20, 99), 
#'   run=c(1, 1, 1, 2, 2, 2, 3, 3, 3))
#' )
#'
#' # Create an fMRI dataset with 1 scan and a mask
#' dset <- fmri_dataset("scan1.nii", mask="mask.nii", TR=2, 
#'   run_length=300, 
#'   event_table=data.frame(onsets=c(3, 20, 99), run=rep(1, 3))
#' )
#' 
#' # Create an fMRI dataset with a backend
#' backend <- nifti_backend(c("scan1.nii", "scan2.nii"), mask_source="mask.nii")
#' dset <- fmri_dataset(backend, TR=2, run_length=c(150, 150))
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
    assert_that(is.character(mask), msg="'mask' should be the file name of the binary mask file")
    mode <- match.arg(mode)
    
    maskfile <- paste0(base_path, "/", mask)
    scan_files <- paste0(base_path, "/", scans)
    
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
  
  frame <- sampling_frame(run_length, TR)
  
  # Get dimensions to validate run_length
  dims <- backend_get_dims(backend)
  assert_that(sum(run_length) == dims$time,
              msg = sprintf("Sum of run_length (%d) must equal total time points (%d)", 
                          sum(run_length), dims$time))
  
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