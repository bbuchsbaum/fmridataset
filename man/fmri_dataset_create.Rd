\name{fmri_dataset_create}
\alias{fmri_dataset_create}
\title{Create fMRI Dataset Objects}
\description{
Creates a unified \code{fmri_dataset} object from various data sources including
matrices, file paths, or pre-loaded objects. The resulting object provides
a consistent interface for accessing fMRI data regardless of the source.
}
\usage{
fmri_dataset_create(
  images, 
  mask = NULL, 
  TR, 
  run_lengths, 
  event_table = NULL,
  censor_vector = NULL, 
  base_path = ".", 
  image_mode = "normal",
  preload_data = FALSE, 
  temporal_zscore = FALSE, 
  voxelwise_detrend = FALSE,
  metadata = list(), 
  ...
)
}
\arguments{
\item{images}{Primary image data. Can be a matrix (time x voxels), 
  character vector of file paths, or list of pre-loaded objects.}
\item{mask}{Spatial mask. Can be a file path, logical vector/matrix, 
  or pre-loaded mask object. If NULL, all voxels are included.}
\item{TR}{Numeric repetition time in seconds.}
\item{run_lengths}{Integer vector specifying the number of timepoints in each run.}
\item{event_table}{Data frame or path to TSV file containing experimental events. 
  Should include 'onset' and 'duration' columns.}
\item{censor_vector}{Logical or numeric vector indicating which timepoints to keep 
  (TRUE/1) or exclude (FALSE/0).}
\item{base_path}{Base directory path for resolving relative file paths.}
\item{image_mode}{Loading mode for file-based images. One of "normal", "bigvec", "mmap", "filebacked".}
\item{preload_data}{Logical indicating whether to load data immediately into memory.}
\item{temporal_zscore}{Logical indicating whether to apply temporal z-scoring preprocessing.}
\item{voxelwise_detrend}{Logical indicating whether to apply voxelwise linear detrending.}
\item{metadata}{Named list of additional metadata to store with the dataset.}
\item{...}{Additional arguments for future expansion.}
}
\value{
An \code{fmri_dataset} object containing:
\item{image_paths, image_objects, or image_matrix}{Data source (only one populated)}
\item{mask_path, mask_object, or mask_vector}{Mask source (if provided)}
\item{sampling_frame}{Temporal structure information}
\item{event_table}{Experimental events (if provided)}
\item{censor_vector}{Censoring information (if provided)}
\item{metadata}{Dataset metadata and options}
\item{data_cache}{Environment for lazy loading cache}
}
\details{
The function automatically determines the dataset type based on the input:
\itemize{
\item \strong{matrix}: In-memory data matrix
\item \strong{file_vec}: Vector of NIfTI file paths
\item \strong{memory_vec}: List of pre-loaded NeuroVec objects
\item \strong{bids_file}: BIDS project with file-based access
\item \strong{bids_mem}: BIDS project with pre-loaded data
}

Data is loaded lazily by default, meaning actual file reading occurs only when
\code{get_data_matrix()} is called. This enables efficient memory usage for
large datasets.

The sampling frame encapsulates temporal structure including TR, run lengths,
and derived properties. Event tables should follow BIDS format with 'onset',
'duration', and 'trial_type' columns at minimum.
}
\examples{
\dontrun{
# From matrix
data_matrix <- matrix(rnorm(1000), nrow = 100, ncol = 10)
dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(50, 50)
)

# From files
file_paths <- c("run1.nii.gz", "run2.nii.gz")
dataset <- fmri_dataset_create(
  images = file_paths,
  mask = "mask.nii.gz",
  TR = 2.5,
  run_lengths = c(180, 180)
)

# With events and preprocessing
events <- data.frame(
  onset = c(10, 30, 50),
  duration = c(2, 2, 2),
  trial_type = c("A", "B", "A")
)

dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(50, 50),
  event_table = events,
  temporal_zscore = TRUE
)
}
}
\seealso{
\code{\link{as.fmri_dataset}}, \code{\link{get_data_matrix}}, 
\code{\link{validate_fmri_dataset}}, \code{\link{sampling_frame}}
}
\author{Bradley Buchsbaum}
\keyword{datasets} 