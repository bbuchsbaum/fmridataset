\name{data_chunks}
\alias{data_chunks}
\alias{data_chunks.fmri_dataset}
\alias{fmri_data_chunk}
\alias{is.fmri_data_chunk}
\title{Data Chunking for fMRI Datasets}
\description{
Creates iterators for processing fMRI data in chunks, enabling efficient
parallel processing and memory management for large datasets.
}
\usage{
\method{data_chunks}{fmri_dataset}(
  x, 
  nchunks = 4, 
  by = c("voxel", "timepoint"), 
  runwise = FALSE,
  apply_preprocessing = TRUE, 
  ...
)

fmri_data_chunk(data, voxel_indices, chunk_num, total_chunks, ...)
is.fmri_data_chunk(x)
}
\arguments{
\item{x}{An \code{fmri_dataset} object or object to test.}
\item{nchunks}{Integer number of chunks to create. Ignored if \code{runwise = TRUE}.}
\item{by}{Character string specifying chunking dimension: "voxel" (default) or "timepoint".}
\item{runwise}{Logical indicating whether to create one chunk per run (default: FALSE).}
\item{apply_preprocessing}{Logical indicating whether to apply preprocessing 
  (temporal z-score, detrending) to chunk data.}
\item{data}{Matrix of data for this chunk (timepoints x voxels).}
\item{voxel_indices}{Integer vector of voxel indices in this chunk.}
\item{chunk_num}{Integer chunk number (1-based).}
\item{total_chunks}{Integer total number of chunks.}
\item{...}{Additional arguments for future expansion.}
}
\value{
\code{data_chunks} returns an \code{fmri_chunk_iterator} object that can be:
\itemize{
\item Iterated using \code{for} loops
\item Used with \code{foreach} for parallel processing
\item Manually stepped through with \code{iter$nextElem()}
}

Each iteration yields an \code{fmri_data_chunk} object with components:
\item{data}{Matrix of chunk data (timepoints x voxels)}
\item{voxel_indices}{Voxel indices in the original dataset}
\item{chunk_num}{Current chunk number}
\item{total_chunks}{Total number of chunks}
\item{run_indices}{Run information (if applicable)}

\code{fmri_data_chunk} creates a chunk object manually.
\code{is.fmri_data_chunk} returns logical indicating if object is a data chunk.
}
\details{
Data chunking enables efficient processing of large fMRI datasets by:

\itemize{
\item \strong{Memory Management}: Process data in smaller pieces
\item \strong{Parallel Processing}: Compatible with \code{foreach} and similar frameworks
\item \strong{Flexible Chunking}: By voxels, timepoints, or runs
\item \strong{Preprocessing Integration}: Applies dataset preprocessing options
}

\strong{Chunking Strategies:}
\itemize{
\item \strong{by = "voxel"}: Splits voxel dimension for spatial analyses
\item \strong{by = "timepoint"}: Splits temporal dimension for time-based analyses  
\item \strong{runwise = TRUE}: Creates one chunk per run
}

The iterator is compatible with standard R iteration patterns and 
parallel processing frameworks.
}
\examples{
\dontrun{
# Create dataset
dataset <- fmri_dataset_create(
  images = matrix(rnorm(10000), 100, 100),
  TR = 2.0,
  run_lengths = c(50, 50)
)

# Chunk by voxels for spatial analysis
voxel_chunks <- data_chunks(dataset, nchunks = 4)

# Process chunks sequentially
for (chunk in voxel_chunks) {
  cat("Processing chunk", chunk$chunk_num, 
      "with", ncol(chunk$data), "voxels\n")
  
  # Your analysis here
  result <- cor(chunk$data)
}

# Chunk by runs
run_chunks <- data_chunks(dataset, runwise = TRUE)

# Parallel processing with foreach
if (requireNamespace("foreach", quietly = TRUE)) {
  library(foreach)
  
  results <- foreach(chunk = voxel_chunks, .combine = 'c') \%do\% {
    # Compute mean activation for this chunk
    mean(chunk$data)
  }
}

# Manual iteration
iter <- data_chunks(dataset, nchunks = 3)
chunk1 <- iter$nextElem()
chunk2 <- iter$nextElem()
}
}
\seealso{
\code{\link{fmri_dataset_create}}, \code{\link{get_data_matrix}}, 
\code{\link{preload_data}}
}
\author{Bradley Buchsbaum}
\keyword{iteration} 