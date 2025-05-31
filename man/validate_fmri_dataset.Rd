\name{validate_fmri_dataset}
\alias{validate_fmri_dataset}
\alias{is.fmri_dataset}
\title{Validate fMRI Dataset Objects}
\description{
Comprehensive validation function that checks internal consistency
of an \code{fmri_dataset} object. Verifies data integrity, dimensional compatibility,
and temporal bounds across all components.
}
\usage{
validate_fmri_dataset(x, check_data_load = FALSE, verbose = FALSE)
is.fmri_dataset(x)
}
\arguments{
\item{x}{An \code{fmri_dataset} object to validate, or any object to test.}
\item{check_data_load}{Logical indicating whether to validate by loading actual data.
  If FALSE (default), performs lightweight checks. If TRUE, loads data to verify
  dimensions match metadata (may be slow for large datasets).}
\item{verbose}{Logical indicating whether to print validation progress (default: FALSE)}
}
\value{
\code{validate_fmri_dataset} returns logical TRUE if all checks pass, 
or throws informative error messages if validation fails.

\code{is.fmri_dataset} returns logical indicating if the object is an fmri_dataset.
}
\details{
\strong{Validation Checks Performed:}
\enumerate{
  \item \strong{Object Structure}: Validates fmri_dataset class and required fields
  \item \strong{Sampling Frame}: Checks sampling_frame object consistency
  \item \strong{Data Sources}: Validates image and mask source integrity
  \item \strong{Dimensional Consistency}: Verifies run_lengths match data dimensions
  \item \strong{Mask Compatibility}: Ensures mask dimensions match image dimensions
  \item \strong{Event Table Bounds}: Validates event onsets/durations within time bounds
  \item \strong{Censor Vector}: Checks censor_vector length matches total timepoints
  \item \strong{Metadata Integrity}: Validates metadata consistency
}

\strong{Performance Notes:}
\itemize{
\item With \code{check_data_load = FALSE}: Fast validation using metadata only
\item With \code{check_data_load = TRUE}: Complete validation including data loading
}
}
\examples{
\dontrun{
# Create a dataset
dataset <- fmri_dataset_create(
  images = matrix(rnorm(1000), 100, 10),
  TR = 2.0,
  run_lengths = c(50, 50)
)

# Quick validation (metadata only)
validate_fmri_dataset(dataset)

# Complete validation (loads data)
validate_fmri_dataset(dataset, check_data_load = TRUE, verbose = TRUE)

# Check if object is fmri_dataset
is.fmri_dataset(dataset)  # TRUE
is.fmri_dataset(matrix(1:10))  # FALSE
}
}
\seealso{
\code{\link{fmri_dataset_create}}, \code{\link{summary.fmri_dataset}}
}
\author{Bradley Buchsbaum}
\keyword{validation} 