% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset_constructors.R
\name{latent_dataset}
\alias{latent_dataset}
\title{Create a Latent Dataset Object}
\usage{
latent_dataset(lvec, TR, run_length, event_table = data.frame())
}
\arguments{
\item{lvec}{An instance of class \code{LatentNeuroVec}. (Typically, a \code{LatentNeuroVec} is
created using the \code{fmristore} package.)}

\item{TR}{Repetition time (TR) of the fMRI acquisition.}

\item{run_length}{A numeric vector specifying the length of each run in the dataset.}

\item{event_table}{An optional data frame containing event information. Default is an empty data frame.}
}
\value{
A latent dataset object of class \code{c("latent_dataset", "matrix_dataset", "fmri_dataset", "list")}.
}
\description{
This function creates a latent dataset object, which encapsulates a dimension-reduced
subspace of "latent variables". The dataset is a list containing information about the latent
neuroimaging vector, TR, number of runs, event table, base path, sampling frame, and censor.
}
\examples{
\dontrun{
# Create a matrix with 100 rows and 1000 columns (voxels)
X <- matrix(rnorm(100 * 1000), 100, 1000)
pres <- prcomp(X)
basis <- pres$x[, 1:25]
loadings <- pres$rotation[, 1:25]
offset <- colMeans(X)

# Create a LatentNeuroVec object (requires the fmristore package)
lvec <- fmristore::LatentNeuroVec(basis, loadings,
  neuroim2::NeuroSpace(c(10, 10, 10, 100)),
  mask = rep(TRUE, 1000), offset = offset
)

# Create a latent_dataset
dset <- latent_dataset(lvec, TR = 2, run_length = 100)
}
}
