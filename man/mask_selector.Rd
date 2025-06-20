% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/series_selector.R
\name{mask_selector}
\alias{mask_selector}
\title{Mask-based Series Selector}
\usage{
mask_selector(mask)
}
\arguments{
\item{mask}{A logical vector matching the dataset's mask length, or a 3D logical array}
}
\value{
An object of class \code{mask_selector}
}
\description{
Select voxels that are TRUE in a binary mask.
}
\examples{
\dontrun{
# Using a logical vector
mask_vec <- backend_get_mask(dataset$backend)
sel <- mask_selector(mask_vec > 0.5)
}
}
