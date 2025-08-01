% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/series_selector.R
\name{roi_selector}
\alias{roi_selector}
\title{ROI-based Series Selector}
\usage{
roi_selector(roi)
}
\arguments{
\item{roi}{A 3D array, ROIVol, LogicalNeuroVol, or similar mask object}
}
\value{
An object of class \code{roi_selector}
}
\description{
Select voxels within a region of interest (ROI) volume or mask.
}
\examples{
\dontrun{
# Using a binary mask
mask <- array(FALSE, dim = c(64, 64, 30))
mask[30:40, 30:40, 15:20] <- TRUE
sel <- roi_selector(mask)
}
}
