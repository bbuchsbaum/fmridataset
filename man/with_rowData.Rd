% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset_constructors.R
\name{with_rowData}
\alias{with_rowData}
\title{Attach rowData metadata to a DelayedMatrix}
\usage{
with_rowData(x, rowData)
}
\arguments{
\item{x}{A DelayedMatrix}

\item{rowData}{A data.frame of row-wise metadata}
}
\value{
\code{x} with \code{rowData} attribute set
}
\description{
Helper for reattaching metadata after DelayedMatrixStats operations.
}
