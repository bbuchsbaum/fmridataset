% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R
\name{data_chunks}
\alias{data_chunks}
\title{Create Data Chunks for Processing}
\usage{
data_chunks(x, nchunks = 1, runwise = FALSE, ...)
}
\arguments{
\item{x}{An fMRI dataset object}

\item{nchunks}{Number of chunks to create (default: 1)}

\item{runwise}{If TRUE, create run-wise chunks (default: FALSE)}

\item{...}{Additional arguments passed to methods}
}
\value{
A chunk iterator object
}
\description{
Generic function to create data chunks for parallel processing from
various fMRI dataset types. Supports different chunking strategies.
}
