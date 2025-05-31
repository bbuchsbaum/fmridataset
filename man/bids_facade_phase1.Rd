\name{bids_facade_phase1}
\alias{bids}
\alias{discover}
\alias{print.bids_facade}
\alias{discover.bids_facade}
\alias{print.bids_discovery_simple}
\alias{as.fmri_dataset.bids_facade}
\title{Minimal Elegant BIDS Facade}
\description{
Functions for opening and querying BIDS datasets using a lightweight wrapper
around the \pkg{bidser} package.
}
\usage{
bids(path, ...)

discover(x, ...)
}
\arguments{
\item{path}{Path to a BIDS dataset.}
\item{x}{Object on which to perform discovery.}
\item{...}{Additional arguments passed to methods.}
}
\value{
\code{bids} returns an object of class \code{bids_facade}.\cr
\code{discover} returns a simple list describing the dataset.
}
\seealso{\code{\link{assess_quality}} for quality assessment utilities.}

