\name{bids_facade_phase2}
\alias{assess_quality}
\alias{discover.bids_facade}
\alias{print.bids_discovery_enhanced}
\alias{assess_quality.bids_facade}
\alias{print.bids_quality_report}
\title{Enhanced BIDS Facade}
\description{
Adds quality assessment helpers and richer discovery output on top of the
phase~1 facade.
}
\usage{
assess_quality(x, ...)
}
\arguments{
\item{x}{Object to assess. Additional parameters depend on the method.}
\item{...}{Further arguments passed to underlying functions.}
}
\value{
A \code{bids_quality_report} object containing any available quality metrics
and confound information.
}
\seealso{\code{\link{bids}} and \code{\link{discover}}.}

