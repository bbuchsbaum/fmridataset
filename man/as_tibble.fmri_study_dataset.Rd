% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/study_dataset_access.R
\name{as_tibble.fmri_study_dataset}
\alias{as_tibble.fmri_study_dataset}
\title{Convert fmri_study_dataset to a tibble or DelayedMatrix}
\usage{
\method{as_tibble}{fmri_study_dataset}(x, materialise = FALSE, ...)
}
\arguments{
\item{x}{An \code{fmri_study_dataset} object}

\item{materialise}{Logical; return a materialised tibble? Default \code{FALSE}.}

\item{...}{Additional arguments (unused)}
}
\value{
Either a \code{DelayedMatrix} with metadata attributes or a tibble
when \code{materialise = TRUE}.
}
\description{
Primary data access method for study-level datasets. By default this
returns a lazy \code{DelayedMatrix} with row-wise metadata attached. When
\code{materialise = TRUE}, the data matrix is materialised and returned as
a tibble with metadata columns prepended.
}
