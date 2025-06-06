% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/storage_backend.R
\name{backend_open}
\alias{backend_open}
\title{Open a Storage Backend}
\usage{
backend_open(backend)
}
\arguments{
\item{backend}{A storage backend object}
}
\value{
The backend object (possibly modified with state)
}
\description{
Opens a storage backend and acquires any necessary resources (e.g., file handles).
Stateless backends can implement this as a no-op.
}
\keyword{internal}
