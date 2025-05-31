\name{sampling_frame}
\alias{sampling_frame}
\alias{is.sampling_frame}
\alias{samples}
\alias{global_onsets}
\alias{blockids}
\alias{blocklens}
\alias{n_timepoints}
\alias{n_runs}
\alias{get_TR}
\alias{get_run_lengths}
\alias{get_total_duration}
\alias{get_run_duration}
\alias{samples.sampling_frame}
\alias{global_onsets.sampling_frame}
\alias{blockids.sampling_frame}
\alias{blocklens.sampling_frame}
\alias{n_timepoints.sampling_frame}
\alias{n_runs.sampling_frame}
\alias{get_TR.sampling_frame}
\alias{get_run_lengths.sampling_frame}
\alias{get_total_duration.sampling_frame}
\alias{get_run_duration.sampling_frame}
\title{Sampling Frame for fMRI Temporal Structure}
\description{
Creates and manipulates sampling frame objects that represent the temporal
structure of fMRI datasets. A sampling frame encapsulates run lengths,
repetition time (TR), and provides derived temporal properties.
}
\usage{
sampling_frame(TR, run_lengths, start_time = 0, precision = 1e-6)
is.sampling_frame(x)

\method{samples}{sampling_frame}(x, ...)
\method{global_onsets}{sampling_frame}(x, ...)
\method{blockids}{sampling_frame}(x, ...)
\method{blocklens}{sampling_frame}(x, ...)
\method{n_timepoints}{sampling_frame}(x, ...)
\method{n_runs}{sampling_frame}(x, ...)
\method{get_TR}{sampling_frame}(x, ...)
\method{get_run_lengths}{sampling_frame}(x, ...)
\method{get_total_duration}{sampling_frame}(x, ...)
\method{get_run_duration}{sampling_frame}(x, run_id = NULL, ...)
}
\arguments{
\item{TR}{Numeric repetition time in seconds. Can be a single value (same TR for all runs) 
  or vector with one value per run.}
\item{run_lengths}{Integer vector specifying the number of timepoints in each run.}
\item{start_time}{Numeric offset for the first sample (default: 0).}
\item{precision}{Numeric precision for time calculations (default: 1e-6).}
\item{x}{A \code{sampling_frame} object or object to test.}
\item{run_id}{Integer vector specifying which run(s) to get duration for. 
  If NULL (default), returns duration for all runs.}
\item{...}{Additional arguments (ignored).}
}
\value{
\code{sampling_frame} returns a \code{sampling_frame} object with components:
\item{TR}{Vector of TR values (one per run)}
\item{blocklens}{Vector of run lengths in timepoints}
\item{n_runs}{Number of runs}
\item{total_timepoints}{Total number of timepoints across all runs}
\item{start_time}{Start time offset}
\item{precision}{Time precision}

The accessor methods return:
\item{samples}{Vector of sample indices (1-based)}
\item{global_onsets}{Vector of global onset times in seconds}
\item{blockids}{Vector of run IDs for each timepoint}
\item{blocklens}{Vector of run lengths}
\item{n_timepoints}{Total number of timepoints}
\item{n_runs}{Number of runs}
\item{get_TR}{TR value(s)}
\item{get_run_lengths}{Vector of run lengths}
\item{get_total_duration}{Total duration in seconds}
\item{get_run_duration}{Duration(s) for specified run(s)}
}
\details{
The sampling frame provides a comprehensive representation of fMRI temporal
structure. It supports:

\itemize{
\item Multiple runs with different lengths
\item Per-run TR values (for multi-echo or variable TR sequences)
\item Precise temporal calculations
\item Integration with event timing
\item Backwards compatibility with fmrireg
}

Global onset times are calculated as cumulative time from the start,
accounting for different TRs and run boundaries.
}
\examples{
\dontrun{
# Simple single-run sampling frame
sf <- sampling_frame(TR = 2.0, run_lengths = 100)
print(sf)

# Multi-run with same TR
sf_multi <- sampling_frame(TR = 2.5, run_lengths = c(180, 180, 160))

# Multi-run with different TRs
sf_varied <- sampling_frame(
  TR = c(2.0, 2.5, 2.0), 
  run_lengths = c(100, 80, 100)
)

# Access temporal properties
n_timepoints(sf_multi)  # Total timepoints
n_runs(sf_multi)        # Number of runs
get_TR(sf_multi)        # TR values
get_total_duration(sf_multi)  # Total duration in seconds

# Get run-specific information
get_run_duration(sf_multi, run_id = 2)  # Duration of run 2

# Access sample-level information
onsets <- global_onsets(sf_multi)  # Onset time for each sample
runs <- blockids(sf_multi)         # Run ID for each sample
}
}
\seealso{
\code{\link{fmri_dataset_create}}, \code{\link{get_sampling_frame}}
}
\author{Bradley Buchsbaum}
\keyword{temporal} 