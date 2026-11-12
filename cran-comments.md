## Pre-submission checklist

These items must be resolved before this file is used for a submission:

* `delarr` is now in Suggests. CRAN has 0.1.0; `as_delarr()` needs 0.2.0,
  which is on r-universe. Tests, the example, and the vignette chunk that use
  it are conditional on that version.
* Confirm that `neuroim2` and `fmrihrf` are current on CRAN at the required
  versions.
* Remove the `Remotes:` field from DESCRIPTION and set a release version
  (no `.9000` component).
* Re-run `R CMD check --as-cran` with a LaTeX toolchain and network access,
  plus win-builder and R-hub.

## R CMD check results

Last local run: R 4.3.3, Ubuntu 24.04, `--as-cran --no-manual
--run-donttest`, without LaTeX and without `multidesign`, `neurosurf`, and
`zarr` (not installable on R 4.3). Examples,
`--run-donttest` examples, tests, and vignettes all passed. The only
findings were:

* NOTE: `Remotes` field, development version number (see the checklist).
* NOTE: Suggests not available on CRAN (see below).
* ERROR/WARNING: PDF manual not built because `pdflatex` was not installed.
  This must be re-checked on a machine with LaTeX.

## Suggests not in mainstream repositories

`fmristore`, `fmrigds`, `fmrilatent`, `multidesign`, `neuroatlas`, and
`neurosurf`, and the required version of `delarr`, are available from
<https://bbuchsbaum.r-universe.dev>, which DESCRIPTION declares in
`Additional_repositories`. Every use is conditional on `requireNamespace()`:
tests skip, examples and vignette chunks are guarded, and functions that need
one of them (`write_frame()` and `open_frame()` need `fmristore`) fail with a
structured error that names the missing package.

## Downstream dependencies

There are currently no downstream dependencies for this package on CRAN.
