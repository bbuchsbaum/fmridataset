## R CMD check results

0 errors | 1 warning | 1 note

## Test environments

* local macOS (aarch64-apple-darwin20), R 4.5.1

## Warnings

### Strong dependencies not in mainstream repositories

This package depends on `delarr`, `fmrihrf`, and `neuroim2` which are not yet
on CRAN. These packages are being prepared for CRAN submission. This package
will only be submitted to CRAN after its dependencies are accepted.

### Suggests not in mainstream repositories

`bidser` and `fmristore` are optional dependencies from GitHub. They are used
only for advanced features (BIDS integration and HDF5 fMRI storage) and are
properly wrapped with `requireNamespace()` checks.

## Notes

### New submission

This is the first submission of this package to CRAN.

### HTML validation

The "tidy doesn't look like recent enough HTML Tidy" note is a tool
availability issue on the test system, not a package issue.

## Downstream dependencies

There are currently no downstream dependencies for this package on CRAN.
