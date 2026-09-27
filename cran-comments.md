# CRAN submission notes (draft)

This is the `0.11.0.9000` development candidate. It is not ready to upload.

## Local check, 2026-09-27

On macOS arm64 with R 4.5.1, `R CMD build` followed by `R CMD check
--as-cran` completed with **0 errors, 0 warnings, and 2 notes**. The check ran
tests, examples (including `--run-donttest`), rebuilt all four vignettes,
and built the PDF manual. The isolated library contained `delarr 0.2.0` from
the pinned source revision, `bidser 0.5.2` from its source checkout, and
`fmristore 0.1.0.9000` from the pinned revision. The run used an installed
UTF-8 locale and `RGL_USE_NULL=TRUE`, as CI does on macOS.

The tests reported 3,797 passes and two skips: the already-tracked `fmrigds`
result-metadata contract violation, and a negative dependency test that is
only run when `bidser` is absent or too old.

The incoming-feasibility note reports a first submission, the development
version's large component, the `Remotes` field, and optional packages outside
mainstream repositories. `multidesign` is not currently available from the
listed R-universe repository. The second note says this machine's HTML Tidy
is too old for manual HTML validation.

## Before submission

- Release `delarr >= 0.2.0` to CRAN or Bioconductor. CRAN currently offers
  `delarr 0.1.0`, which cannot satisfy this package's hard dependency.
- Cut a release version greater than `0.11.0.9000` (for example `0.11.1`),
  because downstream packages already require that development version;
  `0.11.0` would not satisfy their bounds. Remove development-only `Remotes`
  and recheck the resulting tarball. Resolve distribution for optional
  packages, particularly `multidesign`.
- Run the final tarball on the supported hosted platforms and update these
  notes with the exact results. The local macOS check is not cross-platform
  evidence.
