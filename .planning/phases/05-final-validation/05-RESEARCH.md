# Phase 5: Final Validation - Research

**Researched:** 2026-01-22
**Domain:** CRAN compliance and R package quality assurance
**Confidence:** HIGH

## Summary

Phase 5 focuses on achieving zero errors, zero warnings, and zero notes from `R CMD check --as-cran` to ensure CRAN submission readiness. Current package state reveals specific fixable issues rather than fundamental problems:

**Current Status**: The package currently produces 1 ERROR, 2 WARNINGs, 1 NOTE when running `R CMD check --as-cran`:
- ERROR: Tests fail due to missing blosc package (optional zarr codec dependency)
- WARNING 1: CRAN incoming feasibility - Remotes field not allowed, non-CRAN dependencies
- WARNING 2: Unstated dependencies in tests (DelayedArray, DelayedMatrixStats, rhdf5)
- NOTE: HTML validation tool issue (cosmetic, unavoidable on some systems)

**Primary recommendation:** Fix issues in dependency order: (1) remove Remotes field and unused test references, (2) add missing packages to Suggests with proper Additional_repositories field, (3) add conditional skipping to tests requiring blosc, (4) verify clean install and vignette builds.

## Standard Stack

The established tools for CRAN compliance checking:

### Core
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| R CMD check --as-cran | Built-in | CRAN compliance validation | Official CRAN requirement |
| devtools::check() | Latest | R-friendly wrapper for R CMD check | Industry standard development tool |
| rcmdcheck::rcmdcheck() | Latest | Programmatic check execution | Captures results for analysis |

### Supporting
| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| devtools::release() | Latest | Pre-submission verification | Before CRAN submission |
| rhub::check_for_cran() | Latest | Multi-platform testing | To verify cross-platform compatibility |
| goodpractice::gp() | Latest | Code quality checks | Optional quality improvement |

### Testing Fresh Installation
| Approach | Purpose | When to Use |
|----------|---------|-------------|
| R CMD INSTALL --preclean | Clean local install | Final verification before submission |
| Fresh R session + install.packages() | User experience simulation | Verify installation from tarball |
| Docker/container testing | Platform isolation | Optional for complex dependencies |

**Installation:**
```bash
# Core tools (already available)
R CMD check --as-cran package.tar.gz

# Optional verification tools
install.packages(c("devtools", "rcmdcheck", "rhub", "goodpractice"))
```

## Architecture Patterns

### CRAN Submission Workflow
```
1. Fix all ERRORs and WARNINGs
   ├── Update DESCRIPTION (remove Remotes, add Additional_repositories)
   ├── Fix unstated dependencies
   └── Fix test failures

2. Eliminate NOTEs (zero tolerance per user requirement)
   ├── Add conditional test skipping
   ├── Document unavoidable NOTEs in cran-comments.md
   └── Fix file size issues if any

3. Verify vignettes and examples
   ├── Build vignettes successfully
   └── Run all examples without errors

4. Test fresh installation
   ├── Install from tarball in clean session
   └── Verify package loads and functions work

5. Create/update cran-comments.md
   └── Document any unavoidable issues for CRAN reviewers
```

### Pattern 1: Handling Optional Dependencies in DESCRIPTION
**What:** Proper declaration of optional dependencies from non-CRAN repositories
**When to use:** When package uses optional functionality from GitHub, Bioconductor, or other repos

**CRAN Policy (HIGH confidence):**
- Remotes field is NOT allowed in CRAN submissions
- Strong dependencies (Imports/Depends) must be from CRAN or Bioconductor only
- Optional dependencies (Suggests) from non-CRAN repos must use Additional_repositories field
- All packages must be used conditionally with requireNamespace() checks

**Example:**
```r
# DESCRIPTION file structure for fmridataset

# WRONG - Remotes field not allowed on CRAN
Remotes:
    bbuchsbaum/delarr,
    bbuchsbaum/fmrihrf

# RIGHT - Use Additional_repositories for Suggests packages only
# Note: Strong dependencies (delarr, fmrihrf) must be on CRAN first
Additional_repositories: https://bbuchsbaum.github.io/drat/

Suggests:
    bidser,
    fmristore,
    blosc,
    DelayedArray,
    DelayedMatrixStats
```

**Source:** [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html) states: "For packages in the Suggests or Enhances fields, [...] if these are not from the mainstream repositories [...] where to obtain them at a repository should be specified in an 'Additional_repositories' field of the DESCRIPTION file."

### Pattern 2: Conditional Test Skipping for Optional Dependencies
**What:** Tests that use optional packages must skip gracefully when packages unavailable
**When to use:** Always, for any package in Suggests field

**Example:**
```r
# Source: testthat best practices

test_that("zarr backend works with blosc codec", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")  # ADD THIS - currently missing

  # Test code using zarr with blosc codec
  arr <- array(rnorm(64), dim = c(4, 4, 4, 1))
  tmp_dir <- tempfile()
  z <- zarr::as_zarr(arr, location = tmp_dir)
  # ...
})
```

**Why it matters:** R CMD check runs tests with `_R_CHECK_FORCE_SUGGESTS_=false`, so tests cannot assume Suggests packages are available. Tests must skip gracefully or they cause ERRORs.

### Pattern 3: Handling Bioconductor Dependencies in Tests
**What:** Tests using Bioconductor packages (DelayedArray, rhdf5) via ::: operators
**When to use:** When testing optional functionality that uses Bioconductor packages

**Current Issue:**
```r
# WARNING from R CMD check:
# '::' or ':::' imports not declared from:
#   'DelayedArray' 'DelayedMatrixStats' 'rhdf5'
```

**Solution:**
1. Add packages to Suggests field in DESCRIPTION
2. Tests already use skip_if_not_installed() - this is correct
3. Remove rhdf5 references per user requirement (hdf5r is the correct dependency)

**Example fix:**
```r
# DESCRIPTION
Suggests:
    DelayedArray,
    DelayedMatrixStats,
    hdf5r,  # NOT rhdf5 - user clarified hdf5r is correct
    blosc

# tests/testthat/test_*.R
test_that("delayed array conversion works", {
  skip_if_not_installed("DelayedArray")
  # Test code using DelayedArray::function()
})
```

### Pattern 4: Fresh Installation Testing
**What:** Verify package installs and loads in clean R session
**When to use:** Final verification before submission

**Example:**
```bash
# Build tarball
R CMD build .

# Test installation in clean session
R --vanilla -e "install.packages('fmridataset_0.8.9.tar.gz', repos = NULL, type = 'source')"

# Verify loading
R --vanilla -e "library(fmridataset); print(packageVersion('fmridataset'))"

# Test basic functionality
R --vanilla -e "library(fmridataset); mat <- matrix(rnorm(100), 10, 10); ds <- fmri_mem_dataset(mat, TR = 2, run_length = 10)"
```

### Anti-Patterns to Avoid

- **Using Remotes field for CRAN submission:** Field is not recognized by CRAN and causes WARNING. Use Additional_repositories instead for Suggests packages, and ensure strong dependencies are on CRAN.
- **Assuming Suggests packages are available in tests:** Tests must use skip_if_not_installed() for every optional package.
- **Including rhdf5 as dependency:** Per user clarification, hdf5r is the correct HDF5 library for this package, not rhdf5 (Bioconductor).
- **Ignoring NOTEs:** User requirement is zero notes - must fix or document all NOTEs in cran-comments.md.
- **Testing only with all packages installed:** CRAN tests with `_R_CHECK_FORCE_SUGGESTS_=false`, so must verify tests pass with minimal dependencies.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-platform CRAN testing | Manual VM setup | rhub::check_for_cran() | Tests on CRAN's actual infrastructure |
| Pre-submission checks | Manual checklist | devtools::release() | Automates 20+ verification steps |
| Capturing check results | Parse log files | rcmdcheck::rcmdcheck() | Structured result objects |
| Dependency conditional checks | Manual requireNamespace() | testthat::skip_if_not_installed() | Standard, recognized by CRAN |
| Package size analysis | Manual file listing | tools::checkRd(), R CMD check output | Built-in CRAN checks |

**Key insight:** CRAN has specific expectations about how optional dependencies are handled. Using standard functions like skip_if_not_installed() signals to reviewers that you understand R package conventions.

## Common Pitfalls

### Pitfall 1: Remotes Field in DESCRIPTION
**What goes wrong:** Package gets WARNING on CRAN incoming feasibility check: "Unknown, possibly misspelled, fields in DESCRIPTION: 'Remotes'"
**Why it happens:** Remotes is a devtools-specific field not recognized by CRAN. It's for development convenience, not production packages.
**How to avoid:**
- Remove Remotes field before CRAN submission
- Ensure all strong dependencies (Imports/Depends) are on CRAN or Bioconductor
- Use Additional_repositories for optional dependencies (Suggests) from non-CRAN repos
- Consider setting up a drat repository for GitHub packages
**Warning signs:** `R CMD check --as-cran` shows "Unknown, possibly misspelled, fields"

**Reality check:** fmridataset cannot be submitted to CRAN until delarr, fmrihrf, and neuroim2 are also on CRAN, as these are strong dependencies. The Additional_repositories field only works for Suggests packages, not Imports.

### Pitfall 2: Unstated Dependencies in Tests
**What goes wrong:** `R CMD check --as-cran` produces WARNING: "'::' or ':::' imports not declared from: 'PackageName'"
**Why it happens:** Tests use package::function() or package:::function() but package is not declared in DESCRIPTION
**How to avoid:**
- Every package used in tests/ must be in Imports or Suggests
- Use testthat::skip_if_not_installed("package") for Suggests packages
- Verify tests pass with `_R_CHECK_FORCE_SUGGESTS_=false`
**Warning signs:**
```
* checking for unstated dependencies in 'tests' ... WARNING
'library' or 'require' call not declared from: 'rhdf5'
```

**Current issues in fmridataset:**
- DelayedArray: Add to Suggests (tests already skip properly)
- DelayedMatrixStats: Add to Suggests (tests already skip properly)
- rhdf5: REMOVE from tests - should be hdf5r per user requirement
- blosc: Add to Suggests, add skip_if_not_installed() to zarr tests

### Pitfall 3: Tests Requiring Optional Codec Dependencies
**What goes wrong:** Tests fail with "Must install package 'blosc' for this functionality" even though zarr is installed
**Why it happens:** zarr::as_zarr() defaults to using blosc codec, which requires separate blosc package. Tests skip for zarr but not for blosc.
**How to avoid:**
1. Add blosc to Suggests field
2. Add skip_if_not_installed("blosc") to all tests using zarr::as_zarr()
3. Alternatively, create test arrays without blosc codec (less preferred)
**Warning signs:** Test failures in zarr_backend.R and zarr_dataset_constructor.R
**Example fix:**
```r
test_that("zarr backend works", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")  # ADD THIS
  # ... rest of test
})
```

### Pitfall 4: Non-CRAN Strong Dependencies
**What goes wrong:** WARNING: "Strong dependencies not in mainstream repositories: delarr"
**Why it happens:** Package lists non-CRAN package in Imports field
**How to avoid:**
- Strong dependencies (Imports, Depends, LinkingTo) must be on CRAN or Bioconductor
- Cannot use Additional_repositories for strong dependencies
- Must submit dependencies to CRAN first, or move to Suggests
**Warning signs:** "Strong dependencies not in mainstream repositories"
**Reality:** This is a blocker - fmridataset cannot be submitted until delarr is on CRAN

### Pitfall 5: Confusing Development vs Submission Configuration
**What goes wrong:** Package works perfectly in development but fails CRAN checks
**Why it happens:** Development uses Remotes field and assumes all Suggests are installed
**How to avoid:**
- Test with `_R_CHECK_FORCE_SUGGESTS_=false` regularly
- Run `R CMD check --as-cran` before every version bump
- Maintain both .Rbuildignore and careful DESCRIPTION management
**Warning signs:** Works with devtools::load_all() but fails R CMD check

### Pitfall 6: Assuming Zero Notes is Optional
**What goes wrong:** Submitter thinks NOTEs are acceptable, but user/CRAN may not
**Why it happens:** Some NOTEs are unavoidable (first submission, non-ASCII data), others are fixable
**How to avoid:**
- User requirement: zero tolerance for notes
- Document truly unavoidable NOTEs in cran-comments.md
- Fix all avoidable NOTEs (size, file patterns, etc.)
**Warning signs:** Any NOTE in R CMD check output
**Acceptable NOTEs (document in cran-comments.md):**
- "New submission" (first-time only)
- "Possibly misspelled words in DESCRIPTION" (if words are correct domain terms)
- "HTML validation: tidy not recent enough" (tool availability issue)

## Code Examples

Verified patterns from official sources and current check results:

### Fixing DESCRIPTION for CRAN Submission
```r
# Source: CRAN Repository Policy
# URL: https://cran.r-project.org/web/packages/policies.html

# BEFORE (current state):
Package: fmridataset
Imports:
    delarr,    # NOT ON CRAN - blocks submission
    fmrihrf    # NOT ON CRAN - blocks submission
Suggests:
    bidser,
    fmristore,
    # ... missing: blosc, DelayedArray, DelayedMatrixStats
Remotes:
    bbuchsbaum/delarr,
    bbuchsbaum/fmrihrf,
    bbuchsbaum/bidser,
    bbuchsbaum/fmristore

# AFTER (CRAN-ready, once dependencies are on CRAN):
Package: fmridataset
Imports:
    delarr,    # Must be on CRAN before submission
    fmrihrf    # Must be on CRAN before submission
Suggests:
    bidser,
    blosc,
    DelayedArray,
    DelayedMatrixStats,
    fmristore,
    hdf5r,     # NOT rhdf5
    # ... other suggests
# Additional_repositories: https://your-repo-url.com
# Note: Remotes field removed - not allowed on CRAN
```

### Adding Conditional Skipping to Tests
```r
# Source: testthat documentation
# Current issue: tests/testthat/test_zarr_backend.R line 54

# BEFORE:
test_that("zarr_backend works with local store", {
  skip_if_not_installed("zarr")

  arr <- array(rnorm(8 * 8 * 4 * 10), dim = c(8, 8, 4, 10))
  tmp_dir <- tempfile()
  z <- zarr::as_zarr(arr, location = tmp_dir)  # FAILS: needs blosc
  # ...
})

# AFTER:
test_that("zarr_backend works with local store", {
  skip_if_not_installed("zarr")
  skip_if_not_installed("blosc")  # ADD THIS

  arr <- array(rnorm(8 * 8 * 4 * 10), dim = c(8, 8, 4, 10))
  tmp_dir <- tempfile()
  z <- zarr::as_zarr(arr, location = tmp_dir)  # Now skips gracefully
  # ...
})
```

### Removing rhdf5 References
```r
# Source: User requirement - hdf5r is the correct HDF5 library
# Files to update: tests/test_optional_packages.R

# BEFORE (line 21):
optional_packages <- list(
  # ...
  rhdf5 = "HDF5 support (Bioconductor)",  # WRONG
  # ...
)

# AFTER:
optional_packages <- list(
  # ...
  hdf5r = "HDF5 support (CRAN)",  # CORRECT
  # ...
)

# Also update lines 49, 98 that reference rhdf5
# Change to hdf5r or remove Bioconductor section entirely
```

### Testing Fresh Installation
```bash
# Source: R CMD check best practices
# URL: https://r-pkgs.org/release.html

# Build package
R CMD build .

# Install in fresh session (no devtools)
R --vanilla <<'EOF'
install.packages("fmridataset_0.8.9.tar.gz", repos = NULL, type = "source")
library(fmridataset)
print(packageVersion("fmridataset"))

# Test basic functionality
mat <- matrix(rnorm(100), 10, 10)
ds <- fmri_mem_dataset(mat, TR = 2, run_length = 10)
print(ds)
EOF

# Verify examples run
R --vanilla -e "example(fmri_dataset, package = 'fmridataset')"
```

### Creating cran-comments.md
```markdown
# Source: CRAN submission best practices
# File: cran-comments.md

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local macOS (aarch64-apple-darwin20), R 4.5.1
* win-builder (devel and release)
* R-hub (ubuntu-gcc-release)

## Downstream dependencies

There are currently no downstream dependencies for this package.

## Notes for CRAN reviewers

### Dependencies not on CRAN

This package depends on several packages not yet on CRAN:
- delarr: Delayed array implementation (planned CRAN submission)
- fmrihrf: fMRI hemodynamic response functions (planned CRAN submission)
- neuroim2: Neuroimaging data structures (planned CRAN submission)

We will submit this package only after these dependencies are accepted to CRAN.

### Optional Bioconductor dependencies

The package optionally uses DelayedArray (Bioconductor) for memory-efficient
array operations. This is properly listed in Suggests and used conditionally
with requireNamespace() checks.

### Zarr backend marked EXPERIMENTAL

The zarr backend is marked as experimental due to the zarr package being
relatively new (v0.1.1). This is documented in package documentation.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Remotes field for GitHub deps | Additional_repositories for Suggests only | CRAN policy (longstanding) | Remotes not allowed on CRAN |
| Assume all Suggests available | Conditional usage with skip_if_not_installed() | R 3.x era | Tests must handle missing packages |
| rhdf5 (Bioconductor) | hdf5r (CRAN) | User decision (project-specific) | Avoid Bioconductor strong dependency |
| Manual multi-platform testing | rhub::check_for_cran() | ~2018 | Automated CRAN-like testing |
| Parse check logs manually | rcmdcheck::rcmdcheck() | ~2016 | Programmatic result analysis |

**Deprecated/outdated:**
- Remotes field in DESCRIPTION for CRAN packages: Development-only, remove before submission
- Assuming DelayedArray is available: Now optional, must use conditionally
- rhdf5 for this package: User specified hdf5r is the correct choice

## Open Questions

Things that couldn't be fully resolved:

1. **CRAN submission timeline for dependencies**
   - What we know: delarr, fmrihrf, neuroim2 are strong dependencies not on CRAN
   - What's unclear: When (or if) these will be submitted to CRAN
   - Recommendation: Cannot submit fmridataset until dependencies are on CRAN. Phase 5 can achieve "CRAN-ready" state (0 errors, 0 warnings, 0 notes) but actual submission is blocked by external dependencies.

2. **Additional_repositories URL for GitHub packages**
   - What we know: CRAN allows Additional_repositories field for Suggests packages
   - What's unclear: Whether GitHub URLs are acceptable or if a drat repository is needed
   - Recommendation: If keeping bidser/fmristore in Suggests, set up a drat repository (https://github.com/eddelbuettel/drat) or provide CRAN-like repository URL

3. **blosc package availability on test platforms**
   - What we know: blosc is on CRAN as of September 2025
   - What's unclear: Whether CRAN test systems have blosc installed
   - Recommendation: Add blosc to Suggests and use skip_if_not_installed(). Tests will skip on systems without blosc, which is acceptable for optional functionality.

4. **Multi-platform testing requirements**
   - What we know: Package tested on macOS (aarch64-apple-darwin20)
   - What's unclear: Whether multi-platform testing is required before Phase 5 completion
   - Recommendation: Minimum for "CRAN-ready" is local platform clean. Optional enhancement: test on win-builder and rhub for comprehensive validation.

## Sources

### Primary (HIGH confidence)
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html) - Official CRAN policies on dependencies, Additional_repositories field, testing requirements
- [R CMD check output](fmridataset.Rcheck/00check.log) - Actual check results showing current errors/warnings/notes
- [R Packages (2e) - R CMD check](https://r-pkgs.org/R-CMD-check.html) - Authoritative guide on R CMD check requirements
- [R Packages (2e) - Releasing to CRAN](https://r-pkgs.org/release.html) - CRAN submission workflow and best practices
- [testthat documentation](https://testthat.r-lib.org/) - skip_if_not_installed() usage

### Secondary (MEDIUM confidence)
- [CRAN Package blosc](https://cran.r-project.org/web/packages/blosc/index.html) - blosc availability on CRAN (verified December 2025)
- [CRAN Package zarr](https://cran.r-project.org/web/packages/zarr/index.html) - zarr package information (verified December 2025)
- [R-hub blog: Optimal workflows for package vignettes](https://blog.r-hub.io/2020/06/03/vignettes/) - Vignette build requirements

### Tertiary (LOW confidence - general guidance)
- WebSearch results on Additional_repositories usage - Confirmed general pattern but specifics vary by use case
- Community discussions on CRAN submission - Best practices confirmation but not authoritative

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - R CMD check and devtools are official, well-documented tools
- Architecture: HIGH - CRAN policies are official and stable; check output is factual
- Pitfalls: HIGH - Based on actual check output from fmridataset package
- Code examples: HIGH - Based on official CRAN policy docs and actual package code

**Research date:** 2026-01-22
**Valid until:** 2026-03-22 (60 days - CRAN policies stable, tools mature)

**Critical constraint:** Package CANNOT be submitted to CRAN until delarr, fmrihrf, and neuroim2 dependencies are also on CRAN. Phase 5 can achieve "CRAN-ready" state (0 errors, 0 warnings, 0 notes) but actual submission is blocked by external dependencies per CRAN policy requiring strong dependencies to be in mainstream repositories.

**User requirements incorporated:**
- rhdf5 must be removed - hdf5r is the correct HDF5 dependency (from CONTEXT.md)
- Zero tolerance for notes - must achieve 0 notes, not just 0 errors/warnings (from CONTEXT.md)
- Fix issues immediately as found - don't batch or defer (from CONTEXT.md)
