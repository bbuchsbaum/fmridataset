# Phase 1: CRAN Quick Wins - Research

**Researched:** 2026-01-22
**Domain:** R package CRAN compliance (R CMD check)
**Confidence:** HIGH

## Summary

This phase focuses on fixing mechanical CRAN compliance issues that prevent R CMD check from running cleanly. Research reveals that CRAN-01 through CRAN-06 are all well-documented, standard R package hygiene issues with established solutions.

The core issues are:
1. **Dependency declarations** - Test and vignette dependencies must be in Suggests field
2. **Exported function calling test helpers** - Architectural antipattern that violates package boundaries
3. **Cross-package documentation links** - Requires package anchors for fmrihrf references
4. **Build ignore patterns** - Planning directory and non-standard files must be excluded

All issues have straightforward fixes following established R package development patterns. No complex refactoring or breaking changes required.

**Primary recommendation:** Fix issues in order CRAN-01, CRAN-02 (most impactful architectural fix), CRAN-03, CRAN-04, CRAN-05, CRAN-06. Each is independent and can be verified incrementally.

## Standard Stack

### Core Tools
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| R CMD check | R >= 4.3.0 | Package validation | Official CRAN validation tool |
| devtools::check() | Latest | Interactive checking | Wrapper around R CMD check with better output |
| roxygen2 | >= 7.3.0 | Documentation generation | Standard for Rd generation and cross-references |
| testthat | >= 3.0.0 | Testing framework | Already in use, has skip_if_not_installed() |
| usethis | Latest | Package scaffolding | Has use_build_ignore() for .Rbuildignore |

### Supporting
| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| rcmdcheck | Latest | Programmatic checking | CI/CD pipelines (future) |
| checkhelper | Latest | Find missing globals | Debugging check issues |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| devtools::check() | R CMD check directly | devtools provides better formatted output |
| roxygen2 | Manual Rd files | roxygen2 is standard, manual Rd error-prone |

**Installation:**
All tools already present in DESCRIPTION or base R.

## Architecture Patterns

### Recommended Package Structure
```
R/
├── exported_functions.R    # Can call other R/* functions
├── internal_helpers.R      # Available to all R/* code
└── testthat_helpers.R      # Test utilities exported for test use

tests/testthat/
├── helper-*.R              # NOT available to R/* code
├── setup-*.R               # Run before tests, not with load_all()
└── test-*.R                # Test files
```

### Pattern 1: Dependency Declaration in DESCRIPTION
**What:** All packages used in tests, vignettes, or examples must be declared
**When to use:** Always - CRAN requirement

**Suggests field rules:**
- Test dependencies (DelayedArray, rhdf5, devtools, iterators, withr)
- Vignette dependencies (microbenchmark, pryr)
- Optional feature dependencies (already present: fmristore, Rarr, etc.)

**Example:**
```r
Suggests:
    testthat (>= 3.0.0),
    DelayedArray,
    rhdf5,
    devtools,
    iterators,
    withr,
    microbenchmark,
    pryr,
    # ... other suggests
```

### Pattern 2: Cross-Package Rd Links
**What:** Links to functions in other packages require package anchors
**When to use:** Any @seealso or inline reference to external package function

**Syntax evolution (roxygen2 >= 7.1.0):**
```r
# OLD (generates WARNING):
#' @seealso \code{\link{sampling_frame}}

# NEW (correct):
#' @seealso [fmrihrf::sampling_frame()] for creating temporal structures
```

Source: [roxygen2 changelog](https://roxygen2.r-lib.org/news/index.html) - roxygen2 7.1.1+ looks up files for cross-package links to avoid "Non-file package-anchored links" warnings.

### Pattern 3: .Rbuildignore for Non-Standard Files
**What:** Perl regex patterns (case-insensitive) to exclude files from package bundle
**When to use:** Always for development/documentation files not part of package

**Common patterns:**
```
^\.planning$          # Planning directory
^CLAUDE\.md$          # Development docs
^.*_repomix\.xml$     # Build artifacts
^experiments$         # Experimental code
```

Source: [R-hub .Rbuildignore guide](https://blog.r-hub.io/2020/05/20/rbuildignore/)

### Pattern 4: Test Helpers vs Package Code
**What:** Clear separation between test utilities and package functionality
**When to use:** Always - architectural boundary

**Decision matrix:**
| Code Type | Location | Reason |
|-----------|----------|--------|
| Used by tests only | tests/testthat/helper-*.R | Not installed with package |
| Used by R/* and tests | R/testthat-helpers.R | Available to package code |
| Used by exported functions | R/internal-helpers.R | Part of package namespace |

**CRITICAL:** Exported functions in R/* CANNOT reliably call functions in tests/testthat/helper-*.R because helpers aren't installed with the package.

Source: [R-hub testthat utility belt](https://blog.r-hub.io/2020/11/18/testthat-utility-belt/)

### Anti-Patterns to Avoid

- **Exported function sourcing test helpers:** R/golden_data_generation.R calling tests/testthat/helper-golden.R violates package boundaries
- **Unguarded Suggests usage in tests:** For heavy dependencies, use skip_if_not_installed(), but tidyverse convention is to assume Suggests are available during testing
- **Manual Rd editing:** Always use roxygen2, never edit man/*.Rd directly
- **Incomplete .Rbuildignore:** Missing patterns cause CRAN NOTEs about non-standard files

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Check package | Custom validation | devtools::check() | Matches CRAN exactly |
| Add .Rbuildignore entry | Manual regex | usethis::use_build_ignore() | Handles escaping correctly |
| Test skipping | Custom conditionals | testthat::skip_if_not_installed() | Standard, well-tested |
| Cross-package links | \link{func} | [pkg::func()] | Roxygen2 generates correct anchored links |
| Find missing imports | Grep NAMESPACE | checkhelper::print_globals() | Finds all undefined globals |

**Key insight:** R package development has mature tooling. The problems in CRAN-01 through CRAN-06 all have tool-based solutions that are more reliable than manual approaches.

## Common Pitfalls

### Pitfall 1: Test Dependencies Not in Suggests
**What goes wrong:** R CMD check fails with "Package required but not available" when trying to run tests
**Why it happens:** Tests use library() or :: without declaring dependency
**How to avoid:** Add all test dependencies to Suggests field in DESCRIPTION
**Warning signs:** Check log shows missing packages during test execution

**Source:** [R Packages book - Dependencies in Practice](https://r-pkgs.org/dependencies-in-practice.html)

### Pitfall 2: Exported Functions Calling Test Helpers
**What goes wrong:** Installed package fails at runtime because helper functions don't exist
**Why it happens:** Helper functions in tests/testthat/ aren't installed with package
**How to avoid:** Move shared code to R/ directory, not tests/testthat/helper*.R
**Warning signs:** Works during devtools::load_all() but fails after install.packages()

**Source:** [R-hub testthat utility belt](https://blog.r-hub.io/2020/11/18/testthat-utility-belt/) - "helpers are NOT installed with the package"

### Pitfall 3: Unanchored Cross-Package Links
**What goes wrong:** R CMD check WARNING: "Non-file package-anchored links"
**Why it happens:** \link{func} without package name when func is from another package
**How to avoid:** Use [pkg::func()] in roxygen2 comments, not bare \link{func}
**Warning signs:** Check log shows "cannot find file for topic" or "ambiguous link"

**Source:** [roxygen2 issue #1612](https://github.com/r-lib/roxygen2/issues/1612) - roxygen2 7.1.1+ requires anchored links for cross-package references

### Pitfall 4: Assuming All Suggests Are Available
**What goes wrong:** Tests fail on systems missing suggested packages
**Why it happens:** Suggests aren't mandatory dependencies
**How to avoid:** Use skip_if_not_installed() for heavy dependencies; tidyverse convention is to assume Suggests available during check
**Warning signs:** CRAN checks fail but local checks pass

**Source:** [R Packages book - Dependencies](https://r-pkgs.org/dependencies-in-practice.html) - Different philosophies exist; tidyverse assumes Suggests available

### Pitfall 5: .planning in Package Bundle
**What goes wrong:** R CMD check NOTE: "Non-standard file/directory found at top level"
**Why it happens:** .planning directory not in .Rbuildignore
**How to avoid:** Add ^\.planning$ to .Rbuildignore
**Warning signs:** Check log lists unexpected top-level directories

**Source:** [R-hub .Rbuildignore guide](https://blog.r-hub.io/2020/05/20/rbuildignore/)

## Code Examples

Verified patterns from official sources:

### CRAN-01: Add Test Dependencies to Suggests
```r
# In DESCRIPTION file, add to Suggests section:
Suggests:
    testthat (>= 3.0.0),
    DelayedArray,     # Used in test_study_backend_memory.R
    rhdf5,            # Used in test_backend_integration.R
    devtools,         # May be used in test helpers
    iterators,        # Used in test-golden-datasets.R, test_edge_cases.R
    withr,            # Used in 7 test files for temp environment setup
    # ... existing suggests
```
Source: Current DESCRIPTION + Grep results

### CRAN-02: Fix generate_all_golden_data Architecture
```r
# OPTION A: Move helper to R/ directory
# Create R/golden-data-helpers.R with @noRd tag
#' @noRd
generate_all_golden_data <- function() {
  # Implementation from tests/testthat/helper-golden.R
}

# OPTION B: Keep in tests, don't export wrapper
# Remove or unexport generate_golden_test_data() in R/golden_data_generation.R
# Keep it as inst/scripts/generate_golden_data.R only
```
Source: [R-hub internal functions](https://blog.r-hub.io/2019/12/12/internal-functions/)

### CRAN-03: Fix sampling_frame Cross-Reference
```r
# In R/dataset_methods.R or wherever get_TR is documented:
#' @seealso
#' [fmrihrf::sampling_frame()] for creating temporal structures,
#' [get_total_duration()] for total scan duration
```
Source: [roxygen2 cross-reference vignette](https://cran.r-project.org/web/packages/roxygen2/vignettes/index-crossref.html)

### CRAN-04: Add Vignette Dependencies
```r
# In DESCRIPTION, add to Suggests:
Suggests:
    # ... existing
    microbenchmark,  # Used in vignettes for performance comparisons
    pryr,            # Used in vignettes for memory profiling
```

### CRAN-05 & CRAN-06: Update .Rbuildignore
```r
# Use usethis::use_build_ignore() or manually add to .Rbuildignore:
^\.planning$
^CRAN_guidance\.md$
^repomix-output\.txt$
^fmristore_repomix\.xml$
# Note: Some patterns already present, just add missing ones
```
Source: [usethis use_build_ignore docs](https://usethis.r-lib.org/reference/use_build_ignore.html)

### Conditional Test Skipping (if needed)
```r
# In tests/testthat/test-h5-backend.R:
test_that("H5 backend loads data", {
  skip_if_not_installed("rhdf5")  # Only if rhdf5 is truly optional
  # test code
})
```
Source: [testthat skip functions](https://testthat.r-lib.org/reference/skip.html)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| \link{func} | [pkg::func()] | roxygen2 7.1.1 (2020) | Required for R 4.5.0+ documentation |
| Manual Rd links | Markdown links in roxygen | roxygen2 7.0.0 (2020) | Cleaner, less error-prone |
| Suggests optional everywhere | Suggests assumed during check | Ongoing debate | Tidyverse assumes available |
| Manual .Rbuildignore | usethis::use_build_ignore() | usethis 1.0.0 (2018) | Automatic escaping |

**Deprecated/outdated:**
- Bare `\link{external_func}` without package anchor: Generates warnings in R 4.5.0+
- Exporting test data generation functions: Test infrastructure shouldn't be user-facing API
- Missing .Rbuildignore entries: CRAN increasingly strict about non-standard files

## Open Questions

Things that couldn't be fully resolved:

1. **generate_all_golden_data architecture decision**
   - What we know: Current design violates package boundaries (R/* calling tests/*)
   - What's unclear: Whether to move to R/ or remove export entirely
   - Recommendation: OPTION B (remove export) is cleaner - golden data generation is development-time tooling, not user-facing feature. Keep script in inst/scripts/ for manual use.

2. **Test skipping philosophy**
   - What we know: Tidyverse assumes Suggests available, but some packages guard usage
   - What's unclear: Whether to add skip_if_not_installed() guards
   - Recommendation: Follow tidyverse convention - assume Suggests available during check. Only add guards if specific dependency (like sf) is known to be difficult to install.

3. **rhdf5 vs hdf5r dependency**
   - What we know: DESCRIPTION has hdf5r in Suggests, tests use rhdf5
   - What's unclear: Are both needed or is this inconsistent?
   - Recommendation: Audit test files - if both are used, add rhdf5. If only one is actually used, standardize.

## Sources

### Primary (HIGH confidence)
- [R Packages (2e) - Dependencies in Practice](https://r-pkgs.org/dependencies-in-practice.html) - Suggests field best practices
- [roxygen2 Changelog](https://roxygen2.r-lib.org/news/index.html) - Cross-package link syntax
- [R-hub .Rbuildignore Guide](https://blog.r-hub.io/2020/05/20/rbuildignore/) - Build ignore patterns
- [R-hub Testthat Utility Belt](https://blog.r-hub.io/2020/11/18/testthat-utility-belt/) - Helper file conventions
- [R-hub Internal Functions](https://blog.r-hub.io/2019/12/12/internal-functions/) - Internal vs exported functions
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html) - Official CRAN requirements

### Secondary (MEDIUM confidence)
- [testthat Skipping Tests](https://testthat.r-lib.org/articles/skipping.html) - skip_if_not_installed() usage
- [roxygen2 Cross-reference Vignette](https://cran.r-project.org/web/packages/roxygen2/vignettes/index-crossref.html) - Link syntax details
- [GitHub roxygen2 #1612](https://github.com/r-lib/roxygen2/issues/1612) - Anchored links discussion

### Tertiary (LOW confidence)
- None - all findings verified with official documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All tools are standard R package development tools
- Architecture: HIGH - Patterns documented in official R Packages book and R-hub blog
- Pitfalls: HIGH - All pitfalls verified with official documentation or authoritative sources

**Research date:** 2026-01-22
**Valid until:** 60 days (2026-03-23) - R package standards stable, but check for roxygen2/testthat updates

**Key findings for planner:**
1. All six requirements (CRAN-01 through CRAN-06) have standard solutions
2. CRAN-02 requires architectural decision (move code or remove export)
3. Issues are independent - can be fixed in any order, though CRAN-01/02 have most impact
4. No breaking changes required - all fixes are internal/configuration
5. Verification is built-in: R CMD check will show remaining issues after each fix
