# Phase 5: Final Validation - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Ensure fmridataset passes R CMD check --as-cran with zero errors, zero warnings, and zero notes. This includes clean installation, working examples, and successful vignette builds. The phase resolves any issues surfaced by the check — it does not add new functionality.

</domain>

<decisions>
## Implementation Decisions

### Issue resolution approach
- Fix issues immediately as they're found — don't batch or defer
- Thorough investigation for unclear issues — understand root cause before fixing
- Opportunistic cleanup — if something adjacent to a fix is clearly wrong, address it
- Strict scope otherwise — only touch what R CMD check explicitly flags

### Note handling strategy
- **rhdf5 must be removed** — hdf5r is the correct HDF5 dependency (user clarification)
- Must achieve zero notes — compress, restructure, or remove anything that triggers size notes
- Fix all file pattern notes — update .Rbuildignore for anything flagged
- Document truly unavoidable notes in cran-comments.md with explanation for CRAN reviewers

### Claude's Discretion
- Whether to add CI checks or tests for issues that might recur (prevention vs overhead tradeoff)
- Example/vignette handling if issues arise (update, dontrun, or remove)
- Fresh install testing scope (local vs multi-platform)

</decisions>

<specifics>
## Specific Ideas

- User explicitly noted rhdf5 should not be a dependency — the package uses hdf5r as the preferred HDF5 library
- Zero tolerance for notes — package should pass completely clean

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-final-validation*
*Context gathered: 2026-01-22*
