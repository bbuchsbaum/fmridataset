# Phase 3: Zarr Decision - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Investigate Zarr backend viability and make a go/no-go decision for CRAN submission. Determine whether to keep Zarr (migrating from Rarr to the newer zarr package) or remove it entirely. This is an investigation and decision phase, not feature development.

</domain>

<decisions>
## Implementation Decisions

### Decision criteria
- Prefer the newer `zarr` CRAN package (https://cran.r-project.org/web/packages/zarr/) over Rarr
- Local Zarr file support is sufficient — cloud paths are nice-to-have, not required
- Core read/write operations must work reliably — no workarounds for basic functionality
- If zarr package has bugs or missing features in core operations, that's a no-go

### Investigation depth
- Comprehensive testing: various dtypes, chunk sizes, compression, edge cases
- Benchmark against H5 backend — compare read/write speeds
- Use small test data (64×64×30 brain, 100 timepoints) for quick iteration
- Quick glance at CRAN status and maintenance — verify actively maintained, not abandoned

### If keeping Zarr (go decision)
- Migrate zarr_backend.R from Rarr to zarr package
- Add zarr to Suggests (optional dependency), not Imports
- Error with install hint if user tries Zarr backend without zarr package installed
  - Clear error: "Install zarr package: install.packages('zarr')"

### If removing Zarr (no-go decision)
- Cross that bridge when we come to it — no pre-decision on removal approach

### Claude's Discretion
- Specific test cases and edge cases to cover
- Benchmark methodology details
- How to structure investigation documentation
- Technical details of zarr package API migration

</decisions>

<specifics>
## Specific Ideas

- New zarr package discovered: https://cran.r-project.org/web/packages/zarr/index.html
- This is newer than Rarr and should be evaluated as the preferred option
- User wants reliable core functionality over experimental features

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-zarr-decision*
*Context gathered: 2026-01-22*
