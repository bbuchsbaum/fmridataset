# Phase 7: Developer Vignettes - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 3 developer vignettes (backend-development-basics, backend-registry, extending-backends) to have executable examples that match the current API. Developers should be able to follow the guides to create their own custom storage backends.

</domain>

<decisions>
## Implementation Decisions

### Teaching approach
- Build incrementally — start with simplest working backend, add features one at a time
- Goal of basics vignette: developer can create a read-only backend
- Use synthetic/in-memory data for examples — no external dependencies
- Focus on happy path — mention error handling exists but don't clutter tutorial code

### Code example depth
- Minimal viable implementations — only required methods (backend_get_data, backend_get_dims)
- No test code in vignettes — implementation only, testing is assumed knowledge
- extending-backends vignette adds optional methods (iteration, subsetting, chunking) on top of basics

### Claude's Discretion
- Whether to include performance benchmarks where relevant to the operation
- Exact structure of the incremental build-up
- How much explanation vs code in each section
- Cross-references between the 3 developer vignettes

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-developer-vignettes*
*Context gathered: 2026-01-24*
