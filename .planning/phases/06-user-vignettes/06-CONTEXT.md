# Phase 6: User Vignettes - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can learn fmridataset from executable, accurate documentation covering getting started, architecture, HDF5 usage, and multi-subject analysis. Covers 4 vignettes: fmridataset-intro.Rmd, architecture-overview.Rmd, h5-backend-usage.Rmd, and study-level-analysis.Rmd.

</domain>

<decisions>
## Implementation Decisions

### Teaching approach
- Concept-first: explain the idea, then show implementation
- Simple-to-complex progression within each vignette
- Technical tone (precise, documentation-style, minimal narrative)
- Include gotchas and common mistakes inline where relevant

### Example data strategy
- Use synthetic (generated) data — no external files or bundled data
- Always set.seed() for reproducibility — same output every run
- Minimal viable realism — just enough structure to demonstrate API (random matrices, reasonable dimensions)
- Show data creation code — user sees how synthetic data is made

### Depth and scope
- Hybrid purpose: tutorial flow with reference-friendly sections
- Focused scope: one core concept/workflow per vignette
- Selective output: show code output only when it adds understanding
- "See also" / "Next steps" section at the end of each vignette

### Cross-referencing
- Sequential reading order: intro → architecture → h5-backend → study-level
- Build on prior knowledge: later vignettes can skip basics covered earlier
- No explicit prerequisites section — sequence is implied

### Claude's Discretion
- How to reference function documentation (inline vs link to ?function based on complexity)
- Exact structure of "See also" sections
- Which code outputs to show vs hide

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

*Phase: 06-user-vignettes*
*Context gathered: 2026-01-23*
