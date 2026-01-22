# Phase 2: Tech Debt - Context

**Gathered:** 2026-01-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Resolve H5 backend resource management issues: ensure file handles are properly closed in all error paths, and commit the storage_backend.R fix. No new features — pure reliability fixes.

</domain>

<decisions>
## Implementation Decisions

### Error recovery behavior
- Clean up resources on failure, no retry logic
- Let caller handle recovery decisions
- Failed operations should leave no dangling handles

### Handle lifecycle
- Close handles immediately after use
- Use `on.exit()` protection for all H5 operations
- No handle pooling — simple open/use/close pattern

### Logging/diagnostics
- Use R's standard `warning()` for recoverable issues
- Use `stop()` for unrecoverable failures
- No custom logging infrastructure

### Testing approach
- Standard testthat with `withr::defer()` for cleanup verification
- Test error paths explicitly (induce failures, verify cleanup)
- No elaborate stress testing framework

### Claude's Discretion
- Exact placement of `on.exit()` calls
- Whether to use `tryCatch` vs `on.exit` in specific contexts
- Helper function organization

</decisions>

<specifics>
## Specific Ideas

"Sensible behavior consistent with quality code but not over-engineered" — follow standard R package patterns, nothing custom.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-tech-debt*
*Context gathered: 2026-01-22*
