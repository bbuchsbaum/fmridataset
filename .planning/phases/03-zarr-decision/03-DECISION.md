# Phase 3: Zarr Decision

**Decision:** migrate-zarr (use CRAN zarr package)
**Date:** 2026-01-22
**Based on:** 03-INVESTIGATION.md findings and user preference

## Rationale

1. **Pure CRAN dependency** - Eliminates Bioconductor installation complexity
   - Users can install with simple `install.packages("zarr")`
   - No BiocManager, no version conflicts, no separate repos
   - Maintains CRAN eligibility for fmridataset

2. **Zarr v2 not needed** - Package creates new stores, doesn't need legacy support
   - Users working with existing v2 stores can convert them externally
   - v3 is the current standard and future of Zarr
   - Accepting this limitation simplifies codebase

3. **Current implementation broken** - Existing Rarr-based code used non-existent API
   - Used `Rarr::read_zarr_array(path = ...)` which doesn't exist
   - Required rewrite anyway, regardless of package choice
   - Starting fresh with working API

4. **Ecosystem improvement** - Stress testing helps mature the new CRAN zarr package
   - Package is very new (v0.1.1, Dec 2025)
   - Real-world usage will help identify and fix issues
   - Contributing to pure R ecosystem for scientific computing

## Trade-offs Accepted

- **Zarr v3 only** - No v2 compatibility
  - Users with v2 stores must convert externally
  - Documented clearly in README and function docs

- **New package** - Less field-tested (0.1.1, Dec 2025)
  - May discover bugs during usage
  - Marked as EXPERIMENTAL to set user expectations
  - Will report issues to maintainer

- **Cloud paths undocumented** - Local files tested, cloud uncertain
  - S3/GCS/Azure support not verified
  - Users can test and report findings
  - Will document known limitations

## Implementation

**Completed in 03-02-PLAN.md:**

- Rewrote R/zarr_backend.R using zarr:: R6 API
- Replaced Rarr with zarr in DESCRIPTION Suggests
- Updated tests for new API (38 tests passing)
- Marked backend as EXPERIMENTAL in documentation
- R CMD check passes cleanly

**API Changes:**

- `zarr::open_zarr(path)` returns zarr store (R6 object)
- Access root array via `store$root` (single-array stores)
- Array properties via `$shape`, `$dtype`, `$chunks`
- Data access via `array[, , , ]` subsetting
- Write via `as_zarr(arr, location=path)`

**Simplified Backend:**

- Removed data_key/mask_key parameters (single-array stores)
- Removed cache_size parameter (zarr handles caching internally)
- Backend stores single 4D fMRI arrays
- Mask data provided externally if needed

## Impact

**For Users:**

- Install: `install.packages("zarr")` instead of `BiocManager::install("Rarr")`
- Cannot read Zarr v2 stores (document in README)
- Backend marked EXPERIMENTAL - report bugs
- Simplified API: just provide path, no key parameters

**For Maintainers:**

- Cleaner dependency tree (pure CRAN)
- Working implementation (old one was broken)
- May need to forward bugs to CRAN zarr maintainer
- Document v3-only limitation prominently

**For Future:**

- Zarr backend provides cloud-native storage option
- When zarr package matures, can remove EXPERIMENTAL tag
- May add v2 compatibility if requested and feasible
- Pattern established for R6-based storage backends

## Next Steps

1. Update README with Zarr backend documentation
   - Installation instructions
   - Zarr v3-only limitation
   - EXPERIMENTAL status
   - Example usage

2. Add example to vignettes if useful
   - Creating Zarr stores from fMRI data
   - Performance comparison with NIfTI/HDF5

3. Test cloud storage if accessible
   - S3, GCS, Azure paths
   - Document findings

4. Monitor zarr package updates
   - Track bug fixes
   - Update when API stabilizes
   - Consider removing EXPERIMENTAL tag at v1.0
