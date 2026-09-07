# Project state

Date: 2026-09-06
Development version: 0.10.0.9000
Active milestone: 0.10 core frame, contracted for 1.0
Working branch: `release/1.0-contraction` (built on the PR 85 tip `3ae565e`)

## What changed on 2026-09-06

- The pre-frame 0.x architecture was removed rather than carried as adapters:
  dataset constructors, storage backends and registry, sampling-frame
  accessors, chunk iteration, series and selectors, groups, BIDS HDF5
  wrappers, config I/O, their tests, golden fixtures, and vignettes. The last
  commit carrying that surface is `3ae565e`. ADR-001 records the decision;
  there is no compatibility layer and no serialized-object upgrader.
- The semantic contraction train (`codex/beh-view-contract` through
  `codex/durable-id-policy`) was replayed onto the main line by cherry-pick:
  descriptor-consistent view assays, API audiences, bounded `explain()`, the
  canonical `frame_schema()`, the typed identity contract (ADR-006), typed
  metadata and lineage, lossless `bind_observations()`, canonical study links
  (FDS study schema v2), the canonical encoder with golden vectors, and the
  durable ID policy (ADR-007). The train's `upgrade_dataset()` commit was not
  taken.
- Three core gaps closed: keyed-domain validators with two-dimensional axis
  blocks (ADR-008, fixes the bind flattening defect), source revision
  fingerprints versus explicit `content_hash()` (ADR-009), and the selection
  algebra (ADR-010, in progress at the time of writing).
- Four executable vignettes replace the removed ones; the pkgdown reference is
  grouped by topic; README and the amended ADR-003/004/005 no longer describe
  the removed surface.

## Baseline qualifications

- Local suite with the pinned companions (fmristore `e7fbbfc`, multidesign
  `2ed3f30`, fmrigds `67cc585`, bidser 0.5.1) installed in an isolated
  library: 0 failures, 0 errors, 3017 passing, 2 skips (a bidser dev-dependency
  message test and the walking skeleton, see below). `lintr` clean.
- The walking skeleton skips because the pinned fmrigds writes result
  diagnostics into frame metadata, which the typed-metadata contract rejects.
  That is an fmrigds certification gap, recorded on its Mote issue.
- The `.h5` source provided by the pinned fmristore does not detect a stale
  file the way NIfTI and Zarr sources do; recorded on the fmristore alignment
  issue.
- Hosted CI has not yet run on this branch; the evidence above is local.

## Companion consumers

Sixteen local development packages still call the removed API (fmrireg,
fmrireg.gnef, fmrireg.garrr, fmrireg.cca, fmriplot, fmriproj, and others).
They migrate directly to `fmri_frame`, `fmri_collection`, and `fmri_study`;
`multidesign` already uses only the frame API. Nothing in this package waits
on them.

Historical 0.9 planning material remains under `.planning/milestones/` and
`.planning/phases/` and must not be presented as current release evidence.
