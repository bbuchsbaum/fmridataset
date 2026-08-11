# fmridataset 1.0 project

## Objective

Make `fmri_frame` the canonical, spatially typed observation-by-feature data
model, with `fmri_collection` and `fmri_study` above it. Numerical arrays remain
lazy and backend-neutral; experimental hierarchy, spatial identity, mappings,
and provenance are explicit and validated.

## Active releases

| Release | Product gate |
|---|---|
| 0.10.0 | Core axes, spaces, sources, frames, views, and walking skeleton |
| 0.11.0 | Sharded execution, atomic HDF5 storage, entities, collections, studies |
| 0.12.0 | Design compiler, full spatial algebra, frame-native group analysis |
| 1.0.0 | Legacy removal, API/schema freeze, full-scale certification |

The remote `v0.9.0` tag is historical and is not an ancestor of the current
main line. It is preserved, not rewritten. Older milestone documents under
`.planning/milestones/` and `.planning/phases/` are historical evidence, not
the active roadmap.

## Non-negotiable contracts

1. One frame has one observation domain and one feature domain.
2. Aligned assays share exact axis IDs, not merely dimensions.
3. Experimental factors are annotations and relations, not tensor axes.
4. Spatial compatibility requires feature IDs and a space digest.
5. Slicing is synchronized, lazy, and non-dropping.
6. Canonical objects contain descriptors, not runtime handles or closures.
7. No metadata operation implicitly reads assay values.
8. Model matrices are derived objects with complete term provenance.

See `inst/architecture/ADR-001-canonical-data-model.md` for package ownership.
