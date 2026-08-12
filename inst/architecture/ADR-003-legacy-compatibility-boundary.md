# ADR-003: Legacy compatibility boundary

Status: accepted
Date: 2026-08-12

## Decision

`fmridataset` 1.0 does not ship the pre-frame dataset/backend architecture and
does not create another top-level compatibility package. The final 0.x source
line is frozen at commit `21de2ed441939a789d3704df52b959c7264e7b0c` and is
the maintenance boundary for applications that still require:

- `fmri_dataset`, `matrix_dataset`, `fmri_study_dataset`, or `latent_dataset`;
- open-handle storage backends and the backend registry;
- sampling-frame temporal generics and selectors;
- `fmri_series`, data-chunk iterators, or `fmri_group` reducers;
- the historical BIDS HDF5 wrapper API or DelayedArray backend seeds.

The 1.0 core retains only the canonical semantic model and its extension
protocol: frames, collections, studies, axes, entities, relations, feature
spaces and maps, serializable array sources, views, bounded execution, FDS,
and explicit spatial interoperability.

## Migration boundary

`upgrade_dataset()` reads supported self-contained 0.x objects and returns an
`fmri_frame`. Conversion methods for serialized class names remain internal;
their constructors and behavioral APIs are not exported. Provisional
schema-v1 frames migrate without numerical reads. In-memory matrix, series,
and NeuroVec inputs have explicit conversions. Sampling frames and ambiguous
file/backend datasets fail early because the generic cannot invent an assay
source or spatial identity.

Physical storage packages implement `ArraySource` and FDS codecs. Design and
statistical packages consume exported frame protocols. They do not revive the
legacy dataset hierarchy.

## Consequences

- The core no longer imports `fmrihrf`.
- There is one public data architecture rather than parallel frame and dataset
  systems.
- Users needing the historical API remain on the last 0.x line while migrating
  serialized data explicitly.
- Removal is intentional and breaking; it is covered by namespace tests,
  migration fixtures, package checks, and downstream frame-native tests.
