# Requirements: fmridataset 1.0

## Semantic correctness

- All assays align to exact observation and feature IDs.
- Scalar and multivariate axis annotations slice synchronously.
- Nesting, crossing, entities, and relations have distinct validated forms.
- Spatial compatibility is never inferred from dimensions.
- Missing observations are absent rows; missing coverage is explicit validity.

## Execution and storage

- Canonical sources and plans serialize without open handles or closures.
- Printing, explaining, filtering, and feature selection read zero assay bytes.
- Full realization is rejected before reading when it exceeds its memory budget.
- HDF5 sources push selections down and close handles on every failure path.
- Append publishes new shards without rewriting prior shards.
- Failed writes publish no partial dataset.

## Analysis

- Compiled designs preserve factors, contrasts, components, grouping, and term
  provenance.
- Blockwise results agree with trusted dense references under method-specific
  tolerances.
- Results are invariant to supported block sizes, sharding, and worker counts.
- Group methods preserve the distinction between effect-only, z/p-combination,
  and beta-plus-variance analyses.

## Release

- 0.10 provides an end-to-end memory/HDF5 walking skeleton.
- HDF5 is the only persistent backend required for 1.0 certification.
- The full 40 GB reference workload runs on controlled nightly infrastructure.
- Legacy top-level constructors and classes are absent from the 1.0 API.
