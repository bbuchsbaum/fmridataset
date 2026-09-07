# Roadmap: fmridataset 1.0

## 0.10: core frame

- repository, test, and namespace contracts;
- stable IDs, axis frames, axis blocks, and aligned assays;
- `index_space`, `volume_space`, and serializable array sources;
- `fmri_frame`, synchronized lazy views, bounded collection, and image recovery;
- first memory/HDF5 design-and-fit walking skeleton.

## 0.11: scalable studies

- provider-based `delarr` execution and source pushdown;
- row-sharded sources and workload-aware planning;
- atomic HDF5 FDS schema, append, recovery, and migration;
- entity registries, relations, hierarchy, collections, and studies.

## 0.12: design and spatial algebra

- complete `multidesign` compiler and term metadata;
- surface, parcel, basis, and composite spaces;
- explicit maps, validity, and common-space virtual frames;
- frame-native `fmrigds` reducers and result frames.

## 1.0: freeze and certification

- (done early, 2026-09-06: the legacy top-level API was removed during 0.10
  rather than after a 0.12 migration release; see ADR-001 and STATE.md);
- unified user and developer documentation;
- reverse-dependency and cross-platform validation;
- controlled full-scale HDF5 performance and failure certification;
- public API and FDS schema freeze.

The ticket-level dependency graph is authoritative in Beads.
