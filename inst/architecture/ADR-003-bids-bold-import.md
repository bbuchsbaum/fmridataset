# ADR-003: A narrow BIDS BOLD importer

Status: accepted
Date: 2026-08-13

## Decision

`fmridataset` owns one narrow on-ramp from an fMRIPrep derivatives tree to the
canonical semantic object:

```r
read_bids_bold(path, subject, ...)
```

It returns one lazy `fmri_frame` for one subject in one homogeneous volume
space. `bidser` owns discovery and BIDS entity parsing; `fmridataset` owns axis
identity, spatial validation, lazy sources, entities, relations, and tables.
The dependency direction remains `fmridataset -> bidser` through `Suggests`.

The importer is deliberately not a workflow facade. It does not expose
`discover()`, quality assessment, preprocessing, or workflow construction.

## Stable identity

Scans are ordered by their normalized BIDS-root-relative path. A scan ID is
that relative path with the NIfTI extension removed. Observation IDs are:

```text
<scan-id>::volume-<zero-based six-digit volume position>
```

The ordinary `volume_index` annotation is one-based for R users. Consequently,
IDs are independent of discovery order and of the absolute location of the
BIDS tree. Subsetting a completed frame preserves the imported IDs.

The importer passes these IDs through the existing `axis_frame(id = ...)`
contract. This ADR does not add a package-wide ID-policy API.

## Spatial domain and masks

The first implementation accepts fMRIPrep `desc-preproc` BOLD files only. All
selected files must have identical dimensions, affine, and dtype. Equal shape
alone is not spatial compatibility.

The default `mask = "intersection"` finds the most specific compatible brain
mask for each run and intersects their supports. An explicit mask path or
`volume_space` is also accepted. No resampling occurs. Empty intersections,
ambiguous masks, multiple spaces, multi-echo selections, and incompatible
grids fail before an assay is constructed.

Automatic brain-mask discovery depends on fMRIPrep BIDS entities and filename
conventions. The selected root-relative mask paths are recorded in the volume
space metadata so the resulting support is auditable. Re-check the discovery
contract when supporting a new fMRIPrep major version; callers can freeze the
choice with an explicit mask path or `volume_space`.

Union masks and feature-validity relations are deferred until their missing
coverage semantics are implemented explicitly.

## Experimental structure

The observation table has one row per acquired volume. Subject and run data
are de-duplicated entity frames. Symbolic relations connect observations to
runs and runs to subjects.

BIDS events retain their natural cardinality in a keyed `event_table`. They are
associated with `scan_id`; they are not repeated over acquired volumes or
implicitly convolved. Confounds are deferred pending a separate decision about
observation blocks versus auxiliary tables.

## Dependency and release boundary

The importer requires `bidser` 0.5.0 or newer, published through the package's
declared `Additional_repositories` R-universe. That version exports vectorized
`bids_entities()`, `n_volumes.character()`, and the existing fMRIPrep discovery,
TR, mask, and event APIs. It performs an explicit runtime capability/version
check and disables `bidser`'s on-disk index so importing does not modify the
source dataset. Public release claims require the declared repository to resolve
an installable compatible artifact and the installed-package integration test to
pass. A GitHub `Remotes:` entry is unnecessary for that repository-backed
dependency contract.

## Non-goals

- multiple subjects or a study-level importer;
- raw-data preprocessing;
- implicit resampling or registration;
- surface, CIFTI, or multi-echo import;
- union masks;
- confound compilation;
- replacement of the legacy `compress_bids_study()` HDF5 workflow;
- revival of the abandoned conversational `bids()` facade.
