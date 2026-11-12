# ADR-011: Saved studies are BIDS derivative datasets; FDS JSON

Status: accepted for the 1.0 contract. Extends the persistence half of
ADR-002 (FDS logical schema) to whole studies, and records the observation-ID
scheme of BIDS imports.

## Context

`write_frame()` and `open_frame()` persist one frame through `fmristore`.
Saving an `fmri_study` took a manual recipe: write each frame, `saveRDS()` the
study manifest, reopen the frames, and call `study_from_fds_manifest()`. That
recipe was not atomic, and `saveRDS()` made the study manifest readable only
from R, although FDS is meant to be storage-neutral. The shipped JSON schemas
were checked only for internal consistency; no real manifest was ever written
as JSON.

The study concept is not new. Bioconductor's `MultiAssayExperiment` and
scverse's `MuData` hold several assays over shared subjects, and BIDS already
organises subjects, sessions, runs, and events as files. BIDS 1.10 also uses
`DatasetType: "study"` for a directory that bundles `sourcedata/`, `rawbids/`,
and `derivatives/`; an `fmri_study` is an analysis object, not that.

## Decision

### On disk, a study is a BIDS derivative dataset

`write_study()` writes a directory that BIDS tools recognise as a derivative
dataset, and adds only what BIDS has no place for:

| File | Role | BIDS status |
|---|---|---|
| `dataset_description.json` | `Name`, `BIDSVersion`, `DatasetType: "derivative"`, `GeneratedBy` (fmridataset and version), optional `SourceDatasets` and `DatasetLinks` | BIDS |
| `participants.tsv` | The subject entity, when its IDs are valid BIDS labels; `study.json` stays authoritative | BIDS |
| `study.json` | The study manifest in FDS JSON, with each frame referenced by its semantic digest instead of copied | package, in `.bidsignore` |
| `storage.json` | Where each frame and study-level array is stored, the study digest, and package versions; not part of identity | package, in `.bidsignore` |
| `frames/` | One `fmristore` HDF5 file per frame; collection members get one each | package, in `.bidsignore` |
| `arrays.h5` | Study-level arrays (entity blocks), read back into memory with their declared dtype | package, in `.bidsignore` |

Frames go in `frames/` rather than at BIDS-style paths: HDF5 is not a BIDS
data format, so BIDS-style names would misrepresent the files, and frames that
combine subjects have no settled BIDS location (a `group-` entity is only
proposed, in BEP039).

The directory is written under a temporary name beside the target and renamed
into place when complete; `overwrite = TRUE` moves the old directory aside
first and restores it if the rename fails.

`open_study()` reopens frames lazily, checks each against the digest in
`study.json`, rebuilds the full manifest from the reopened frames, checks it
against the study digest in `storage.json`, and calls
`study_from_fds_manifest()`. A replaced frame file or an edited `study.json`
raises `fmridataset_error_source_stale`.

### FDS JSON, version 1 (`org.fmridataset.fds-json/v1`)

Manifests hold R values whose types enter their digests, so the JSON encoding
is lossless. It is plain where unambiguous and tagged elsewhere:

- a length-1 character, logical, integer, or double without attributes and
  not `NA` is a JSON string, boolean, or number; doubles always carry a
  decimal point or exponent and integers never do;
- a list with unique, non-empty names not starting with `$`, and no other
  attributes, is a JSON object; an unnamed list without attributes is an array;
- anything else is an object with one value tag (`$chr`, `$lgl`, `$int`,
  `$dbl`, `$cplx`, `$raw`, `$list`, `$null`, `$sparse`) and optional `$names`
  and `$attr`. `NA` is `null`; in `$dbl`, `NaN`, `Inf`, `-Inf`, and negative
  zero are the strings `"NaN"`, `"Inf"`, `"-Inf"`, and `"-0"`.

Doubles are written with 17 significant digits. Decoding gives back an
`identical()` value (sparse matrices: identical canonical bytes), so
`fds_study_manifest_digest()` survives the round trip.

### BIDS observation IDs and provenance

`read_bids_bold()` names each run by its acquisition: the entities `sub`,
`ses`, `task`, `acq`, `ce`, `rec`, `dir`, `run`, `echo`, `part`, in that order,
with labels copied from the file name. Derivative-only entities (`space`,
`res`, `desc`) and the directory are excluded, so one acquisition has one set
of IDs in every output space. Observation IDs append the 1-based volume number
(`sub-01_task-memory_run-01_vol-0001`). Inputs are recorded in provenance as
BIDS URIs, `bids:raw:<path>`, where `raw` is the imported dataset and
`write_study(dataset_links =)` can say where it lives.

## Consequences

- A study can be inspected without R: its subjects, links, metadata, and
  provenance are in JSON, and BIDS tools index the dataset.
- Frame manifests are still stored by `fmristore` with R `serialize()`. They
  are not moved to FDS JSON yet because they embed feature-space operators
  inline (a basis space's encoder and decoder), which would make whole-brain
  manifests hundreds of megabytes of text. Moving large operators into
  referenced arrays is a separate frame-format change, to be made together
  with `fmristore`.
- `study.json` is the encoded form of the FDS study manifest, not the plain
  JSON shape `fds-study-v2.schema.json` describes; validating decoded
  manifests against that schema remains future work.
- Study-level arrays are read into memory on open; they are entity-sized.
