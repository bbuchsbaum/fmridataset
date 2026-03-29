# Compressed BIDS Study Archive: Revised Architecture

## The Problem

An fMRI study lives as thousands of files across a BIDS directory tree — NIfTI images, events.tsv files, confounds, JSON sidecars. You want to capture an entire study in **a single compressed HDF5 file** that is queryable by subject/task/session/run, preserves the experimental design (events), and supports multiple compression strategies.

## The Ecosystem

```
Tier 1: neuroim2              (spatial foundation — NeuroVec, NeuroVol, NeuroSpace)
Tier 2: bidser                (BIDS project querying — bids_project, read_events, preproc_scans)
        fmrihrf               (temporal structure — sampling_frame, blocklens)
Tier 3: fmristore             (HDF5 parcellated storage — H5ParcellatedMultiScan, cluster averages)
        neuroarchive           (HDF5 transform pipelines — LNA format, basis/quant/HRBF compression)
Tier 4: fmridataset            (unified dataset interface — backends, event_table, fmri_group)
```

**Current integration state: zero runtime coupling between the 4 packages.** They share neuroim2 as a common foundation but have no bridge code connecting them.

## Placement Decision: Build in fmridataset

fmridataset is the natural home because:

1. It's the highest-level interface (Tier 4) — designed to abstract over storage
2. Its pluggable backend architecture is exactly the extension mechanism we need
3. It already has `event_table`, `sampling_frame`, `fmri_study_dataset`, `fmri_group`
4. It already Suggests both `bidser` and `fmristore`
5. Adding a new backend is the idiomatic extension pattern

**fmristore stays as-is** — it provides parcellation infrastructure that the new backend delegates to. **neuroarchive is not a dependency yet** — the schema leaves a clean seam for future LNA compression, but Phase 1 implements parcellated mode only.

---

## Critical Design Decisions

### Decision 1: Per-scan backends sharing one H5 file (not a monolith)

`fmri_study_dataset` expects a list of per-subject backends composed via `study_backend`. Methods like `get_data_matrix(study, subject_id="01")` reach into `x$backend$backends[[idx]]`. The `StudyBackendSeed` for DelayedArray derives column count from the backends.

**Resolution:** Create lightweight `bids_h5_scan_backend` objects — one per scan — each holding a reference to the same shared H5 file handle plus its group path. The user experience is still "one file":

```r
study <- bids_h5_dataset("my_study.h5")   # one file, one call
```

Internally — **two-level nesting** that matches `fmri_study_dataset`'s expectations:

1. Open H5 file once → shared ref-counted connection
2. Create one `bids_h5_scan_backend` per scan (just a group path pointer)
3. Group scans by subject. For each subject:
   - If the subject has multiple runs: compose that subject's scan backends into a `study_backend` (acting as a run-composite)
   - If single run: use the scan backend directly
   - Wrap into a `fmri_dataset` with per-subject event_table, sampling_frame, and censor
4. Compose the per-subject `fmri_dataset` objects via `fmri_study_dataset(datasets, subject_ids)`

The top level is **subject-composite** (`study_backend` over per-subject datasets). Each subject's backend may itself be **scan-composite** (a `study_backend` over that subject's scan backends). This matches how `get_data_matrix(study, subject_id="01")` indexes into `x$backend$backends[[idx]]` — it reaches the subject level, not individual scans.

This gives us `data_chunks()`, `as_delarr()`, and per-subject access for free.

### Decision 2: Parcellated data is native feature-space (K parcel columns, not V voxels)

This is the main structural issue. The current backend contract couples three things:

- `backend_get_dims()$spatial` → 3D volume geometry
- `backend_get_mask()` → logical vector of length `prod(spatial)`
- `backend_get_data(rows, cols)` → matrix where `ncol == sum(mask)`

This coupling is enforced in `validate_backend()` (storage_backend.R:170-188), `study_backend` column bounds (study_backend.R:161), `as_delarr()` column sizing (as_delarr.R:26), selectors (series_selector.R:39), and printing.

For parcellated data, the columns are K parcels, not V voxels. Trying to keep voxel geometry in the backend while returning parcel data breaks `study_backend`, `as_delayed_array()`, and chunking.

**Resolution:** For Phase 1, treat parcellated data honestly — it's feature-space data with K columns:

```r
backend_get_dims.bids_h5_scan_backend → list(spatial = c(K, 1, 1), time = T)
backend_get_mask.bids_h5_scan_backend → rep(TRUE, K)
backend_get_data.bids_h5_scan_backend → matrix [T, K]
```

The original voxel geometry, brain mask, and cluster-to-voxel mapping live in the H5 file as metadata (under `/spatial/` and `/parcellation/`), accessible through dedicated methods on the `bids_h5_dataset` object — but they do not flow through the backend contract. This means:

- `validate_backend()` passes: `length(mask) == prod(spatial) == K`, `sum(mask) == K`
- `study_backend` works: column bounds are K, consistent across scans
- `as_delarr()` works: column count derived from mask is K
- `data_chunks()` works unchanged
- `index_selector()` works over parcel columns (select parcels 1:10, etc.)
- ROI/sphere/voxel selectors do **not** work on parcellated data — correct, because parcellated data doesn't have voxel resolution

**No voxel-feature space refactor needed for Phase 1.** delarr is the primary lazy matrix path in fmridataset; DelayedArray is a secondary fallback. Phase 1 does not need to optimize for DelayedArray/StudyBackendSeed at all. If LNA mode is added later (voxel-reconstructable data), that's when the backend contract distinction might need revisiting.

### Decision 3: neuroarchive deferred, schema extensible

neuroarchive is less mature than fmridataset. We prepare the way:

- The HDF5 root attribute `compression_mode` is the dispatch key
- `/scans/<name>/data/` group structure accommodates different payloads per mode
- Phase 1 implements `"parcellated"` only
- neuroarchive is **not** added to Suggests yet
- Future modes register by adding a handler for their `compression_mode` string

### Decision 4: Events and task selection are Phase 1

A BIDS study typically has multiple tasks. Nobody analyzes rest and nback simultaneously. Task selection is not convenience — it's core workflow. The plan includes:

- A **scan manifest** as a first-class component of `bids_h5_dataset`
- `task` column flows into event_table for every scan
- A plain subsetting helper `subset_bids_h5(study, task=, subject=, session=, run=)` using standard evaluation (not NSE)

---

## HDF5 Schema (v1.0)

```
/
├── [attrs: format = "bids_h5_study", version = "1.0"]
├── [attrs: compression_mode = "parcellated"]
├── [attrs: writer_version = "<fmridataset version>"]
│
├── bids/                              # Study-level BIDS metadata
│   ├── name                           # study name (string)
│   ├── space                          # e.g. "MNI152NLin2009cAsym"
│   ├── pipeline                       # e.g. "fmriprep"
│   ├── dataset_description            # JSON string
│   ├── participants/                  # participants.tsv columns
│   │   ├── participant_id             # string array
│   │   └── <other columns>            # age, sex, group, etc.
│   ├── tasks                          # string array
│   └── sessions                       # string array (absent if none)
│
├── spatial/                           # Original voxel geometry (metadata only)
│   ├── header/                        # NIfTI-like (dim, pixdim, qform)
│   ├── mask                           # [X, Y, Z] uint8 — original brain mask
│   └── voxel_coords                   # [N_voxels, 3] int32
│
├── parcellation/                      # Parcellated mode: cluster info
│   ├── cluster_map                    # [N_voxels] int32 — voxel-to-cluster
│   ├── cluster_ids                    # [K] unique cluster IDs
│   └── cluster_meta/                  # Optional: labels, network names, etc.
│       ├── labels                     # string array
│       └── ...
│
├── scans/                             # Per-run data
│   └── sub-01_task-nback_run-01/
│       ├── data/
│       │   └── summary_data           # [T, K] float32 (parcellated mode)
│       │                              # Future modes add their own datasets here
│       ├── events/                    # Column-per-dataset from events.tsv
│       │   ├── [attrs: n_events = N]
│       │   ├── onset                  # float64
│       │   ├── duration               # float64
│       │   ├── trial_type             # string
│       │   └── ...                    # task-specific columns
│       ├── confounds/                 # Optional confound regressors
│       │   └── data                   # [T, n_confounds] float64
│       │       [attrs: names = c(...)]
│       ├── censor                     # [T] uint8 (0 = keep, 1 = censor)
│       └── metadata/
│           ├── subject                # string
│           ├── task                   # string
│           ├── session                # string (absent if none)
│           ├── run                    # string
│           └── tr                     # float64
│
└── scan_index/                        # Fast lookup table
    ├── scan_name                      # string array
    ├── subject                        # string array
    ├── task                           # string array
    ├── session                        # string array
    ├── run                            # string array
    ├── n_time                         # int array
    ├── time_offset                    # int array (cumulative start row)
    ├── has_events                     # logical
    └── has_confounds                  # logical
```

### Schema design notes

- **`/scan_index/time_offset`**: Cumulative start row per scan for efficient row→scan mapping without scanning all `n_time` values.
- **Events as column arrays**: Better HDF5 performance than compound datasets for variable-length strings. `n_events` attribute enables efficient read without probing lengths.
- **`/spatial/` and `/parcellation/` are metadata**: They record the original voxel geometry and cluster mapping for provenance and reconstruction, but the backend contract operates in parcel-space (K columns).
- **`compression_mode` at root**: Dispatch key for future modes. Readers that don't recognize the mode fail with an informative error.
- **Censor vector per scan**: Stored as uint8 to match existing fmri_dataset convention.
- **`writer_version`**: Package version that wrote the file, for debugging.

---

## API Design

### Writer

```r
compress_bids_study <- function(
  x,                            # bidser::bids_project or BIDS directory path
  file,                         # output .h5 path
  mode = c("parcellated"),      # only parcellated in Phase 1

  # Parcellated mode:
  clusters,                     # ClusteredNeuroVol (atlas in study space)
  summary_fun = mean,           # aggregation function per cluster

  # Common:
  mask = NULL,                  # LogicalNeuroVol (default: derive from clusters)
  space = "MNI152NLin2009cAsym",
  tasks = NULL,                 # character filter — NULL means all tasks
  subjects = NULL,              # character filter — NULL means all subjects
  sessions = NULL,              # character filter — NULL means all sessions
  confounds = NULL,             # confound_set() from bidser, or character vector
  compression = 4L,             # HDF5 compression level
  verbose = TRUE
)
# Returns: bids_h5_dataset object (reader for the newly created file)
```

**Internal workflow (streaming, one scan at a time):**
1. Resolve `x` → `bidser::bids_project` (if path, create project)
2. Query scan manifest via `bidser::preproc_scans()`, filter by tasks/subjects/sessions
3. Validate TR consistency across scans (fail early if TRs differ)
4. Write shared infrastructure: `/bids/`, `/spatial/`, `/parcellation/`
5. For each scan (streaming):
   - Read NIfTI via `neuroim2::read_vec(scan_path, mask)`
   - Compute parcel averages via `fmristore::summarize_by_clusters()`
   - Write `[T, K]` to `/scans/<name>/data/summary_data`
   - Read events via `bidser::read_events()` → write column arrays
   - Read confounds via `bidser::read_confounds()` → write matrix
   - Release NIfTI from memory
6. Write `/scan_index/` with cumulative `time_offset`
7. Return `bids_h5_dataset(file)`

### Reader

```r
bids_h5_dataset <- function(file, preload = FALSE)
# Returns: bids_h5_study_dataset (subclass of fmri_study_dataset) with:
#   - scan_manifest: tibble of scan metadata (subject, task, session, run, n_time)
#   - event_table: combined events from all runs, with columns:
#       onset, duration, trial_type, ..., run (BIDS label), run_id (sequential int),
#       subject_id, task, session
#   - sampling_frame: temporal structure from TR + run lengths
#   - censor: concatenated censor vector across all subjects (managed by
#       bids_h5_study_dataset, not inherited from fmri_study_dataset)
#   - backend: subject-composite study_backend, where each subject's backend
#       is itself a scan-composite study_backend (for multi-run subjects)
#   - h5_connection: shared ref-counted H5 file handle
#
# BIDS-specific accessors (local generics with conditional bidser delegation):
#   participants(study)  → character vector of subject IDs
#   tasks(study)         → character vector of task names
#   sessions(study)      → character vector of session names (or NULL)
#   scan_manifest(study) → tibble of per-scan metadata
```

### Backend

```r
bids_h5_scan_backend <- function(h5_connection, scan_group_path, n_parcels, n_time)
# Lightweight object: shared H5 handle + group path + cached dims
# Class: c("bids_h5_scan_backend", "storage_backend")
#
# Contract implementation:
#   backend_get_dims()     → list(spatial = c(n_parcels, 1, 1), time = n_time)
#   backend_get_mask()     → rep(TRUE, n_parcels)
#   backend_get_data(r, c) → reads /scans/<name>/data/summary_data[r, c]
#   backend_get_metadata() → list(compression_mode, task, subject, session, run, tr)
```

### Shared H5 Connection

```r
h5_shared_connection <- function(file)
# Ref-counted wrapper around hdf5r::H5File
# - h5_connection$handle: the open H5File object
# - h5_connection$ref_count: incremented per scan backend
# - h5_connection$release(): decrements; closes file when ref_count hits 0
# - h5_connection$acquire(): increments ref_count
```

### Query & Access

```r
study <- bids_h5_dataset("my_study.h5")

# Scan manifest
study$scan_manifest
#>   scan_name                    subject task  session run n_time has_events
#>   sub-01_task-rest_run-01      01      rest  pre     01  200    TRUE
#>   sub-01_task-nback_run-01     01      nback pre     01  300    TRUE
#>   sub-02_task-rest_run-01      02      rest  pre     01  200    TRUE
#>   sub-02_task-nback_run-01     02      nback pre     01  300    TRUE

# BIDS metadata
participants(study)              # → c("01", "02")
tasks(study)                     # → c("rest", "nback")
sessions(study)                  # → c("pre") or NULL

# Subset by task (standard evaluation, returns new bids_h5_study_dataset)
nback <- subset_bids_h5(study, task = "nback")
rest  <- subset_bids_h5(study, subject = c("01", "02"), task = "rest")

# Standard fmridataset API works on subsets
get_data_matrix(nback)           # → [T_total, K] matrix
get_TR(nback)                    # → 2.0
n_runs(nback)                    # → runs across all subjects for nback

# Per-subject data
get_data_matrix(nback, subject_id = "01")

# Events include task column
study$event_table
#>   onset duration trial_type run subject_id task    session
#>   ...

# Group operations (requires explicit conversion — as_fmri_group is not
# an S3 method on fmri_study_dataset; we provide a helper)
group <- study_to_group(nback)
results <- group_map(group, function(ds) {
  mat <- get_data_matrix(ds)
  cor(mat)
})

# Confounds — keyed by scan_name for unambiguous access
get_confounds(study, scan_name = "sub-01_task-nback_run-01")  # → single tibble
get_confounds(study, subject = "01")  # → named list of tibbles (one per scan)
get_confounds(study, task = "nback")  # → named list of tibbles

# Parcellation metadata (not in backend contract, on the study object)
parcellation_info(study)
#>   list(cluster_ids, cluster_map, labels, n_parcels)
```

### Subsetting Implementation

`subset_bids_h5(study, task=, subject=, session=, run=)`:

1. Filter `study$scan_manifest` by provided arguments
2. Select corresponding per-scan backends (already in memory, just pointers)
3. Regroup by subject → new per-subject `fmri_dataset` objects
4. Return new `bids_h5_study_dataset` via `fmri_study_dataset()` with:
   - Filtered event_table (only matching scans, preserving both `run` and `run_id`)
   - Filtered sampling_frame (only matching run lengths)
   - Concatenated censor vector from matching scans
   - Shared H5 connection (same handle, ref count adjusted)

---

## New Code in fmridataset

| File | Contents |
|:-----|:---------|
| `R/bids_h5_backend.R` | `bids_h5_scan_backend` class, 6-method contract, `h5_shared_connection` |
| `R/bids_h5_dataset.R` | `bids_h5_dataset()` reader, `bids_h5_study_dataset` class, `subset_bids_h5()`, `study_to_group()`, `participants()`, `tasks()`, `sessions()`, `parcellation_info()`, `get_confounds()` |
| `R/bids_h5_write.R` | `compress_bids_study()` writer |
| `R/bids_h5_events.R` | Event read/write helpers for HDF5 column arrays |
| `tests/testthat/test-bids_h5.R` | Round-trip tests, subsetting, events, confounds |

### DESCRIPTION changes
- Add `neuroarchive` to Suggests: **NO — deferred**
- All required optional deps already in Suggests: bidser, fmristore, hdf5r

---

## Existing Infrastructure to Reuse

| What | From | Function/Class |
|:-----|:-----|:---------------|
| BIDS querying | bidser | `bids_project()`, `preproc_scans()`, `read_events()`, `read_confounds()`, `participants()`, `tasks()`, `sessions()` |
| Mock BIDS | bidser | `create_mock_bids()` |
| Parcel averaging | fmristore | `summarize_by_clusters()` |
| HDF5 I/O | hdf5r | Direct H5 read/write |
| Temporal structure | fmrihrf | `sampling_frame()` |
| Dataset interface | fmridataset | `fmri_dataset()`, `fmri_study_dataset()`, `study_backend`, backend registry |
| Spatial objects | neuroim2 | `NeuroVec`, `LogicalNeuroVol`, `ClusteredNeuroVol` |

---

## Implementation: Phase 1 (Parcellated Mode)

### Step 1: `bids_h5_scan_backend` + `h5_shared_connection`
- S3 class implementing 6-method contract in parcel feature-space
- Ref-counted shared H5 connection
- Register in backend registry as `"bids_h5_scan"`
- Unit tests for contract compliance

### Step 2: `compress_bids_study()` writer
- Streaming: one scan at a time, never hold >1 NIfTI in memory
- Dependency checks upfront (bidser, fmristore, hdf5r)
- Write schema v1.0 with all groups
- Write scan_index with time_offset
- Validate TR consistency across scans

### Step 3: `bids_h5_dataset()` reader + `bids_h5_study_dataset` class
- Open H5, read scan_index → per-scan backends → group by subject
- Compose via `fmri_study_dataset()`
- scan_manifest as first-class field
- `participants()`, `tasks()`, `sessions()` methods. These generics are owned by bidser (a Suggests-only dep), so methods must use conditional S3 registration via `S3method()` directives in NAMESPACE with `.onLoad()` fallback, or define local generics that defer to bidser's when available. The plan must not silently require bidser to be attached for these to work. Approach: define fmridataset-local generics that check for and delegate to bidser generics if loaded, otherwise dispatch on local methods only.

### Step 4: `subset_bids_h5()` for task/subject/session/run filtering
- Filter manifest → select backends → recompose study_dataset
- Test: subset by task, by subject, by task+subject

### Step 5: Events, confounds, censor
- Column-array HDF5 event read/write
- `get_confounds(study, scan_name=)` accessor — keyed by scan_name, returns named list of tibbles when multiple scans match; single tibble when unambiguous
- Censor vector: `fmri_study_dataset()` does not propagate censor (R/dataset_constructors.R:486 drops it). `bids_h5_study_dataset` must aggregate per-scan censor vectors itself and store as a top-level field. The per-subject `fmri_dataset` objects carry their own censor; the study-level censor is the concatenation across all subjects.
- Task column in event_table
- Events carry both BIDS `run` (the BIDS run label, e.g. "01") and internal `run_id` (sequential integer across runs within a subject, as fmridataset expects in multiple places)

### Step 6: Integration tests
- Round-trip: `create_mock_bids()` → `compress_bids_study()` → `bids_h5_dataset()` → verify
- `get_data_matrix()` returns correct `[T, K]`
- `event_table` matches original events.tsv content
- `subset_bids_h5(task = "nback")` returns correct subset
- `data_chunks()` works on the result
- `as_delarr()` works on the result
- `index_selector()` works over parcel columns
- `fmri_group()` works on the result
- Missing dependency errors are informative

## Future Phases

### Phase 2: LNA mode (when neuroarchive is ready)
- Add `neuroarchive` to Suggests
- New compression_mode handler for `"lna"`
- `bids_h5_scan_backend` extended (or new subclass) for voxel-space data
- May require backend contract revision to distinguish spatial vs feature dims
- Lazy voxel selection via `fmri_series()` selectors

### Phase 3: Convenience and polish
- `bids()` facade function (from existing `data-raw/bids_integration_proposal.md`)
- Incremental update support (add subjects to existing archive)
- Parallel write option
- HDF5 chunk sizing optimization
- Vignette and pkgdown article

---

## Key Constraints & Non-Goals (Phase 1)

- **No voxel selectors on parcellated data.** Parcels are the feature space. `index_selector()` works (select parcels by column index), but ROI/sphere/voxel selectors do not apply. This is honest about what parcellated data is.
- **TR must be constant across all scans.** `fmri_study_dataset` enforces this. Writer validates upfront.
- **No neuroarchive dependency.** Schema leaves a seam; code doesn't touch it.
- **No NSE filtering.** `subset_bids_h5()` uses standard evaluation with named arguments.
- **No incremental writes.** The archive is written once from a complete BIDS directory.

## Verification Checklist

1. Round-trip: mock BIDS → write → read → verify events, data, metadata
2. `get_data_matrix()` returns `[T, K]` with correct K = n_parcels
3. `validate_backend()` passes on `bids_h5_scan_backend`
4. `study_backend` correctly composes per-scan backends
5. `subset_bids_h5(task = ...)` produces valid study_dataset
6. `event_table` has task, session, subject_id, run columns
7. `data_chunks()` iterates correctly over study
8. `as_delarr()` produces correct `[T_total, K]` lazy matrix
9. Missing bidser/fmristore/hdf5r produces clear error
10. R CMD check passes
