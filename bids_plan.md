# Compressed BIDS Study Archive: Architecture Proposal

## The Problem

An fMRI study lives as thousands of files across a BIDS directory tree — NIfTI images, events.tsv files, confounds, JSON sidecars. You want to capture an entire study in **a single compressed HDF5 file** that is queryable by subject/task/session/run, preserves the experimental design (events), and supports multiple compression strategies.

## The Ecosystem (as explored)

```
Tier 1: neuroim2              (spatial foundation — NeuroVec, NeuroVol, NeuroSpace)
Tier 2: bidser                (BIDS project querying — bids_project, read_events, preproc_scans)
        fmrihrf               (temporal structure — sampling_frame, blocklens)
Tier 3: fmristore             (HDF5 parcellated storage — H5ParcellatedMultiScan, cluster averages)
        neuroarchive           (HDF5 transform pipelines — LNA format, basis/quant/HRBF compression)
Tier 4: fmridataset            (unified dataset interface — backends, event_table, fmri_group)
```

**Current integration state: zero runtime coupling between the 4 packages.** They share neuroim2 as a common foundation but have no bridge code connecting them.

## Key Findings

### fmridataset (~/code/fmridataset)
- **Already the interface layer** — pluggable backends (matrix, nifti, h5, zarr, latent, study)
- Has `event_table` for experimental design, `sampling_frame` for temporal structure
- Has `fmri_study_dataset` for multi-subject containers, `fmri_group` for group operations
- Has `fmri_series` with lazy voxel selection (index, ROI, sphere selectors)
- Has chunked iteration (`data_chunks`) for memory-efficient processing
- Suggests both `bidser` and `fmristore` — designed for optional integration
- **Has an unimplemented BIDS integration proposal** in `data-raw/bids_integration_proposal.md`

### neuroarchive (~/code/neuroarchive)
- LNA v2.0 format with transform pipelines (quant, basis, embed, temporal, HRBF, etc.)
- 10-500x compression with controllable quality
- `LNANeuroVec` for lazy reconstructed access with ROI/time subsetting
- `lna_dataset` class for BIDS-derivative directory structure
- **No parcellation support**, no events storage
- Compression is voxel-level (reconstructable to original resolution)

### fmristore (~/code/fmristore)
- `H5ParcellatedMultiScan` for clustered multi-run storage
- `summarize_by_clusters()` for computing parcel averages
- Simple, direct HDF5 storage — no transform pipelines
- **Full parcellation support**, no events storage, no BIDS awareness

### bidser (~/code/bidser)
- `bids_project` S3 class — query subjects, tasks, sessions, runs
- `preproc_scans()` — find fmriprep preprocessed NIfTI files
- `read_events()` — read events.tsv into nested tibbles
- `read_confounds()` — read confound regressors with version-aware aliasing
- Completely independent of the other 3 packages

## Recommendation: Build in fmridataset, not a new package

**fmridataset is the natural home** because:

1. It's the highest-level interface (Tier 4) — designed to abstract over storage
2. Its pluggable backend architecture is exactly the extension mechanism we need
3. It already has `event_table`, `sampling_frame`, `fmri_study_dataset`, `fmri_group`
4. It already Suggests both `bidser` and `fmristore`
5. Adding a new backend is the idiomatic extension pattern
6. Avoids creating yet another package with complex dependency management

**fmristore and neuroarchive stay as-is** — they provide compression infrastructure that the new backend delegates to.

## Architecture

```
User API (fmridataset)
├── compress_bids_study()          <- WRITER: bids_project -> .h5
├── bids_h5_dataset()              <- READER: .h5 -> fmri_study_dataset
└── bids_h5_backend                <- BACKEND: serves data from .h5 file

         ┌──────────────────┐
         │  compress mode?  │
         └────┬────────┬────┘
              │        │
     parcellated     lna
              │        │
      fmristore    neuroarchive
   (cluster avg)  (transform pipeline)
```

### Two Compression Modes

| Mode | Backing | Compression | Reconstruction | Use Case |
|:-----|:--------|:------------|:---------------|:---------|
| **`parcellated`** | fmristore | 100-1000x | Parcel averages only | ROI analyses, connectivity, most fMRI analyses |
| **`lna`** | neuroarchive | 10-500x | Full voxel resolution | Searchlight, fine-grained spatial analyses |

Both modes produce the same top-level HDF5 layout. The `/scans/<name>/data/` group differs by mode.

## HDF5 Schema

```
/
├── [attrs: format = "bids_h5_study", version = "1.0"]
├── [attrs: compression_mode = "parcellated" | "lna"]
│
├── bids/                              # Study-level metadata
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
├── spatial/                           # Shared spatial reference
│   ├── header/                        # NIfTI-like (dim, pixdim, qform)
│   ├── mask                           # [X, Y, Z] uint8
│   └── voxel_coords                   # [N_voxels, 3] int32
│
├── parcellation/                      # (parcellated mode only)
│   ├── cluster_map                    # [N_voxels] int32
│   ├── cluster_ids                    # unique IDs
│   └── cluster_meta/                  # labels, network names, etc.
│
├── compression/                       # (LNA mode only)
│   ├── transforms/                    # Transform descriptors (JSON)
│   └── basis/                         # Shared basis matrices
│
├── scans/                             # Per-run data
│   ├── sub-01_task-rest_run-01/
│   │   ├── data/
│   │   │   ├── summary_data           # [T, K] float32 (parcellated)
│   │   │   └── values | quantized     # [T, V] (LNA mode)
│   │   ├── events/                    # Column-per-dataset
│   │   │   ├── onset                  # float64
│   │   │   ├── duration               # float64
│   │   │   ├── trial_type             # string
│   │   │   └── ...                    # task-specific columns
│   │   ├── confounds/                 # Optional
│   │   │   └── data                   # [T, n_confounds] float64
│   │   │       [attrs: names = c(...)]
│   │   └── metadata/
│   │       ├── subject, session, task, run
│   │       └── tr                     # repetition time
│   └── sub-02_task-rest_run-01/
│       └── ...
│
└── scan_index/                        # Fast lookup (avoid scanning groups)
    ├── scan_name                      # string array
    ├── subject, session, task, run    # string arrays
    ├── n_time                         # int array
    ├── has_events                     # logical
    └── has_confounds                  # logical
```

## API Design

### Writer

```r
compress_bids_study <- function(
  x,                            # bidser::bids_project or BIDS directory path
  file,                         # output .h5 path
  mode = c("parcellated", "lna"),

  # Parcellated mode:
  clusters = NULL,              # ClusteredNeuroVol (atlas in MNI space)
  summary_fun = mean,

  # LNA mode:
  transforms = c("basis", "quant"),
  transform_params = list(),

  # Common:
  mask = NULL,                  # LogicalNeuroVol (default: from clusters or fmriprep)
  space = "MNI152NLin2009cAsym",
  tasks = NULL,                 # character filter
  subjects = NULL,              # character filter
  sessions = NULL,              # character filter
  confounds = NULL,             # confound_set() or character vector
  compression = 4L,
  verbose = TRUE
)
# Returns: bids_h5_dataset object (reader for the newly created file)
```

**Internal workflow:**
1. Query bidser for scan manifest (all subject/session/task/run combos)
2. Write shared spatial infrastructure (mask, header, parcellation or basis)
3. For each scan:
   - Read preprocessed NIfTI via `neuroim2::read_vec(scan_path, mask)`
   - Compress: `summarize_by_clusters()` (parcellated) or `write_lna()` (LNA)
   - Read events via `bidser::read_events()` -> store as column arrays
   - Read confounds via `bidser::read_confounds()` -> store as matrix
4. Write `/bids/` metadata + `/scan_index/` lookup table
5. Return reader object

### Reader

```r
bids_h5_dataset <- function(file, preload = FALSE)
# Returns: fmri_study_dataset with:
#   - event_table: combined events from all runs
#   - sampling_frame: temporal structure from TR + run lengths
#   - backend: bids_h5_backend (lazy HDF5 access)
#
# The result integrates seamlessly with fmridataset's existing API.
```

### Query & Access

```r
study <- bids_h5_dataset("my_study.h5")

# BIDS metadata
participants(study)           # -> c("01", "02", ...)
tasks(study)                  # -> c("rest", "nback")
sessions(study)               # -> c("pre", "post") or NULL

# Standard fmridataset API works
get_TR(study)                 # -> 2.0
n_runs(study)                 # -> total runs across all subjects
n_timepoints(study)           # -> total timepoints

# Access specific subject's data
sub01 <- filter_study(study, subject == "01")
mat <- get_data_matrix(sub01) # -> [T, K] matrix (parcellated) or [T, V] (LNA)

# Events
study$event_table             # -> all events, all subjects

# Lazy voxel selection (LNA mode)
sel <- roi_selector(my_roi_mask)
series <- fmri_series(study, selector = sel)

# Group operations
group <- as_fmri_group(study)
results <- group_map(group, function(ds) {
  mat <- get_data_matrix(ds)
  connectivity <- cor(mat)
  # ...
})
```

## New Code in fmridataset

| File | Contents |
|:-----|:---------|
| `R/bids_h5_backend.R` | **NEW** -- `bids_h5_backend` class implementing backend contract |
| `R/bids_h5_dataset.R` | **NEW** -- `bids_h5_dataset()` constructor, query methods |
| `R/bids_h5_write.R` | **NEW** -- `compress_bids_study()` writer function |
| `R/bids_h5_events.R` | **NEW** -- Event read/write helpers for HDF5 |
| `DESCRIPTION` | Add `neuroarchive` to Suggests |
| `tests/testthat/test_bids_h5.R` | **NEW** -- Tests using `bidser::create_mock_bids()` |

## Existing Infrastructure to Reuse

| What | From Package | Function/Class |
|:-----|:-------------|:---------------|
| BIDS querying | bidser | `bids_project()`, `preproc_scans()`, `read_events()`, `read_confounds()` |
| Parcel averaging | fmristore | `summarize_by_clusters()` |
| HDF5 writing | fmristore | `h5_write()`, `ensure_h5_groups()`, `build_nifti_header()` |
| Transform compression | neuroarchive | `write_lna()`, `read_lna()` |
| Temporal structure | fmrihrf | `sampling_frame()` |
| Dataset interface | fmridataset | `fmri_study_dataset()`, `fmri_group()`, backend contract |
| Spatial objects | neuroim2 | `NeuroVec`, `LogicalNeuroVol`, `ClusteredNeuroVol` |

## Implementation Phases

### Phase 1: Parcellated mode (v1)
- `compress_bids_study()` with `mode = "parcellated"`
- `bids_h5_backend` for reading parcellated data
- `bids_h5_dataset()` constructor
- Events read/write
- Tests with mock BIDS data

### Phase 2: LNA mode (v2)
- `compress_bids_study()` with `mode = "lna"`
- Extend `bids_h5_backend` for LNA-compressed data
- Lazy voxel selection via neuroarchive's inverse pipeline
- Shared basis support

### Phase 3: Convenience (v3)
- `bids()` facade function (from the existing fmridataset proposal)
- pkgdown articles / vignettes
- Performance optimization for large studies (parallel write, streaming)

## Why Not a New Package?

| Concern | Answer |
|:--------|:-------|
| Dependency bloat in fmridataset? | All new deps are already Suggests (bidser, fmristore). Adding neuroarchive to Suggests costs nothing. |
| Scope creep? | The backend pattern is fmridataset's explicit extension mechanism. |
| Could split later? | Yes, if the code grows large, it can be extracted. Starting integrated is simpler. |
| Maintenance? | One package to maintain vs. coordinating across two. |

## Verification

1. Round-trip test: `create_mock_bids()` -> `compress_bids_study()` -> `bids_h5_dataset()` -> verify events, data, metadata
2. `get_data_matrix()` returns correct dimensions
3. `event_table` matches original events.tsv content
4. `participants()`, `tasks()`, `sessions()` return correct values
5. Group operations (`fmri_group`) work on the result
6. R CMD check passes on fmridataset with new code
