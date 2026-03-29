# fmrilatent Integration Plan

## Goal

Enable fmridataset to work directly with fmrilatent's latent representations, both as in-memory objects and as a compression mode in the BIDS H5 archive.

## Context

**fmrilatent** (~/code/fmrilatent) provides:
- `LatentNeuroVec` S4 class: `basis [T,K] × loadings [V,K] + offset [V]`
- 14+ encoding methods via `encode()` + spec objects (PCA, Slepian, DCT, wavelets, HRBF, hierarchical templates)
- Template system for reusable spatial dictionaries across subjects
- Lazy evaluation via BasisHandle/LoadingsHandle with registry caching
- No HDF5 storage — pure compute + in-memory, RDS for template persistence

**Current fmridataset state:**
- `latent_backend` already accepts `LatentNeuroVec` objects (line 73), but accesses slots assuming fmristore's class (direct `@basis`/`@loadings` matrix access)
- fmrilatent's `LatentNeuroVec` has the same slot names but may use `BasisHandle`/`LoadingsHandle` (lazy) instead of concrete matrices
- `latent_dataset` provides `get_latent_scores()`, `get_spatial_loadings()`, `reconstruct_voxels()` — exactly the right API for latent data
- BIDS H5 archive (Phase 1) supports `compression_mode = "parcellated"` with a clean seam for new modes

**Relationship between packages:**
```
fmrilatent    = COMPUTE (encode fMRI → LatentNeuroVec)
neuroarchive  = STORAGE (compress + HDF5 pipeline, deferred)
fmridataset   = INTERFACE (backends, datasets, study operations, BIDS H5 archive)
```

---

## Phase A: Direct Bridge (fmrilatent → fmridataset)

### Scope

Make `latent_backend` and `latent_dataset` work seamlessly with fmrilatent's `LatentNeuroVec` objects, without requiring fmristore.

### Problem

`backend_open.latent_backend` (latent_backend.R:109) unconditionally requires fmristore:
```r
if (!requireNamespace("fmristore", quietly = TRUE)) stop(...)
```

Then it reads data via `fmristore::read_vec()` for file paths. For in-memory objects, it accesses slots directly (`obj@basis`, `obj@loadings`). This works when the basis/loadings are concrete matrices, but fmrilatent may store `BasisHandle`/`LoadingsHandle` objects that need materialization.

### Changes

**A1. Conditional fmristore dependency in latent_backend**

`backend_open.latent_backend` should only require fmristore when source contains file paths. When source is a list of in-memory `LatentNeuroVec` objects, fmristore is not needed.

```r
# Current: always requires fmristore
if (!requireNamespace("fmristore", quietly = TRUE)) stop(...)

# Proposed: only require fmristore for file-path sources
needs_fmristore <- is.character(backend$source) ||
  any(vapply(backend$source, is.character, logical(1)))
if (needs_fmristore && !requireNamespace("fmristore", quietly = TRUE)) stop(...)
```

**A2. Handle fmrilatent's lazy basis/loadings**

When accessing `@basis` and `@loadings`, use fmrilatent's accessor generics (`basis()`, `loadings()`) instead of direct slot access. These generics handle both concrete matrices and BasisHandle/LoadingsHandle transparently.

```r
# Current: direct slot access
first_ncomp <- ncol(data[[1]]@basis)
n_voxels <- nrow(data[[1]]@loadings)

# Proposed: use S4 generics (work for both fmristore and fmrilatent)
first_ncomp <- ncol(fmrilatent::basis(data[[1]]))
n_voxels <- nrow(fmrilatent::loadings(data[[1]]))
```

But we can't import fmrilatent generics directly (it's Suggests only). Solution: use `methods::slot()` as fallback, or define local accessor helpers that try fmrilatent first, then fall back to direct slot access.

Better approach — use a local helper:
```r
.get_basis_matrix <- function(obj) {
  # Try as.matrix first (works for both Handle and matrix types)
  b <- obj@basis
  if (is.matrix(b) || inherits(b, "Matrix")) return(b)
  # If it's a Handle, materialize it
  if (requireNamespace("fmrilatent", quietly = TRUE)) {
    return(as.matrix(fmrilatent::basis(obj)))
  }
  as.matrix(b)
}
```

**A3. Validate slot structure rather than package origin**

The `latent_backend` constructor checks `inherits(item, "LatentNeuroVec")`. Both fmristore and fmrilatent define this class. The backend should work with either, validating by slot structure:
- Has `basis` slot (matrix-like, T × K)
- Has `loadings` slot (matrix-like, V × K)
- Has `offset` slot (numeric vector or NULL)
- Inherits from `NeuroVec` (neuroim2 base class)

**A4. Add fmrilatent to Suggests**

In DESCRIPTION, add `fmrilatent` to Suggests.

**A5. Documentation update**

Update `latent_backend` and `latent_dataset` roxygen to mention fmrilatent as a supported source.

### Acceptance Criteria

- [ ] `latent_dataset(source = list(fmrilatent_lvec), TR = 2, run_length = 100)` works without fmristore installed
- [ ] `get_latent_scores()` returns correct `[T, K]` matrix from fmrilatent objects
- [ ] `get_spatial_loadings()` returns correct loadings from fmrilatent objects
- [ ] `reconstruct_voxels()` works on fmrilatent-sourced latent_dataset
- [ ] `get_component_info()` returns valid metadata
- [ ] Objects with BasisHandle/LoadingsHandle (lazy) are handled correctly
- [ ] Objects with concrete matrices (explicit) also still work
- [ ] fmristore-based .lv.h5 file path workflow is unbroken
- [ ] `validate_backend()` passes on fmrilatent-sourced latent_backend
- [ ] R CMD check passes with fmrilatent in Suggests

---

## Phase B: Latent Mode in BIDS H5 Archive

### Scope

Add `compression_mode = "latent"` to the BIDS H5 archive, using fmrilatent's `encode()` to compress each scan.

### Prerequisite

Phase A complete (latent_backend works with fmrilatent objects).

### HDF5 Schema Extension

```
/scans/<name>/data/                    # latent mode
  ├── basis                            # [T, K] float64 — temporal scores
  ├── loadings                         # [V, K] float32 — spatial loadings
  ├── offset                           # [V] float32 — per-voxel offset
  ├── [attrs: k = K]
  ├── [attrs: encoding_family = "dct"] # which fmrilatent spec was used
  └── [attrs: encoding_params = "{}"]  # JSON of spec parameters
```

Root attribute: `compression_mode = "latent"`

The `/spatial/` group stores the full brain mask and voxel geometry (same as parcellated mode). The `/parcellation/` group is absent. A new `/latent_meta/` group stores shared encoding info:

```
/latent_meta/
  ├── encoding_family                  # string: "dct", "slepian", "pca", etc.
  ├── encoding_params                  # JSON string of spec parameters
  ├── n_components                     # integer K
  └── template/                        # Optional: shared template (RDS blob or H5 groups)
```

### Writer Extension

```r
compress_bids_study <- function(
  x, file,
  mode = c("parcellated", "latent"),  # extended

  # Parcellated mode (existing):
  clusters = NULL,
  summary_fun = mean,

  # Latent mode (new):
  encoding = NULL,         # fmrilatent spec object, e.g. spec_time_dct(k=15)
  n_components = NULL,     # shorthand: if encoding is NULL, use PCA with this K
  template = NULL,         # optional: shared HierarchicalBasisTemplate

  # Common (existing):
  mask = NULL, space = "MNI152NLin2009cAsym",
  tasks = NULL, subjects = NULL, sessions = NULL,
  confounds = NULL, compression = 4L, verbose = TRUE
)
```

Latent mode workflow (per scan, streaming):
1. Read NIfTI via `neuroim2::read_vec(scan_path)`
2. Extract masked matrix `[T, V]`
3. Encode: `fmrilatent::encode(mat, encoding, mask = mask_vol)`
4. Extract: `basis [T,K]`, `loadings [V,K]`, `offset [V]`
5. Write to `/scans/<name>/data/basis`, `loadings`, `offset`
6. Write events, confounds, censor (same as parcellated)
7. Release memory

### Backend Extension

`bids_h5_scan_backend` dispatches on compression_mode:
- `"parcellated"` → reads `summary_data` → `[T, K]` (existing)
- `"latent"` → reads `basis` → `[T, K]` (same contract!)

Both modes return `[T, K]` from `backend_get_data()`. The backend contract is unchanged.

For latent mode, additional methods on `bids_h5_study_dataset`:
- `get_loadings(study, scan_name = NULL)` → `[V, K]` loadings matrix
- `reconstruct_voxels(study, rows = NULL, voxels = NULL)` → `[T, V_subset]` reconstructed data
- `encoding_info(study)` → list of encoding family, params, K

### Reader: Same Two-Level Composition

The `bids_h5_dataset()` reader works identically for latent mode — per-scan backends grouped by subject. The only difference is what data the backend reads from H5.

### Key Design Decision: Consistent K Across Scans

Same constraint as parcellated: all scans must have the same K (number of components). This is natural when using the same encoding spec across scans. Heterogeneous K would break `study_backend` composition.

However, loadings **can differ** per scan (each scan is encoded independently). Shared loadings via a template (Phase B2) is an optimization, not a requirement.

### Acceptance Criteria

- [ ] `compress_bids_study(mode = "latent", encoding = spec_time_dct(k=15))` produces valid H5
- [ ] H5 file contains `/scans/<name>/data/basis`, `loadings`, `offset`
- [ ] `bids_h5_dataset()` reads latent-mode archives correctly
- [ ] `get_data_matrix()` returns `[T_total, K]` (latent scores)
- [ ] `subset_bids_h5(task = ...)` works on latent-mode archives
- [ ] `get_loadings(study, scan_name = ...)` returns per-scan loadings
- [ ] `reconstruct_voxels(study, rows = 1:10, voxels = roi)` works
- [ ] `encoding_info(study)` returns correct encoding metadata
- [ ] Parcellated mode is completely unaffected
- [ ] Round-trip: mock BIDS → compress (latent) → read → verify scores match direct encode()
- [ ] Sparse loadings stored efficiently (CSC or compressed H5)
- [ ] R CMD check passes

---

## Phase B2: Shared Templates (Optional Enhancement)

When all scans use the same spatial template (e.g., `parcel_basis_template()` or `build_hierarchical_template()`), the loadings are derivable from the template + a projection coefficient matrix. This can dramatically reduce H5 file size.

```
/latent_meta/template/
  ├── loadings          # [V, K_template] — shared spatial dictionary
  ├── gram_factor       # Cholesky/QR factor for projection
  └── meta              # JSON: basis_spec, reduction, center flag
```

Per-scan data becomes just `[T, K]` projection coefficients (no per-scan loadings needed).

### Acceptance Criteria

- [ ] `compress_bids_study(mode = "latent", template = my_template)` uses shared template
- [ ] Per-scan data is only `[T, K]` coefficients (no per-scan loadings)
- [ ] Template stored once in `/latent_meta/template/`
- [ ] File size significantly smaller than per-scan loadings approach
- [ ] Reconstruction still works via template loadings × coefficients + offset

---

## Dependency Summary

| Phase | Requires | Adds to Suggests |
|:------|:---------|:-----------------|
| A (bridge) | fmrilatent | fmrilatent |
| B (BIDS H5 latent) | fmrilatent + Phase A | (already added) |
| B2 (shared templates) | fmrilatent + Phase B | (already added) |

neuroarchive remains deferred. It could eventually provide an alternative encoding pipeline that produces `LatentNeuroVec` objects, feeding into the same latent_backend/BIDS H5 infrastructure.

---

## Implementation Order

1. **Phase A** — Direct bridge (~100 LOC, 1-2 sessions)
2. **Phase B** — Latent BIDS H5 mode (~400 LOC, 2-3 sessions)
3. **Phase B2** — Shared templates (~200 LOC, 1 session)

Phase A unblocks fmrilatent users immediately. Phase B builds on the BIDS H5 infrastructure from the parcellated work. Phase B2 is an optimization that matters for large studies.
