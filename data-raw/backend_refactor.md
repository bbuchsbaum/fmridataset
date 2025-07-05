
### **Revised & Finalized Ticketed Sprint Plan: A Resilient, Pluggable `fmridataset`**

This sprint refactors `fmridataset` into a storage-agnostic container system built on an explicit and robust `StorageBackend` contract. It incorporates key architectural decisions around data orientation, metadata handling, and error semantics from the outset.

**Sprint Goal:** Refactor `fmridataset` to use a pluggable backend system. Prove the architecture's success by implementing a battle-hardened `NiftiBackend` and a proof-of-concept `Hdf5Backend`, complete with clear documentation and performance assertions.

---

#### **Ticket FDT-01: Design & Define the `StorageBackend` S3 Contract**
*   **Goal:** Create a minimal, explicit, and robust S3 interface that all storage backends must adhere to. This is the foundational design work.
*   **Acceptance Criteria:**
    1.  A new file, `R/storage_backend.R`, is created.
    2.  It defines S3 generics for the full backend lifecycle and data access:
        *   `backend_open(backend)` & `backend_close(backend)`: Manages stateful resources like file handles. Stateless backends can have no-op implementations.
        *   `backend_get_dims(backend)`: Returns a named list `list(spatial = c(x,y,z), time = N)`.
        *   `backend_get_mask(backend)`: Returns a logical vector. **Must conform to two invariants:**
            *   `length(mask) == prod(backend_get_dims(backend)$spatial)`
            *   `sum(mask) > 0` (no empty masks allowed). `NA` values in the mask are disallowed.
        *   `backend_get_data(backend, rows = NULL, cols = NULL)`: Reads data.
            *   Returns data in a canonical **timepoints × voxels** orientation.
            *   `rows` and `cols` are integer vectors for slicing; `NULL` implies all.
        *   `backend_get_metadata(backend)`: Returns a list containing essential neuroimaging metadata (e.g., affine matrix, voxel sizes, intent codes).
    3.  A custom S3 error class hierarchy is defined in `R/errors.R`:
        *   `fmridataset_error` (base class).
        *   `fmridataset_error_backend_io` (for read/write failures).
        *   `fmridataset_error_config` (for invalid setup).
    4.  All generics and error classes are documented with Roxygen.

---

#### **Ticket FDT-02: Implement `NiftiBackend` from Existing Code**
*   **Goal:** Refactor the current `fmri_file_dataset` logic into the first concrete `StorageBackend`, making it more flexible.
*   **Acceptance Criteria:**
    1.  A new constructor, `nifti_backend(source, mask_source, preload, mode)`, is created.
        *   `source` can be a character vector of file paths *or* a list of in-memory `NeuroVec` objects.
        *   `mask_source` can be a file path or an in-memory `NeuroVol` object.
    2.  All methods from the `StorageBackend` contract (`FDT-01`) are implemented.
    3.  `backend_get_metadata()` extracts and returns the affine matrix and other key header fields from the NIfTI source.
    4.  I/O operations from `neuroim2` are wrapped in `tryCatch`, re-throwing errors as `fmridataset_error_backend_io` with informative context.

---

#### **Ticket FDT-03: Refactor `fmri_dataset` to Use a Backend**
*   **Goal:** Modify the high-level `fmri_dataset` to be a thin, stateless wrapper that delegates all I/O to its backend.
*   **Acceptance Criteria:**
    1.  The existing `fmri_file_dataset` is renamed to `fmri_dataset_legacy` to allow for side-by-side comparison during the transition.
    2.  The `fmri_dataset` constructor is simplified to accept a `backend` object.
    3.  `get_data_matrix`, `get_mask`, etc., are refactored to be thin wrappers around `backend_get_data` and `backend_get_mask`.
    4.  All existing tests for file-based datasets are updated to use the new backend architecture and continue to pass.

---

#### **Ticket FDT-04: Update `data_chunks` to be Backend-Aware & Performant**
*   **Goal:** Ensure the chunking mechanism streams data directly from the backend without loading the entire dataset into memory, and prove it with benchmarks.
*   **Acceptance Criteria:**
    1.  The `data_chunks` method for backend-based datasets is updated.
    2.  The `get_seq_chunk` and `get_run_chunk` internal functions now call `backend_get_data(backend, rows = ..., cols = ...)` to read only the necessary data slice for that chunk.
    3.  A new test using `bench::mark()` is added to assert that iterating through chunks **does not** allocate memory proportional to the full 4D array size, catching accidental eager reads. The test should show `mem_alloc` is closer to the size of one chunk, not the whole dataset.

---

#### **Ticket FDT-05: Proof-of-Concept: Implement a Realistic `Hdf5Backend`**
*   **Goal:** Prove the architecture's extensibility by adding a new file format backend that handles common performance and concurrency issues.
*   **Acceptance Criteria:**
    1.  A new `hdf5_backend` constructor is created (adds `rhdf5` to `Suggests`).
    2.  It implements the `StorageBackend` contract. `backend_open` acquires a file handle, and `backend_close` releases it. This design ensures thread-safety in parallel loops (each worker calls `backend_open`).
    3.  An integration test is added that:
        1.  Creates a `matrix_dataset`.
        2.  Writes its data to a temporary HDF5 file **with gzip compression and chunking by voxel**.
        3.  Loads it via `fmri_dataset(backend = h5_backend(...))`.
        4.  Asserts data integrity after the round-trip.

---

#### **Ticket FDT-06: Codebase Housekeeping and Contributor Tooling**
*   **Goal:** Formalize coding standards and cleanup leftover artifacts to create a welcoming environment for external contributors.
*   **Acceptance Criteria:**
    1.  The `tests/run_tests.R` script is pruned to source only existing, relevant files from the refactored structure.
    2.  A GitHub Action for `usethis::use_lintr_ci()` is added to the repository.
    3.  A `CONTRIBUTING.md` file is created, specifying key style guidelines (e.g., line length, pipe style) and pointing to the `lintr` configuration.
    4.  All internal helper functions are correctly marked with `@keywords internal`.

---

#### **Ticket FDT-07: Create 'Extending `fmridataset`' Vignette with Visuals**
*   **Goal:** Create clear, visual, and actionable documentation for the new pluggable architecture.
*   **Acceptance Criteria:**
    1.  A new vignette (`vignettes/extending-backends.Rmd`) is created.
    2.  It includes a **Mermaid.js diagram** illustrating the data flow: `fmri_dataset` -> `StorageBackend` -> `Physical Storage`.
    3.  The vignette provides a complete, commented walkthrough of how to build a simple `CsvBackend`.
    4.  The vignette explicitly documents the `StorageBackend` contract, including data orientation and metadata requirements.

---

#### **Future-Proofing Considerations (Post-Sprint)**
While not part of this sprint, the following concepts are acknowledged to guide the design:
*   **Multi-Subject Collections:** The S3 class name `fmri_study` (or `fmri_collection`) will be reserved for a future milestone to manage lists of `fmri_dataset` objects.
*   **Asynchronous I/O:** The `backend_get_data` contract is synchronous for now but does not preclude a future `async_backend` that returns a `promise` or `future` object, enabling cloud-native backends (S3/Zarr).
*   **Python Interoperability:** The clean separation of concerns and explicit error classes will facilitate creating `reticulate` wrappers around the backend system.

---

#### **Addendum: Latent Backend - Special Data Access Semantics**

A specialized `latent_backend` has been implemented to handle LatentNeuroVec objects from the fmristore package. This backend has **fundamentally different data access semantics** compared to other backends:

**Key Differences:**
*   **Data Orientation:** Returns **latent scores** (temporal basis components), not reconstructed voxel data
*   **Matrix Dimensions:** Data is (time × components), not (time × voxels)  
*   **Mask Meaning:** Logical vector indicating active components, not spatial voxels
*   **Purpose:** Enables efficient analysis in compressed latent space

**Rationale:**
The latent backend is designed for workflows that operate directly on compressed representations. Since LatentNeuroVec stores data as `data = basis %*% t(loadings) + offset`, returning the full reconstructed voxel data would:
1. Defeat the purpose of compression
2. Be computationally inefficient  
3. Not align with typical analysis workflows in latent space

**Usage Pattern:**
```r
# Standard backends return voxel data
nifti_data <- backend_get_data(nifti_backend)  # dim: [time, voxels]

# Latent backend returns latent scores  
latent_data <- backend_get_data(latent_backend)  # dim: [time, components]
```

This design decision prioritizes computational efficiency and aligns with the mathematical structure of latent representations. Users needing full voxel reconstruction should use fmristore's reconstruction methods directly.