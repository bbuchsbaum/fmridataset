Okay, this is a fantastic meta-review and provides excellent direction. I'll synthesize this into a complete proposal with granular tickets, focusing on the `fmridataset` package itself, assuming `fmrireg` will adapt to use it.

The core idea is a new, standalone `fmridataset` package.

---

# **`fmridataset` Package: Complete Proposal & Tickets**

## **I. Core Philosophy & Design Goals**

*   **Simplicity & Clarity:** The `fmri_dataset` object should be a straightforward container for fMRI data and essential metadata. Its structure should be intuitive.
*   **Immutability:** Once created, an `fmri_dataset` object represents a specific, unchanging dataset configuration. Modifications should result in new objects if necessary (though primarily, this object is for reading/accessing).
*   **Versatility:** Accommodate data from various sources (raw files, BIDS projects, in-memory matrices, pre-loaded `NeuroVec` objects).
*   **Lazy Loading by Default:** For file-based data, images and masks should only be loaded into memory when explicitly requested by an accessor function (e.g., `get_data_matrix()`), with optional preloading.
*   **Loose Coupling with `bidser`:** `bidser` integration is optional and enabled only if the package is installed.
*   **Backward Compatibility (for `fmrireg`):** The structure and accessors should allow `fmrireg` to easily adapt its existing `fmri_dataset` usage by creating an `fmridataset` object via its new constructor.

## **II. S3 Class: `fmri_dataset` Structure**

The `fmri_dataset` will be an S3 object (a named list) with the following core components:

```
structure(
  list(
    # --- Data Sources (One of these will be primary) ---
    image_paths = NULL,      # Character vector: Full paths to NIfTI image files
    image_objects = NULL,    # List: Pre-loaded NeuroVec/NeuroVol objects
    image_matrix = NULL,     # Matrix: In-memory data matrix (time x voxels)

    mask_path = NULL,        # Character: Full path to NIfTI mask file
    mask_object = NULL,      # LogicalNeuroVol: Pre-loaded mask object
    mask_vector = NULL,      # Logical vector/matrix for image_matrix

    # --- Essential Metadata & Structure ---
    sampling_frame = NULL,   # 'sampling_frame' object (TR, blocklens, etc.)
    event_table = NULL,      # 'tibble' of event data (onset, duration, trial_type, etc.)
    censor_vector = NULL,    # Optional: Logical or numeric vector for censoring

    # --- Descriptive & Provenance Metadata ---
    metadata = list(
      dataset_type = NULL,   # Character: "file_vec", "memory_vec", "matrix", "bids_file", "bids_mem"
      source_description = NULL, # Character: Origin of the data (e.g., BIDS project path, list of files)
      TR = NULL,             # Numeric: Repetition Time (redundant with sampling_frame but useful)
      base_path = NULL,      # Character: Base path for relative file paths if image_paths used
      # BIDS-specific metadata (populated by as.fmri_dataset.bids_project)
      bids_info = list(
        project_path = NULL,
        subject_id = NULL,
        session_id = NULL,
        task_id = NULL,
        run_ids = NULL, # Could be multiple runs
        image_type_source = NULL # e.g., "raw", "preproc"
      ),
      # File-loading options (if dataset_type == "file_vec" or "bids_file")
      file_options = list(
        mode = "normal",
        preload = FALSE
      ),
      # Matrix preprocessing options
      matrix_options = list(
        temporal_zscore = FALSE,
        voxelwise_detrend = FALSE
      ),
      # User-provided extra metadata
      extra = list()
    ),

    # --- Internal Cache ---
    data_cache = new.env(hash = TRUE, parent = emptyenv()) # For memoized/loaded data
  ),
  class = "fmri_dataset"
)
```

**Key Decisions:**
1.  Multiple `image_*` and `mask_*` slots: Only one set will be populated based on input type, indicated by `metadata$dataset_type`. Accessors will abstract this.
2.  `sampling_frame`: A dedicated S3 object (first-class) holding `TR`, `run_length`, etc. This is fundamental.
3.  `metadata`: Structured for clarity (runtime, BIDS, file options, matrix options, extra).
4.  `data_cache`: For lazy loading and memoization of read data.
5.  **Refined `dataset_type` terminology:** "file_vec" (NIfTI on disk), "memory_vec" (preloaded NeuroVec), "matrix" (plain matrix), "bids_file"/"bids_mem" (BIDS-derived with lazy/preloaded).

## **II.A. First-Class `sampling_frame` Object**

The `sampling_frame` will be its own S3 class with dedicated constructor and accessors:

```r
sampling_frame <- function(run_lengths, TR) {
  structure(
    list(
      run_lengths = run_lengths,
      TR = TR,
      total_timepoints = sum(run_lengths),
      n_runs = length(run_lengths)
    ),
    class = "sampling_frame"
  )
}

# Accessors
n_timepoints.sampling_frame <- function(x, run_id = NULL) {
  if (is.null(run_id)) x$total_timepoints else sum(x$run_lengths[run_id])
}

n_runs.sampling_frame <- function(x) x$n_runs
get_TR.sampling_frame <- function(x) x$TR
get_run_lengths.sampling_frame <- function(x) x$run_lengths
```

## **III. Primary Constructor: `fmri_dataset_create()`**

This will be the main, exported constructor in the `fmridataset` package. (Renamed from `fmri_dataset_new` for user-friendliness).

*   **Signature:**
    ```r
    fmri_dataset_create(
      images,           # Path(s), NeuroVec(s), matrix
      mask,             # Path, NeuroVol, logical vector/matrix
      TR,               # Scalar numeric
      run_lengths,      # Numeric vector
      event_table = NULL, # Tibble or path to TSV
      censor_vector = NULL,
      base_path = ".",  # For relative paths in `images`, `mask`, `event_table`
      # File-specific options (if images/mask are paths)
      image_mode = "normal",
      preload_data = FALSE,
      # Matrix preprocessing options
      temporal_zscore = FALSE,
      voxelwise_detrend = FALSE,
      # Extra metadata
      metadata = list(),
      ... # For future expansion
    )
    ```
*   **Responsibilities:**
    *   Determine `dataset_type` based on `images` and `mask` inputs using refined terminology.
    *   Validate all inputs rigorously based on `dataset_type`.
    *   Create the `sampling_frame` object from `TR` and `run_lengths`.
    *   Process `event_table` (read from path if character, ensure tibble).
    *   Populate the `fmri_dataset` list structure correctly.
    *   Store paths as full, absolute paths internally if `base_path` is used.
    *   Store matrix preprocessing options in `metadata$matrix_options`.
    *   If `preload_data` is `TRUE` and type is file-based, trigger initial load into `data_cache`.

## **III.A. Explicit Preloading Helper**

Separate side-effects from constructors with a dedicated preloading function:

```r
preload_data <- function(x, what = c("images", "mask")) {
  what <- match.arg(what, several.ok = TRUE)
  
  if ("images" %in% what && x$metadata$dataset_type %in% c("file_vec", "bids_file")) {
    # Force load images into data_cache
    get_data_matrix(x)
  }
  
  if ("mask" %in% what && !is.null(x$mask_path)) {
    # Force load mask into data_cache
    get_mask_volume(x)
  }
  
  invisible(x)
}
```

## **IV. `as.fmri_dataset()` Generic and Methods**

*   **`as.fmri_dataset.character(paths, TR, run_lengths, ...)`:**
    *   Calls `fmri_dataset_create()` with `images = paths`.
    *   Requires `TR`, `run_lengths`. Other args like `mask_path`, `event_table` (path or tibble), `base_path` passed through.

*   **`as.fmri_dataset.list(object_list, TR, run_lengths, ...)`:** (If `object_list` contains `NeuroVec`s)
    *   Calls `fmri_dataset_create()` with `images = object_list`.
    *   Requires `TR`, `run_lengths`. Mask must be `NeuroVol` or `NULL`.

*   **`as.fmri_dataset.matrix(data_matrix, TR, run_lengths, ...)`:**
    *   Calls `fmri_dataset_create()` with `images = data_matrix`.
    *   Requires `TR`, `run_lengths`. Mask must be logical matrix/vector or `NULL`.

*   **`as.fmri_dataset.bids_project(bids_proj, subject_id, task_id = NULL, ...)`:**
    *   **Crucial:** Must determine `run_lengths` by reading NIfTI headers (suggests `neuroim2` as a `Suggests` dependency).
    *   Uses `bidser` functions (`func_scans`, `preproc_scans`, `read_events`, `get_repetition_time`, `brain_mask`).
    *   Populates `metadata$bids_info`.
    *   Calls `fmri_dataset_create()`.

## **V. Accessor Functions**

Standard S3 generics where appropriate, or simple `get_*` functions.

*   `get_data_matrix(x, run_id = NULL, apply_preprocessing = TRUE)`:
    *   Primary data access. Returns (time x voxels) matrix.
    *   Handles loading from `image_paths` (via `data_cache` with memoization), extracting from `image_objects`, or returning `image_matrix`.
    *   Applies mask.
    *   Applies censoring if `censor_vector` is present.
    *   If `apply_preprocessing = TRUE`, applies temporal z-scoring and voxelwise detrending based on `metadata$matrix_options`.
    *   Optional `run_id` to get data for specific run(s).
*   `get_mask_volume(x)`: Returns `LogicalNeuroVol` (loads from `mask_path` via `data_cache` or returns `mask_object`).
*   `get_sampling_frame(x)`: Returns the `sampling_frame` object.
*   `get_event_table(x)`: Returns the `event_table` tibble.
*   `get_TR(x)`: Convenience, from `get_TR(get_sampling_frame(x))`.
*   `get_run_lengths(x)`: Convenience, from `get_run_lengths(get_sampling_frame(x))`.
*   `get_num_runs(x)`: Convenience, from `n_runs(get_sampling_frame(x))`.
*   `get_num_voxels(x)`: Number of voxels after masking.
*   `get_num_timepoints(x, run_id = NULL)`: Total or per-run timepoints from `n_timepoints(get_sampling_frame(x), run_id)`.
*   `get_censor_vector(x)`: Returns `censor_vector`.
*   `get_metadata(x, field = NULL)`: Accesses `metadata` list.
*   `get_dataset_type(x)`: Returns `metadata$dataset_type`.
*   `get_image_source_type(x)`: Returns class of primary image data (e.g. "character" for paths, "matrix", "NeuroVec").

## **VI. Iteration & Chunking**

*   `data_chunks(x, nchunks = 1, runwise = FALSE, by = c("voxel", "timepoint"))`:
    *   Generic with method `data_chunks.fmri_dataset`.
    *   Returns an iterator (e.g., using `iterators` package or custom).
    *   Each iteration yields an S3 object of class `fmri_data_chunk`: 
        ```r
        structure(
          list(
            data = matrix_chunk, 
            voxel_indices = ..., 
            timepoint_indices = ..., 
            chunk_info = list(
              chunk_num = ..., 
              total_chunks = ...,
              run_num = ..., # if runwise
              chunk_type = by
            )
          ),
          class = "fmri_data_chunk"
        )
        ```
    *   `by = "voxel"` (default): chunks split voxel dimension. `nchunks` applies.
    *   `by = "timepoint"`: chunks split time dimension. `nchunks` applies.
    *   `runwise = TRUE`: `by` is ignored, one chunk per run (data is time x all_voxels_for_that_run).
    *   Internally calls `get_data_matrix` (potentially for sub-runs if `runwise`) and then splits.
    *   Easy to wrap in `purrr::map()` pipelines.

*   `print.fmri_data_chunk(x, ...)`: Print method for chunk objects.

## **VII. Validation**

*   `validate_fmri_dataset(x)`:
    *   Checks internal consistency:
        *   Sum of `run_lengths` in `sampling_frame` matches total timepoints implied by `image_source` (if loadable).
        *   Mask dimensions compatible with image dimensions.
        *   `event_table` onsets/durations are within `sampling_frame` bounds.
        *   `censor_vector` length matches total timepoints.
    *   Returns `TRUE` or throws informative errors.

## **VIII. S3 Methods (Print, Summary)**

*   `print.fmri_dataset(x, ...)`: User-friendly summary.
*   `summary.fmri_dataset(object, ...)`: More detailed summary statistics.

## **IX. Package Structure (File Layout)**

```
fmridataset/
  R/
    aaa_generics.R            # S3 generics (as.fmri_dataset, get_data_matrix, etc.)
    sampling_frame.R          # sampling_frame class definition and methods
    fmri_dataset_class.R      # Definition of the class structure comments
    fmri_dataset_create.R     # fmri_dataset_create() constructor
    fmri_dataset_from_paths.R # as.fmri_dataset.character()
    fmri_dataset_from_list_matrix.R # as.fmri_dataset.list(), .matrix()
    fmri_dataset_from_bids.R  # as.fmri_dataset.bids_project() (loads if bidser present)
    fmri_dataset_accessors.R  # All get_* functions
    fmri_dataset_iterate.R    # data_chunks() and iterator helpers
    fmri_dataset_validate.R   # validate_fmri_dataset()
    fmri_dataset_print_summary.R # print and summary methods
    fmri_dataset_preload.R    # preload_data() helper
    utils.R                   # Internal helpers (e.g., determine_dataset_type)
    zzz.R                     # .onLoad for conditional method registration
  NAMESPACE
  DESCRIPTION
  tests/
    testthat/
      test-sampling-frame.R
      test-constructor.R
      test-from-paths.R
      test-from-bids.R      # Will need mock bidser project
      test-accessors.R
      test-iterate.R
      test-validate.R
      test-preload.R
      test-chunks.R         # Test fmri_data_chunk class
  vignettes/
    creating_fmri_datasets.Rmd
```

## **X. `fmrireg` Compatibility Layer**

In the `fmrireg` package:
1.  Add `fmridataset` to `Imports`.
2.  Modify `fmrireg::fmri_dataset()` to be a wrapper:
    ```r
    fmrireg::fmri_dataset <- function(scans, mask, TR, run_length, event_table=data.frame(),
                                      base_path=".", censor=NULL, preload=FALSE,
                                      mode=c("normal", "bigvec", "mmap", "filebacked")) {
      fmridataset::fmri_dataset_create(images = scans, mask = mask, TR = TR, run_lengths = run_length,
                                     event_table = event_table, censor_vector = censor, base_path = base_path,
                                     image_mode = match.arg(mode), preload_data = preload)
    }
    ```
3.  Similarly, wrap `fmri_mem_dataset`, `matrix_dataset`, `latent_dataset` to call `fmridataset::fmri_dataset_create()` with appropriate argument mapping to `images`, `mask`, and `metadata$dataset_type`.
4.  Update all internal `fmrireg` code that uses the old `fmri_dataset` structure to use the new accessor functions (e.g., `get_data_matrix(dset)` instead of `dset$datamat` or `series(dset$scans, ...)`).

## **XI. Granular Tickets (GitHub Issues)**

**Core Object & Constructor:**
*   `#1`: Design: Finalize `fmri_dataset` S3 object structure (slots for image/mask sources, sf, events, meta, cache).
*   `#2`: Implement: `sampling_frame` S3 class with constructor and accessors (`n_timepoints`, `n_runs`, etc.).
*   `#3`: Implement: `fmri_dataset_create()` primary constructor with input validation and refined type determination.
*   `#4`: Implement: `determine_dataset_type()` internal helper using refined terminology ("file_vec", "memory_vec", "matrix", "bids_file", "bids_mem").

**`as.fmri_dataset` Converters:**
*   `#5`: Implement: `as.fmri_dataset` S3 generic.
*   `#6`: Implement: `as.fmri_dataset.character()` method for file paths.
*   `#7`: Implement: `as.fmri_dataset.list()` method for pre-loaded `NeuroVec` objects.
*   `#8`: Implement: `as.fmri_dataset.matrix()` method for in-memory matrices.
*   `#9`: Implement: `as.fmri_dataset.bids_project()` method (conditional on `bidser`).
    *   `#9.1`: Sub-task: Logic for determining `run_lengths` from NIfTI headers via `neuroim2` for BIDS source.
    *   `#9.2`: Sub-task: Logic for selecting raw vs. preprocessed images based on `image_type`.

**Accessor Functions:**
*   `#10`: Implement: `get_data_matrix()` method (handles all `dataset_type`s, masking, censoring, lazy loading, preprocessing options).
*   `#11`: Implement: `get_mask_volume()` method (handles path/object, lazy loading).
*   `#12`: Implement: `get_sampling_frame()`, `get_event_table()`, `get_TR()`, `get_run_lengths()`, `get_num_runs()` using first-class sampling_frame.
*   `#13`: Implement: `get_num_voxels()`, `get_num_timepoints()`.
*   `#14`: Implement: `get_censor_vector()`, `get_metadata()`, `get_dataset_type()`.

**Preloading & Chunking:**
*   `#15`: Implement: `preload_data()` helper function with explicit content control.
*   `#16`: Implement: `data_chunks.fmri_dataset()` method with `nchunks`, `runwise`, and `by` arguments.
*   `#17`: Implement: `fmri_data_chunk` S3 class with print method for pipeline-friendly chunks.
*   `#18`: Design & Implement: Internal iterator mechanism for `data_chunks`.

**Validation & S3 Methods:**
*   `#19`: Implement: `validate_fmri_dataset()` consistency checker.
*   `#20`: Implement: `print.fmri_dataset()` user-friendly summary.
*   `#21`: Implement: `summary.fmri_dataset()` for detailed statistics.

**Packaging & Documentation:**
*   `#22`: Setup: Create new R package structure for `fmridataset`.
*   `#23`: Docs: Write Rd files for all exported functions and classes.
*   `#24`: Vignette: "Creating and Using `fmri_dataset` Objects".
*   `#25`: Tests: Comprehensive unit tests using algebraic data type approach.
    *   `#25.1`: Sub-task: Construction + retrieval identity tests (`get_data_matrix(create(...)) = expected`).
    *   `#25.2`: Sub-task: Accessor invariance across `dataset_type`s.
    *   `#25.3`: Sub-task: Masking logic consistency across input types.
    *   `#25.4`: Sub-task: `data_chunks` + `validate` composition tests.
    *   `#25.5`: Sub-task: Tests for BIDS-derived datasets (using `bidser::create_mock_bids`).
    *   `#25.6`: Sub-task: Tests for `fmri_data_chunk` class and `preload_data()`.

**`fmrireg` Compatibility Layer (Tracked in `fmrireg` issues, but relevant here):**
*   `#FRegCompat-1`: `fmrireg`: Add `fmridataset` to Imports.
*   `#FRegCompat-2`: `fmrireg`: Refactor `fmrireg::fmri_dataset` (and variants like `fmri_mem_dataset`, etc.) to wrap `fmridataset::fmri_dataset_create()`.
*   `#FRegCompat-3`: `fmrireg`: Update all internal code to use new `fmridataset` accessors (`get_data_matrix`, etc.).

This detailed proposal and ticket list should provide a clear roadmap for developing the `fmridataset` package. The conditional loading for `bidser` and careful handling of different data sources are key to its flexibility.