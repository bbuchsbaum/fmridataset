# The `fmri_series` Query Interface: Final Proposal

**Status**: Ready for Implementation  
**Target**: fmridataset v0.3.0  
**Integration**: Builds on DelayedArray study proposal  

This design provides a single, powerful, and lazy interface for querying voxel data from any `fmri_dataset` object, including the new `fmri_study_dataset`.

---

## **Executive Summary**

The `fmri_series()` function will serve as fmridataset's primary data query interface, offering:

- **Random (direct) access** to spatial and temporal subsets of fMRI data
- **Rich metadata preservation** with spatial and temporal context
- **DelayedArray integration** for memory-efficient large dataset handling
- **Single-subject and study-level support** for flexible analysis workflows
- **Tidy workflow integration** with modern R data science practices

### **Key Design Principles**

1. **Lazy by Default**: All operations return DelayedArray until explicit materialization
2. **Metadata Rich**: Every result includes spatial, temporal, and provenance information
3. **Backend Agnostic**: Works optimally with any storage backend (H5, NIfTI, matrix)
4. **Bioconductor Ready**: S4 containers for seamless ecosystem integration
5. **User Friendly**: S3 interface familiar to R users

---

## **1. API Design**

### **Core Function Signature**

```r
fmri_series(dataset, selector = NULL, timepoints = NULL, 
           output = c("FmriSeries", "DelayedMatrix"), 
           event_window = NULL, ...)
```

**Parameters:**
- **`dataset`**: An `fmri_dataset` or `fmri_study_dataset` object
- **`selector`**: Spatial selector (voxel indices, coordinates, ROIs, masks). `NULL` returns all voxels within mask
- **`timepoints`**: Optional temporal subset (indices, logical vector). `NULL` returns all timepoints
- **`output`**: Return type - `"FmriSeries"` (default) for rich metadata, `"DelayedMatrix"` for direct array access
- **`event_window`**: Reserved for future event-based temporal selection (unused in v1)
- **`...`**: Reserved for backend-specific options

**Key Design Decisions:**
- **Explicit arguments**: Avoids "ellipsis creep" by exposing commonly-used options as named parameters
- **NULL as "all"**: `selector = NULL` and `timepoints = NULL` clearly indicate full data access
- **Future-proof**: `event_window` parameter reserved to avoid breaking changes in v2

### **The Return Object: `FmriSeries`**

```r
# S4 class definition - MUST be defined before any methods use it
setClass("FmriSeries",
  contains = "DelayedMatrix",  # Inherit DelayedArray functionality
  slots = list(
    voxel_info = "DataFrame",      # S4Vectors::DataFrame with spatial metadata (lazy)
    temporal_info = "DataFrame",   # Temporal metadata per timepoint (lazy)
    selection_info = "list",       # Provenance and selection criteria
    dataset_info = "list"          # Source dataset properties
  )
)
```

**Matrix Orientation (IMMUTABLE DECISION):**
- **Rows = timepoints, Columns = voxels** (follows neuroim conventions)
- Consistent with `time × voxels` expectation in neuroimaging
- All downstream methods and documentation will enforce this orientation

**Benefits:**
- **DelayedMatrix inheritance**: Direct compatibility with DelayedMatrixStats, BiocParallel
- **Rich metadata**: Automatic spatial/temporal annotation of results
- **Provenance tracking**: Selection criteria preserved for reproducibility
- **Type safety**: S4 validation ensures data integrity
- **Lazy metadata**: `voxel_info` and `temporal_info` materialized only when accessed

---

## **2. Selector Grammar and Examples**

### **Spatial Selection with Optimization**

```r
# 1. All voxels (optimized fast path)
series_result <- fmri_series(dset)  # selector = NULL by default
series_result <- fmri_series(dset, selector = NULL)  # explicit

# 2. Index-based selection (within mask)
series_result <- fmri_series(dset, selector = c(1000:2000))

# 3. Coordinate-based selection (grid coordinates)
coords <- matrix(c(50, 60, 25,    # x, y, z coordinates
                   51, 60, 25), ncol = 3, byrow = TRUE)
series_result <- fmri_series(dset, selector = coords)

# 4. ROI-based selection (neuroim2 objects)
roi_vol <- neuroim2::ROIVol(...)
series_result <- fmri_series(dset, selector = roi_vol)

# 5. Mask-based selection (logical volume)
mask <- neuroim2::LogicalNeuroVol(...)
series_result <- fmri_series(dset, selector = mask)
```

### **Temporal Selection with Helpers**

```r
# 1. All timepoints (explicit helper)
series_result <- fmri_series(dset, timepoints = all_timepoints(dset))
series_result <- fmri_series(dset, timepoints = NULL)  # equivalent

# 2. Timepoint indices
series_result <- fmri_series(dset, selector = roi_mask, timepoints = 1:100)

# 3. Logical selection
valid_timepoints <- !is.na(dset$event_table$condition)
series_result <- fmri_series(dset, selector = roi_mask, timepoints = valid_timepoints)

# 4. Combined spatiotemporal query
series_result <- fmri_series(dset, 
                            selector = roi_mask,
                            timepoints = c(1:50, 100:150))
```

### **Output Type Control**

```r
# Rich metadata object (default)
rich_result <- fmri_series(dset, selector = roi_mask)
class(rich_result)  # "FmriSeries"

# Direct DelayedMatrix for advanced users
matrix_result <- fmri_series(dset, selector = roi_mask, output = "DelayedMatrix")
class(matrix_result)  # "DelayedMatrix"
```

### **Study-Level Queries**

```r
# Multi-subject data access
study_result <- fmri_series(study_dataset, 
                           selector = roi_mask,
                           timepoints = 1:200)

# Result automatically includes subject_id in temporal_info
study_result@temporal_info
#> # A DataFrame: 4000 x 6
#>   subject_id run_id timepoint    TR condition  stimulus_onset
#>        <chr>  <int>     <int> <dbl>     <chr>          <dbl>
#> 1       sub1      1         1   2.0      rest             NA
#> 2       sub1      1         2   2.0      rest             NA
#> 3       sub1      1         3   2.0      task            6.0
#> ...
```

---

## **3. Implementation Architecture**

### **S3/S4 Hybrid Model**

**Why This Approach:**
- **S3 for user interface**: Familiar, lightweight, tidyverse-compatible
- **S4 for data containers**: Type safety, Bioconductor integration, metadata validation
- **No user friction**: Transparent to end users
- **Minimal code churn**: Existing backend system stays S3

```r
# S3 generic - easy to call, easy to extend
fmri_series <- function(x, ...) {
  UseMethod("fmri_series")
}

# S4 container - rigorous, Bioconductor-friendly
# CRITICAL: Define setClass() before any methods use it
setClass("FmriSeries",
  contains = "DelayedMatrix",
  slots = c(
    voxel_info    = "DataFrame",
    temporal_info = "DataFrame", 
    selection_info = "list",
    dataset_info  = "list"
  )
)

# S3 → S4 bridge in the method
fmri_series.fmri_dataset <- function(x, selector = NULL, timepoints = NULL, 
                                    output = c("FmriSeries", "DelayedMatrix"),
                                    event_window = NULL, ...) {
  output <- match.arg(output)
  
  # Fast path for NULL selector (all voxels)
  if (is.null(selector)) {
    vox <- seq_len(sum(backend_get_mask(x$backend)))
  } else {
    vox <- resolve_selector(x, selector)
  }
  
  # Handle timepoints
  if (is.null(timepoints)) {
    tpts <- seq_len(backend_get_dims(x$backend)$time)
  } else {
    tpts <- resolve_timepoints(x, timepoints)
  }
  
  # Get DelayedMatrix: rows = timepoints, cols = voxels
  dm <- as_delayed_array(x$backend, rows = tpts, cols = vox)
  
  # Return based on output type
  if (output == "DelayedMatrix") {
    return(dm)
  }
  
  # Build FmriSeries with lazy metadata
  new("FmriSeries",
      dm,
      voxel_info = build_voxel_info_lazy(x, vox),
      temporal_info = build_temporal_info_lazy(x, tpts),
      selection_info = list(
        selector = substitute(selector),
        timepoints = substitute(timepoints),
        timestamp = Sys.time(),
        n_voxels_selected = length(vox),
        n_timepoints_selected = length(tpts)
      ),
      dataset_info = list(
        dataset_type = class(x)[1],
        backend_type = class(x$backend)[1],
        total_voxels = sum(backend_get_mask(x$backend)),
        total_timepoints = backend_get_dims(x$backend)$time,
        orientation = "timepoints × voxels"
      ))
}
```

### **Key Helper Functions**

```r
# Spatial resolution with fast NULL path
resolve_selector <- function(dataset, selector) {
  # selector should already be checked for NULL by caller
  # Handle integer indices, coordinates, ROI objects, masks
  # Return: integer vector of voxel indices within dataset mask
}

# Temporal resolution helper
resolve_timepoints <- function(dataset, timepoints) {
  # Handle integer indices, logical vectors
  # Return: integer vector of timepoint indices
}

# Temporal helper for user convenience
all_timepoints <- function(dataset) {
  seq_len(backend_get_dims(dataset$backend)$time)
}

# Lazy metadata builders (materialize only when accessed)
build_voxel_info_lazy <- function(dataset, voxel_indices) {
  # Return lightweight object that builds DataFrame on first access
  # Contains x, y, z coordinates, atlas labels, etc.
}

build_temporal_info_lazy <- function(dataset, timepoint_indices) {
  # Return lightweight object that builds DataFrame on first access
  # Contains run_id, TR, condition, subject_id (for studies), etc.
}
```

---

## **4. Core Methods and Workflows**

### **Essential Methods (v1)**

```r
# Concise display
show(series_result)
#> <FmriSeries> 1,247 voxels × 300 timepoints (lazy)
#> Selector: ROI mask | Backend: h5_backend | Orientation: time × voxels
#> Memory: ~9.5 MB (unrealized)

# Data realization
data_matrix <- as.matrix(series_result)  # Force materialization

# Tidy analysis workflow
analysis_data <- as_tibble(series_result)
#> # A tibble: 374,100 × 10
#>   subject_id run_id timepoint     x     y     z signal condition stimulus_onset
#>        <chr>  <int>     <int> <dbl> <dbl> <dbl>  <dbl>     <chr>          <dbl>
#> 1       sub1      1         1    45    67    32   0.12      rest             NA
#> 2       sub1      1         1    46    67    32   0.15      rest             NA
#> 3       sub1      1         1    47    67    32   0.09      rest             NA
#> ...
```

### **Workflow Integration Examples**

```r
# === Tidy Analysis Workflow ===
result %>%
  as_tibble() %>%
  filter(condition == "task") %>%
  group_by(subject_id, x, y, z) %>%
  summarise(mean_activation = mean(signal)) %>%
  # Continue with standard dplyr operations...

# === DelayedArray/BiocParallel Workflow ===
# Memory-efficient operations on large datasets
voxel_means <- DelayedMatrixStats::colMeans2(series_result)
voxel_vars <- DelayedMatrixStats::colVars(series_result)

# Parallel processing with BiocParallel
BPPARAM <- MulticoreParam(workers = 4)
results <- bplapply(split_data, analysis_function, BPPARAM = BPPARAM)

# === Machine Learning Workflow ===
# Features: timepoints × voxels matrix (already correct orientation)
X <- as.matrix(series_result)  # No transpose needed
y <- series_result@temporal_info$condition
# Ready for classification/regression...
```

---

## **5. Backward Compatibility and Migration**

### **Smooth Transition Strategy with Session-Level Warnings**

```r
# Session state for deprecation warnings
.fmri_series_warned <- FALSE

# Alias with one-time-per-session deprecation warning
series.fmri_dataset <- function(x, ...) {
  if (!.fmri_series_warned) {
    lifecycle::deprecate_warn(
      "1.0.0", 
      "series()", 
      "fmri_series()",
      details = "fmri_series() provides richer metadata and better performance."
    )
    .fmri_series_warned <<- TRUE
  }
  fmri_series(x, ...)
}
```

**Migration Path:**
1. **v0.3.0**: Introduce `fmri_series()`, `series()` alias with warnings
2. **v0.4.0**: Deprecate `series()` more strongly  
3. **v1.0.0**: Remove `series()` alias

---

## **6. Implementation Plan: Two Sprints**

### **Sprint 1: Core Infrastructure (2-3 weeks)**

**Goal**: Implement `FmriSeries` class and single-subject `fmri_series()` method

**S1-T1: Define `FmriSeries` S4 Class**
- Create `R/FmriSeries.R` with class definition (FIRST)
- Implement `show()` method with orientation information
- Basic validation methods
- **AC**: Class instantiates correctly, prints expected summary

**S1-T2: Implement Resolver Helpers**  
- `resolve_selector()` with fast NULL path optimization
- `resolve_timepoints()` for temporal selection
- `all_timepoints()` helper function
- Handle all v1 selector types (indices, coords, ROI objects, NULL)
- **AC**: All selector types correctly resolve to integer vectors, NULL fast-path tested, comprehensive unit tests

**S1-T3: Implement `fmri_series.fmri_dataset` Method**
- Core S3 method orchestrating resolvers and DelayedArray creation
- Integration with existing `as_delayed_array()` backend methods
- Explicit output type handling
- **AC**: Returns valid `FmriSeries`, correct dimensions, populated metadata, DelayedMatrix output option works

**S1-T4: Implement Core Methods**
- `as.matrix.FmriSeries()` for data materialization
- `as_tibble.FmriSeries()` for tidy workflows with lazy metadata realization
- **AC**: Methods return correctly formatted data structures, integration test: `fmri_series() → as_tibble() → dplyr::summarise()`

### **Sprint 2: Study Integration and Polish (1-2 weeks)**

**Goal**: Multi-subject support, deprecation handling, final documentation

**S2-T1: Study-Level Integration**
- `fmri_series.fmri_study_dataset()` method
- Multi-subject metadata handling in `build_temporal_info_lazy()`
- **AC**: Study datasets return correctly structured multi-subject results

**S2-T2: Backward Compatibility**
- `series()` alias with session-level `lifecycle` deprecation warnings
- Ensure one-time warnings per session
- **AC**: Alias works correctly, appropriate warnings displayed only once per session

**S2-T3: Integration Testing**
- Cross-backend compatibility tests
- Multi-subject ordering validation
- Edge case handling (empty selections, single timepoints, etc.)
- Tidyverse integration test chain
- **AC**: All backends pass integration tests, consistent behavior, chained workflow tests pass

**S2-T4: Documentation and Examples**
- Complete Roxygen documentation
- Update study vignette with `fmri_series()` examples
- Performance benchmarks and best practices
- **AC**: Documentation complete, vignette demonstrates real workflows

---

## **7. Performance Considerations and Global Options**

### **Memory Efficiency**
- **Lazy evaluation**: No data loaded until `as.matrix()` or similar materialization
- **DelayedArray chunking**: Automatic optimization for large datasets
- **Backend-specific optimization**: H5 hyperslab selection, memory-mapped NIfTI access
- **Lazy metadata**: voxel_info and temporal_info materialized only when accessed

### **Global Performance Knobs**
```r
# Early exposure of key performance options
options(fmridataset.block_mb = 64)        # DelayedArray block size
options(fmridataset.lazy_metadata = TRUE) # Metadata materialization strategy
options(fmridataset.parallel_backend = "BiocParallel")  # Default parallel backend
```

### **Computational Efficiency** 
- **Minimal overhead**: S3/S4 dispatch cost negligible vs. I/O operations
- **BiocParallel integration**: Seamless parallel processing support
- **Backend optimization**: Each backend can optimize for common access patterns

### **Scalability Targets**
- **Single subject**: Handle 100k+ voxels × 1000+ timepoints efficiently
- **Study level**: Support 100+ subjects with combined DelayedArray operations
- **Storage agnostic**: Performance independent of backend choice

---

## **8. Future Extensions (Post-v1)**

### **Enhanced Selector Grammar**
```r
# Atlas-based selection
fmri_series(dset, selector = atlas("AAL", regions = c("Hippocampus_L", "Hippocampus_R")))

# Event-based temporal selection (using reserved parameter)
fmri_series(dset, event_window = event_window("task_onset", pre = 2, post = 10))

# Complex spatiotemporal queries
fmri_series(dset, 
           selector = sphere(center = c(50, 60, 25), radius = 5),
           event_window = trials(condition == "high_load"))
```

### **Return Type Flexibility**
```r
# SummarizedExperiment integration
se_result <- fmri_series(dset, selector = roi, output = "SummarizedExperiment")

# RaggedExperiment for variable-length runs
ragged_result <- fmri_series(study_dset, output = "RaggedExperiment")
```

### **Advanced Analysis Integration**
- Integration with `limma` for differential expression-style analysis
- Built-in connectivity analysis helpers
- Integration with dimensionality reduction frameworks

---

## **Conclusion**

This proposal provides a comprehensive, production-ready design for the `fmri_series()` interface that:

1. **Solves the core problem**: Random access to fMRI data with rich metadata
2. **Integrates seamlessly**: Works with existing backends and DelayedArray infrastructure  
3. **Supports modern workflows**: Tidy analysis, Bioconductor ecosystem, machine learning
4. **Scales appropriately**: From single subjects to large multi-site studies
5. **Maintains compatibility**: Smooth migration path for existing users
6. **Future-proofs**: Explicit parameter design prevents breaking changes

**Critical Immutable Decisions:**
- **Matrix orientation**: timepoints × voxels (rows × columns)
- **Parameter design**: Explicit `output` and `event_window` parameters (no ellipsis creep)

The two-sprint implementation plan provides a clear path from concept to production, with well-defined acceptance criteria and incremental value delivery.

**Ready for implementation.**
