Johnny,

This is a fantastic synthesis. You've done a great job of pressure-testing the ideas and consolidating them into a single, cohesive, and minimal API. The feedback is sharp, practical, and perfectly aligned with the project's goals. I'm in full agreement with your proposed "minimal v1 scope."

Let's lock in the decisions and move forward with this refined plan.

---

# The `fmri_series` Query Interface: Final Proposal

**Status**: Ready for Implementation  
**Target**: fmridataset v0.3.0  
**Integration**: Builds on DelayedArray study proposal  

This design provides a single, powerful, and lazy interface for querying voxel data from any `fmri_dataset` object, including the new `fmri_study_dataset`.

---

## **Executive Summary**

The `fmri_series()` function will serve as fmridataset's primary data query interface, offering:

- **Non-random access** to spatial and temporal subsets of fMRI data
- **Rich metadata preservation** with spatial and temporal context
- **DelayedArray integration** for memory-efficient large dataset handling
- **Multi-level support** for both single-subject and study-level datasets
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
fmri_series(dataset, selector, timepoints = NULL, ...)
```

**Parameters:**
- **`dataset`**: An `fmri_dataset` or `fmri_study_dataset` object
- **`selector`**: Spatial selector (voxel indices, coordinates, ROIs, masks)
- **`timepoints`**: Optional temporal subset (indices, logical vector, event-based)
- **`...`**: Reserved for future extensions (backend options, return type controls)

### **The Return Object: `FmriSeries`**

```r
setClass("FmriSeries",
  contains = "DelayedMatrix",  # Inherit DelayedArray functionality
  slots = list(
    voxel_info = "DataFrame",      # S4Vectors::DataFrame with spatial metadata
    temporal_info = "DataFrame",   # Temporal metadata per timepoint
    selection_info = "list",       # Provenance and selection criteria
    dataset_info = "list"          # Source dataset properties
  )
)
```

**Benefits:**
- **DelayedMatrix inheritance**: Direct compatibility with DelayedMatrixStats, BiocParallel
- **Rich metadata**: Automatic spatial/temporal annotation of results
- **Provenance tracking**: Selection criteria preserved for reproducibility
- **Type safety**: S4 validation ensures data integrity

---

## **2. Selector Grammar and Examples**

### **Spatial Selection**

```r
# 1. Index-based selection (within mask)
series_result <- fmri_series(dset, selector = c(1000:2000))

# 2. Coordinate-based selection (grid coordinates)
coords <- matrix(c(50, 60, 25,    # x, y, z coordinates
                   51, 60, 25), ncol = 3, byrow = TRUE)
series_result <- fmri_series(dset, selector = coords)

# 3. ROI-based selection (neuroim2 objects)
roi_vol <- neuroim2::ROIVol(...)
series_result <- fmri_series(dset, selector = roi_vol)

# 4. Mask-based selection (logical volume)
mask <- neuroim2::LogicalNeuroVol(...)
series_result <- fmri_series(dset, selector = mask)

# 5. All voxels (within dataset mask)
series_result <- fmri_series(dset, selector = NULL)
```

### **Temporal Selection**

```r
# 6. Timepoint indices
series_result <- fmri_series(dset, selector = roi_mask, timepoints = 1:100)

# 7. Logical selection
valid_timepoints <- !is.na(dset$event_table$condition)
series_result <- fmri_series(dset, selector = roi_mask, timepoints = valid_timepoints)

# 8. Combined spatiotemporal query
series_result <- fmri_series(dset, 
                            selector = roi_mask,
                            timepoints = c(1:50, 100:150))
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
fmri_series.fmri_dataset <- function(x, selector, timepoints = NULL, ...) {
  vox <- resolve_selector(x, selector)
  tpts <- resolve_timepoints(x, timepoints) 
  dm <- as_delayed_array(x$backend, rows = tpts, cols = vox)
  
  new("FmriSeries",
      dm,
      voxel_info = build_voxel_info(x, vox),
      temporal_info = build_temporal_info(x, tpts),
      selection_info = list(
        selector = substitute(selector),
        timepoints = substitute(timepoints),
        timestamp = Sys.time()
      ),
      dataset_info = list(
        dataset_type = class(x)[1],
        backend_type = class(x$backend)[1],
        total_voxels = sum(backend_get_mask(x$backend)),
        total_timepoints = backend_get_dims(x$backend)$time
      ))
}
```

### **Key Helper Functions**

```r
# Spatial resolution
resolve_selector <- function(dataset, selector) {
  # Handle NULL, integer indices, coordinates, ROI objects, masks
  # Return: integer vector of voxel indices within dataset mask
}

# Temporal resolution  
resolve_timepoints <- function(dataset, timepoints) {
  # Handle NULL, integer indices, logical vectors
  # Return: integer vector of timepoint indices
}

# Metadata builders
build_voxel_info <- function(dataset, voxel_indices) {
  # Create DataFrame with x, y, z coordinates, atlas labels, etc.
}

build_temporal_info <- function(dataset, timepoint_indices) {
  # Create DataFrame with run_id, TR, condition, subject_id (for studies), etc.
}
```

---

## **4. Core Methods and Workflows**

### **Essential Methods (v1)**

```r
# Concise display
show(series_result)
#> <FmriSeries> 1,247 voxels × 300 timepoints (lazy)
#> Selector: ROI mask | Backend: h5_backend
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
# Features: timepoints × voxels matrix
X <- t(as.matrix(series_result))  # Transpose for sklearn-style
y <- series_result@temporal_info$condition
# Ready for classification/regression...
```

---

## **5. Backward Compatibility and Migration**

### **Smooth Transition Strategy**

```r
# Alias with deprecation warning
series.fmri_dataset <- function(x, ...) {
  lifecycle::deprecate_warn(
    "1.0.0", 
    "series()", 
    "fmri_series()",
    details = "fmri_series() provides richer metadata and better performance."
  )
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
- Create `R/FmriSeries.R` with class definition
- Implement `show()` method
- Basic validation methods
- **AC**: Class instantiates correctly, prints expected summary

**S1-T2: Implement Resolver Helpers**  
- `resolve_selector()` for spatial selection
- `resolve_timepoints()` for temporal selection
- Handle all v1 selector types (indices, coords, ROI objects, NULL)
- **AC**: All selector types correctly resolve to integer vectors, comprehensive unit tests

**S1-T3: Implement `fmri_series.fmri_dataset` Method**
- Core S3 method orchestrating resolvers and DelayedArray creation
- Integration with existing `as_delayed_array()` backend methods
- **AC**: Returns valid `FmriSeries`, correct dimensions, populated metadata

**S1-T4: Implement Core Methods**
- `as.matrix.FmriSeries()` for data materialization
- `as_tibble.FmriSeries()` for tidy workflows  
- **AC**: Methods return correctly formatted data structures

### **Sprint 2: Study Integration and Polish (1-2 weeks)**

**Goal**: Multi-subject support, deprecation handling, final documentation

**S2-T1: Study-Level Integration**
- `fmri_series.fmri_study_dataset()` method
- Multi-subject metadata handling in `build_temporal_info()`
- **AC**: Study datasets return correctly structured multi-subject results

**S2-T2: Backward Compatibility**
- `series()` alias with `lifecycle` deprecation warnings
- Ensure one-time warnings per session
- **AC**: Alias works correctly, appropriate warnings displayed

**S2-T3: Integration Testing**
- Cross-backend compatibility tests
- Multi-subject ordering validation
- Edge case handling (empty selections, single timepoints, etc.)
- **AC**: All backends pass integration tests, consistent behavior

**S2-T4: Documentation and Examples**
- Complete Roxygen documentation
- Update study vignette with `fmri_series()` examples
- Performance benchmarks and best practices
- **AC**: Documentation complete, vignette demonstrates real workflows

---

## **7. Performance Considerations**

### **Memory Efficiency**
- **Lazy evaluation**: No data loaded until `as.matrix()` or similar materialization
- **DelayedArray chunking**: Automatic optimization for large datasets
- **Backend-specific optimization**: H5 hyperslab selection, memory-mapped NIfTI access

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

# Event-based temporal selection  
fmri_series(dset, timepoints = event_window("task_onset", pre = 2, post = 10))

# Complex spatiotemporal queries
fmri_series(dset, 
           selector = sphere(center = c(50, 60, 25), radius = 5),
           timepoints = trials(condition == "high_load"))
```

### **Return Type Flexibility**
```r
# Direct DelayedArray return for advanced users
delayed_result <- fmri_series(dset, selector = roi, return_type = "DelayedArray")

# SummarizedExperiment integration
se_result <- fmri_series(dset, selector = roi, return_type = "SummarizedExperiment")
```

### **Advanced Analysis Integration**
- Integration with `limma` for differential expression-style analysis
- Built-in connectivity analysis helpers
- Integration with dimensionality reduction frameworks

---

## **Conclusion**

This proposal provides a comprehensive, production-ready design for the `fmri_series()` interface that:

1. **Solves the core problem**: Non-random access to fMRI data with rich metadata
2. **Integrates seamlessly**: Works with existing backends and DelayedArray infrastructure  
3. **Supports modern workflows**: Tidy analysis, Bioconductor ecosystem, machine learning
4. **Scales appropriately**: From single subjects to large multi-site studies
5. **Maintains compatibility**: Smooth migration path for existing users

The two-sprint implementation plan provides a clear path from concept to production, with well-defined acceptance criteria and incremental value delivery.

**Ready for implementation.**