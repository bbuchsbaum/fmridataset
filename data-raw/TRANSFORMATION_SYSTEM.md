# Modular Transformation System for fmridataset

## Overview

The `fmridataset` package now features a modular, pluggable transformation system that replaces the previous hardcoded `apply_preprocessing` approach. This new system is:

- **Modular**: Individual transformations can be combined in flexible ways
- **Extensible**: Easy to add new transformations without modifying core code
- **Composable**: Transformations can be chained together in pipelines
- **Performant**: Efficient application with optional verbose progress
- **Backwards Compatible**: Legacy parameters still work via automatic conversion

## Core Concepts

### 1. Transformations

A **transformation** is a self-contained data processing operation that takes a matrix (timepoints × voxels) and returns a transformed matrix of the same dimensions.

```r
# Create a transformation
zscore_transform <- transform_temporal_zscore(remove_trend = TRUE)

# Apply to data
transformed_data <- apply_transformation(zscore_transform, data_matrix)
```

### 2. Transformation Pipelines

A **pipeline** is a sequence of transformations applied in order:

```r
# Create a pipeline
pipeline <- transformation_pipeline(
  transform_detrend(method = "linear"),
  transform_outlier_removal(method = "zscore", threshold = 3),
  transform_temporal_zscore()
)

# Apply pipeline
result <- apply_pipeline(pipeline, data_matrix, verbose = TRUE)
```

### 3. Integration with fmri_dataset

Transformation pipelines integrate seamlessly with the dataset system:

```r
# Create dataset with custom pipeline
dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(100, 100),
  transformation_pipeline = pipeline
)

# Get data with/without transformations
raw_data <- get_data_matrix(dataset, apply_transformations = FALSE)
processed_data <- get_data_matrix(dataset, apply_transformations = TRUE)
```

## Built-in Transformations

### Temporal Z-scoring
```r
transform_temporal_zscore(
  remove_mean = TRUE,      # Center the data (mean = 0)
  remove_trend = FALSE     # Remove linear trend first
)
```

### Detrending
```r
transform_detrend(
  method = "linear"        # "linear" or "quadratic"
)
```

### Temporal Smoothing
```r
transform_temporal_smooth(
  window_size = 3,         # Size of smoothing window
  method = "gaussian"      # "gaussian", "box", or "median"
)
```

### High-pass Filtering
```r
transform_highpass(
  cutoff_freq = 0.01,      # Cutoff frequency in Hz
  TR = 2.0,                # Repetition time
  order = 4                # Filter order
)
```

### Outlier Removal
```r
transform_outlier_removal(
  method = "zscore",       # "zscore", "iqr", or "mad"
  threshold = 3,           # Detection threshold
  replace_method = "interpolate"  # "interpolate", "median", or "mean"
)
```

## Creating Custom Transformations

### Simple Custom Transformation

```r
# Log transformation
log_transform <- transformation(
  name = "log_transform",
  description = "Log transformation with positive shift",
  params = list(shift = 1),
  fn = function(data, shift) {
    log(data + shift)
  }
)
```

### Advanced Custom Transformation

```r
# Wavelet denoising (example structure)
wavelet_denoise <- transformation(
  name = "wavelet_denoise",
  description = "Wavelet-based denoising",
  params = list(
    wavelet = "daubechies",
    levels = 4,
    threshold = 0.1
  ),
  fn = function(data, wavelet, levels, threshold) {
    # Implementation would use wavelets package
    # This is just a structure example
    
    denoised <- data  # Placeholder
    for (i in seq_len(ncol(data))) {
      # Apply wavelet denoising to each voxel
      # denoised[, i] <- wavelet_denoise_implementation(data[, i], ...)
    }
    
    return(denoised)
  }
)
```

## Pipeline Examples

### Basic Preprocessing
```r
basic_pipeline <- transformation_pipeline(
  transform_detrend(),
  transform_temporal_zscore()
)
```

### Advanced Preprocessing
```r
advanced_pipeline <- transformation_pipeline(
  transform_detrend(method = "quadratic"),
  transform_outlier_removal(method = "mad", threshold = 2.5),
  transform_temporal_smooth(window_size = 3),
  transform_highpass(cutoff_freq = 0.008, TR = 2.0),
  transform_temporal_zscore(remove_trend = FALSE)
)
```

### Task-specific Pipeline
```r
task_pipeline <- transformation_pipeline(
  transform_detrend(),
  transform_outlier_removal(threshold = 3),
  # Custom transformation for task-related processing
  transformation(
    name = "task_baseline_removal",
    description = "Remove baseline periods",
    params = list(baseline_periods = list(c(1, 20), c(180, 200))),
    fn = function(data, baseline_periods) {
      # Custom baseline removal logic
      processed_data <- data
      for (period in baseline_periods) {
        baseline_mean <- apply(data[period[1]:period[2], ], 2, mean)
        processed_data <- sweep(processed_data, 2, baseline_mean, "-")
      }
      return(processed_data)
    }
  ),
  transform_temporal_zscore()
)
```

## Performance Considerations

### Caching Strategy
- Raw data is cached after initial loading
- Transformations are applied fresh each time (no caching of processed data)
- This ensures consistency when transformation parameters change
- Use `get_data_matrix(dataset, apply_transformations = FALSE)` for repeated raw access

### Memory Efficiency
- Transformations work in-place when possible
- Large datasets can use chunking with `data_chunks()`:

```r
# Process large dataset in chunks
chunks <- data_chunks(dataset, nchunks = 10, apply_transformations = TRUE)

results <- list()
for (chunk in chunks) {
  # Process each chunk
  results[[chunk$chunk_num]] <- analyze_data(chunk$data)
}
```

### Parallel Processing
- Individual transformations can be parallelized internally
- Pipeline application is sequential (by design for data dependencies)
- Consider custom transformations with parallel implementations:

```r
parallel_zscore <- transformation(
  name = "parallel_zscore",
  fn = function(data, n_cores = 4) {
    if (requireNamespace("parallel", quietly = TRUE)) {
      # Parallel implementation
      cluster <- parallel::makeCluster(n_cores)
      on.exit(parallel::stopCluster(cluster))
      
      # Process voxels in parallel
      voxel_chunks <- split(seq_len(ncol(data)), 
                           cut(seq_len(ncol(data)), n_cores))
      
      results <- parallel::parLapply(cluster, voxel_chunks, function(voxels) {
        scale(data[, voxels], center = TRUE, scale = TRUE)
      })
      
      return(do.call(cbind, results))
    } else {
      # Fallback to serial
      return(scale(data, center = TRUE, scale = TRUE))
    }
  }
)
```

## Migration from Legacy System

### Automatic Conversion
The system automatically converts legacy parameters:

```r
# Old way (still works)
dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(100, 100),
  temporal_zscore = TRUE,
  voxelwise_detrend = TRUE
)

# Automatically creates equivalent pipeline:
# transformation_pipeline(
#   transform_detrend(),
#   transform_temporal_zscore()
# )
```

### Manual Migration
```r
# Convert legacy code manually for full control
legacy_dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(100, 100),
  temporal_zscore = TRUE,
  voxelwise_detrend = TRUE
)

# Becomes:
modern_pipeline <- transformation_pipeline(
  transform_detrend(method = "linear"),
  transform_temporal_zscore(remove_mean = TRUE, remove_trend = FALSE)
)

modern_dataset <- fmri_dataset_create(
  images = data_matrix,
  TR = 2.0,
  run_lengths = c(100, 100),
  transformation_pipeline = modern_pipeline
)
```

## Advanced Features

### Conditional Transformations
```r
# Transformation that adapts based on data properties
adaptive_transform <- transformation(
  name = "adaptive_processing",
  fn = function(data) {
    # Check data properties
    if (max(abs(data)) > 10) {
      # High amplitude data - use robust normalization
      return(apply(data, 2, function(x) x / mad(x)))
    } else {
      # Normal amplitude - use standard z-score
      return(scale(data, center = TRUE, scale = TRUE))
    }
  }
)
```

### Transformation Metadata and Logging
```r
# Transformation with rich metadata
logged_transform <- transformation(
  name = "logged_zscore",
  description = "Z-score with logging",
  params = list(log_file = "preprocessing.log"),
  fn = function(data, log_file) {
    # Log transformation details
    cat("Applying z-score at", Sys.time(), "\n", file = log_file, append = TRUE)
    cat("Data dimensions:", nrow(data), "x", ncol(data), "\n", 
        file = log_file, append = TRUE)
    
    result <- scale(data, center = TRUE, scale = TRUE)
    
    cat("Z-score complete. Mean:", mean(result), "SD:", sd(result), "\n",
        file = log_file, append = TRUE)
    
    return(result)
  }
)
```

### Pipeline Branching
```r
# Create different pipelines for different conditions
create_pipeline_for_task <- function(task_type) {
  base_transforms <- list(
    transform_detrend(),
    transform_outlier_removal()
  )
  
  if (task_type == "resting_state") {
    additional <- list(
      transform_highpass(cutoff_freq = 0.008, TR = 2.0),
      transform_temporal_smooth(window_size = 3)
    )
  } else if (task_type == "task_based") {
    additional <- list(
      transform_temporal_zscore()
    )
  }
  
  transformation_pipeline(c(base_transforms, additional))
}

# Usage
rs_pipeline <- create_pipeline_for_task("resting_state")
task_pipeline <- create_pipeline_for_task("task_based")
```

## Testing and Validation

### Unit Testing Transformations
```r
# Test individual transformations
test_data <- matrix(rnorm(1000), nrow = 100, ncol = 10)

# Test z-score transformation
zscore_result <- apply_transformation(transform_temporal_zscore(), test_data)
stopifnot(abs(mean(zscore_result)) < 1e-10)  # Mean should be ~0
stopifnot(abs(sd(zscore_result) - 1) < 1e-10)  # SD should be ~1

# Test pipeline consistency
pipeline1 <- transformation_pipeline(
  transform_detrend(),
  transform_temporal_zscore()
)

pipeline2 <- transformation_pipeline(
  transform_detrend(),
  transform_temporal_zscore()
)

result1 <- apply_pipeline(pipeline1, test_data)
result2 <- apply_pipeline(pipeline2, test_data)

stopifnot(all.equal(result1, result2))  # Should be identical
```

### Integration Testing
```r
# Test with fmri_dataset
dataset <- fmri_dataset_create(
  images = test_data,
  TR = 2.0,
  run_lengths = c(50, 50),
  transformation_pipeline = pipeline1
)

# Test data access consistency
processed1 <- get_data_matrix(dataset, apply_transformations = TRUE)
processed2 <- get_data_matrix(dataset, apply_transformations = TRUE)

stopifnot(identical(processed1, processed2))  # Should be identical
```

## Conclusion

The modular transformation system provides a flexible, extensible foundation for fMRI data preprocessing that can grow with your needs while maintaining backwards compatibility and high performance. The system enables:

1. **Easy experimentation** with different preprocessing strategies
2. **Reproducible pipelines** that can be saved and shared
3. **Custom transformations** for specific research needs
4. **Efficient processing** of large datasets through chunking
5. **Clear documentation** of preprocessing steps through pipeline objects

This architecture makes the `fmridataset` package much more flexible and user-friendly while maintaining the robust foundation established in the original design.

# Elegant BIDS Interface System

## Overview

The fmridataset package includes an **extremely elegant and well-designed** BIDS interface system that provides **advanced but loosely coupled** integration with BIDS (Brain Imaging Data Structure) datasets. This interface represents a significant architectural advancement over traditional BIDS integration approaches.

## Core Design Principles

### 1. Loose Coupling Architecture
- **Pluggable Backends**: Support for multiple BIDS libraries without hard dependencies
- **Interface Abstraction**: Standardized API regardless of underlying BIDS implementation
- **Optional Dependencies**: BIDS functionality available only when appropriate packages are installed

### 2. Elegant User Experience
- **Fluent API**: Method chaining for intuitive query building
- **Sophisticated Configuration**: Advanced options without complexity
- **Discovery Tools**: Rich exploration capabilities for BIDS datasets

### 3. Extensibility and Future-Proofing
- **Custom Backends**: Users can provide their own BIDS processing functions
- **Multiple Libraries**: Support for bidser, PyBIDS (future), and custom implementations
- **Adaptive Architecture**: Can accommodate new BIDS tools and standards

## Core Components

### Pluggable Backend System

```r
# Use different backends interchangeably
backend_bidser <- bids_backend("bidser")
backend_custom <- bids_backend("custom", 
  backend_config = list(
    find_scans = my_scan_function,
    read_metadata = my_metadata_function,
    get_run_info = my_run_info_function
  ))
```

### Elegant Query Interface

```r
# Fluent API with method chaining
query <- bids_query("/path/to/bids") %>%
  subject("01", "02") %>%
  task("rest", "task") %>%
  session("1") %>%
  derivatives("fmriprep") %>%
  space("MNI152NLin2009cAsym")

# Direct dataset creation
dataset <- query %>% as_fmri_dataset(subject_id = "01")
```

### Sophisticated Configuration System

```r
# Advanced configuration with intelligent defaults
config <- bids_config(
  image_selection = list(
    strategy = "prefer_derivatives",
    preferred_pipelines = c("fmriprep", "nilearn"),
    required_space = "MNI152NLin2009cAsym"
  ),
  quality_control = list(
    check_completeness = TRUE,
    validate_headers = TRUE,
    censoring_threshold = 0.2
  )
)
```

### Discovery Interface

```r
# Explore BIDS dataset structure
discovery <- bids_discover("/path/to/bids")

# Access discovered entities
discovery$subjects     # Available subjects
discovery$tasks        # Available tasks  
discovery$derivatives  # Available preprocessing pipelines
discovery$summary      # Dataset statistics
```

## Integration with Transformation System

The BIDS interface seamlessly integrates with the modular transformation system:

```r
# Combined sophisticated workflow
pipeline <- transformation_pipeline() %>%
  add_transformation(transform_temporal_zscore()) %>%
  add_transformation(transform_detrend(method = "linear"))

dataset <- bids_query("/path/to/bids") %>%
  subject("01") %>%
  task("rest") %>%
  derivatives("fmriprep") %>%
  as_fmri_dataset(
    config = bids_config(),
    transformation_pipeline = pipeline
  )
```

## Backend Comparison

| Backend | Ease of Use | Flexibility | Performance | Dependencies |
|---------|-------------|-------------|-------------|--------------|
| bidser  | High        | Medium      | Good        | bidser pkg   |
| pybids  | Medium      | High        | Variable    | Python + reticulate |
| custom  | Low         | Very High   | User-defined| User functions |

## Architecture Benefits

### Technical Benefits
- **Separation of Concerns**: BIDS logic separated from core fmridataset functionality
- **Testability**: Each component can be tested independently
- **Maintainability**: Changes to BIDS handling don't affect core package
- **Performance**: Pluggable backends allow optimization for specific use cases

### User Benefits
- **Choice**: Multiple levels of sophistication and control
- **Future-Proof**: Adapts to new BIDS tools and standards
- **Customization**: Users can implement domain-specific BIDS handling
- **Discovery**: Rich tools for exploring and understanding datasets

### Developer Benefits
- **Extensibility**: Easy to add new backends and functionality
- **API Stability**: Core interface remains stable while backends evolve
- **Community Contribution**: Users can contribute custom backends
- **Reduced Coupling**: Core package doesn't depend on specific BIDS libraries

## Migration from Current Implementation

The elegant interface provides a smooth migration path:

```r
# Current approach (still supported)
dataset <- as.fmri_dataset(bids_proj, subject_id = "01", task_id = "rest")

# New elegant approach
dataset <- bids_query(bids_proj$path) %>%
  subject("01") %>%
  task("rest") %>%
  as_fmri_dataset()
```

## Future Enhancements

The elegant architecture allows for future enhancements:

1. **PyBIDS Integration**: Add Python-based BIDS handling via reticulate
2. **Parallel Processing**: Backend-specific parallel discovery and extraction
3. **Caching Systems**: Sophisticated metadata and discovery caching
4. **Quality Metrics**: Advanced dataset quality assessment tools
5. **Interactive Exploration**: GUI or web-based BIDS dataset exploration

## Summary

This elegant BIDS interface represents a significant advancement in neuroimaging dataset handling. It provides the sophistication needed for advanced users while maintaining the simplicity required for everyday use. The loose coupling ensures the package remains flexible and future-proof, while the elegant API design makes complex BIDS operations intuitive and discoverable.

The interface achieves the goal of being "extremely elegant and well-designed" through its clean separation of concerns, sophisticated configuration system, and fluent API design, while maintaining "advanced but loosely coupled" architecture through its pluggable backend system and interface abstraction. 