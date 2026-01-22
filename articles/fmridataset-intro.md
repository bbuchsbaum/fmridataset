# Getting Started with fmridataset

## Motivation: Unified fMRI Data Access

fMRI analyses frequently involve heterogeneous data sources: NIfTI files
from different scanners, BIDS-organized datasets, preprocessed matrices
from various pipelines, and archived HDF5 files. Each format
traditionally requires distinct loading procedures, memory management
approaches, and temporal organization methods.

The fmridataset package provides a unified interface layer that
abstracts these format-specific differences. This abstraction enables:

- **Consistent data access**: Identical functions work across all
  supported formats
- **Format-specific optimization**: Each backend implements optimal
  loading strategies
- **Temporal structure preservation**: Unified handling of runs, TR, and
  event timing
- **Memory efficiency**: Lazy loading and chunking strategies adapt to
  data characteristics

## Quick Start: Multi-Format Data Access

Let’s jump into a concrete example that shows how fmridataset simplifies
a typical workflow. We’ll create a dataset from a simple matrix, add
temporal structure and experimental events, then demonstrate the unified
interface for data access:

``` r
library(fmridataset)

# Create realistic synthetic fMRI data using helper function
activation_periods <- c(20:30, 50:60, 120:130, 150:160)
fmri_matrix <- generate_example_fmri_data(
  n_timepoints = 200,
  n_voxels = 1000,
  n_active = 100,
  activation_periods = activation_periods,
  signal_strength = 0.5
)

# Create dataset with temporal structure
dataset <- matrix_dataset(
  datamat = fmri_matrix,
  TR = 2.0, # 2-second repetition time
  run_length = c(100, 100) # Two runs of 100 timepoints each
)

# Add experimental design using helper function
events <- generate_example_events(
  n_runs = 2,
  events_per_run = 2,
  TR = 2.0,
  run_length = 100
)

dataset$event_table <- events

# Display the dataset
print_dataset_info(dataset, "Dataset Summary")
#> 
#> Dataset Summary
#> ---------------
#> Dataset class: matrix_dataset, fmri_dataset, list 
#> TR: 2 seconds
#> Number of runs: 2 
#> Run lengths: 100 100 
#> Total timepoints: 200 
#> Events: 4 events across 2 runs
```

Now let’s see the unified interface in action:

``` r
# Access temporal information (same methods regardless of data source)
cat("TR:", get_TR(dataset), "seconds\n")
#> TR: 2 seconds
cat("Number of runs:", n_runs(dataset), "\n")
#> Number of runs: 2
cat("Total timepoints:", n_timepoints(dataset), "\n")
#> Total timepoints: 200
cat("Run durations:", get_run_duration(dataset), "seconds\n")
#> Run durations: 200 200 seconds

# Get data for analysis (returns standard R matrix)
all_data <- get_data_matrix(dataset)
cat("Full data dimensions:", dim(all_data), "\n")
#> Full data dimensions: 200 1000

# Get data from specific runs
run1_data <- get_data_matrix(dataset, run_id = 1)
cat("Run 1 dimensions:", dim(run1_data), "\n")
#> Run 1 dimensions: 200 1000

# Access experimental events
cat("\nEvent Table:\n")
#> 
#> Event Table:
print(head(dataset$event_table, 4))
#>       onset duration trial_type run
#> 1  66.66667       10     task_A   1
#> 2 133.33333       10     task_B   1
#> 3 266.66667       10     task_A   2
#> 4 333.33333       10     task_B   2
```

**Technical Note**: This unified interface operates through backend
abstraction. Each data format implements the same contract, enabling
format-independent analysis code.

## Core Concepts

Now that we’ve seen the unified interface in action, let’s understand
the key abstractions that make fmridataset powerful and flexible.

### The Dataset: Your Analysis Starting Point

A dataset in fmridataset represents everything you need for fMRI
analysis in one unified object. Think of it as a smart container that
knows about your data’s spatial organization (which voxels contain brain
tissue), temporal structure (how timepoints are organized into runs),
and experimental design (when events occurred). Unlike working with raw
files or matrices, a dataset maintains all this metadata and provides
consistent access methods regardless of how the underlying data is
stored.

This abstraction enables analysis functions to operate on any dataset
type through method dispatch. The dataset implementation handles
format-specific I/O operations while maintaining consistent public
methods across all backends.

### Storage Backends: Format Independence

Behind every dataset is a storage backend that handles the actual data
input/output operations. When you create a dataset from NIfTI files,
fmridataset automatically creates a NIfTI backend that knows how to read
those files efficiently. If you create a dataset from a matrix, it uses
a matrix backend optimized for in-memory data.

This separation between interface and implementation enables format
independence through the backend abstraction layer. Backend
implementations provide lazy loading for file-based data sources and
chunked processing for memory-constrained operations.

### Temporal Structure: More Than Just Time

fMRI data has rich temporal structure that goes beyond simple time
series. Runs may have different lengths, repetition times vary between
studies, and experimental events need to be aligned with acquisition
timing. The sampling frame abstraction captures all this temporal
complexity in a unified model.

Sampling frames maintain run boundary information and time-to-sample
index mappings. They provide methods for run-specific data extraction
and temporal unit conversion between seconds and TR indices.

### Data Chunking: Efficiency Without Complexity

Modern fMRI datasets can be enormous, often exceeding available memory.
Traditional approaches require you to manually split data into pieces
and carefully manage memory usage. fmridataset provides automatic data
chunking that handles this complexity transparently.

The chunking system partitions datasets into memory-manageable segments
for independent processing. Chunk generation uses configurable
parameters (`nchunks`, `runwise`) to determine voxel groupings, with
each chunk containing data subsets and corresponding voxel metadata.

## Deep Dive: Creating and Using Datasets

With the fundamentals clear, let’s explore how to create datasets from
different sources and use their full capabilities effectively.

### Creating Datasets from Different Sources

#### Matrix Datasets: When Data is Already in Memory

Matrix datasets are perfect when you already have preprocessed data in R
or when working with simulated data. They provide the fastest access
since no file I/O is required:

``` r
# Create from existing matrix
data_matrix <- matrix(rnorm(1000 * 150), nrow = 150, ncol = 1000)

dataset <- matrix_dataset(
  datamat = data_matrix,
  TR = 2.5,
  run_length = c(75, 75) # Two equal runs
)

# Immediate access - no loading delay
data <- get_data_matrix(dataset)
cat("Matrix dataset dimensions:", dim(data), "\n")
#> Matrix dataset dimensions: 150 1000
```

Matrix datasets store all data in memory, providing optimal access
performance for datasets within available RAM limits.

#### File Datasets: Lazy Loading for Large Data

File datasets work with NIfTI files and implement lazy loading, where
data remains on disk until explicitly accessed:

``` r
# File-based dataset (paths would be real NIfTI files)
file_paths <- c(
  "/path/to/subject01_run1.nii.gz",
  "/path/to/subject01_run2.nii.gz"
)
mask_path <- "/path/to/brain_mask.nii.gz"

# Creating the dataset doesn't load any data
dataset <- fmri_file_dataset(
  scans = file_paths,
  mask = mask_path,
  TR = 2.0,
  run_length = c(180, 180)
)

# Data loads when first accessed
print(dataset) # Shows metadata only
# data <- get_data_matrix(dataset)  # This would trigger loading
```

File datasets implement lazy loading and apply brain mask filtering to
minimize memory requirements.

#### Memory Datasets: Working with NeuroVec Objects

If you’re already using the neuroim2 package, memory datasets let you
work with existing NeuroVec objects:

``` r
# Example with NeuroVec objects (requires neuroim2)
if (requireNamespace("neuroim2", quietly = TRUE)) {
  # Create example NeuroVec
  dims <- c(10, 10, 10, 100) # 10x10x10 voxels, 100 timepoints
  nvec <- neuroim2::NeuroVec(
    array(rnorm(prod(dims)), dims),
    space = neuroim2::NeuroSpace(dims)
  )

  # Create mask
  mask_dims <- dims[1:3]
  mask <- neuroim2::NeuroVol(
    array(1, mask_dims),
    space = neuroim2::NeuroSpace(mask_dims)
  )

  # Create dataset
  dataset <- fmri_mem_dataset(
    scans = list(nvec),
    mask = mask,
    TR = 2.0,
    run_length = 100
  )
}
```

Memory datasets preserve the spatial structure of NeuroVec objects while
providing the unified dataset interface.

### Accessing Data Efficiently

#### Direct Data Access

The primary way to get data is through
[`get_data_matrix()`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md),
which always returns a standard R matrix in timepoints × voxels
orientation:

``` r
# Get complete dataset
full_data <- get_data_matrix(dataset)
cat("Full data shape:", dim(full_data), "\n")
#> Full data shape: 100 1000

# Get specific runs
run1 <- get_data_matrix(dataset, run_id = 1)
run2 <- get_data_matrix(dataset, run_id = 2)
cat("Run 1 shape:", dim(run1), "\n")
#> Run 1 shape: 100 1000
cat("Run 2 shape:", dim(run2), "\n")
#> Run 2 shape: 100 1000

# Get multiple runs
runs_1_2 <- get_data_matrix(dataset, run_id = c(1, 2))
cat("Runs 1-2 combined shape:", dim(runs_1_2), "\n")
#> Runs 1-2 combined shape: 100 1000
```

For file-based datasets, these operations trigger data loading. The
package automatically applies brain masks and handles any necessary
format conversions.

#### Memory-Efficient Chunking

For large datasets, chunking enables processing without loading
everything into memory:

``` r
# Create chunks for processing
chunks <- data_chunks(dataset, nchunks = 4)

# Process each chunk independently
results <- list()
for (chunk in chunks) {
  cat(
    "Processing chunk", chunk$chunk_num,
    "with", ncol(chunk$data), "voxels\n"
  )

  # Your analysis here - each chunk$data is a standard matrix
  chunk_result <- colMeans(chunk$data) # Example: compute mean time series
  results[[chunk$chunk_num]] <- chunk_result
}

# Combine results
all_means <- do.call(c, results)
cat("Computed means for", length(all_means), "total voxels\n")
```

The chunking system automatically divides voxels optimally and provides
metadata about which voxels each chunk contains.

#### Run-wise Processing

Many fMRI analyses need to process runs separately. The chunking system
supports this directly:

``` r
# Create one chunk per run
run_chunks <- data_chunks(dataset, runwise = TRUE)

# Process runs independently
run_results <- list()
for (chunk in run_chunks) {
  cat(
    "Processing run", chunk$chunk_num,
    "with", nrow(chunk$data), "timepoints\n"
  )

  # Run-specific analysis
  run_results[[chunk$chunk_num]] <- analyze_run(chunk$data)
}
```

Run-wise chunking ensures each chunk contains data from exactly one run,
making it perfect for analyses that require run boundaries.

### Working with Temporal Structure

#### Understanding Sampling Frames

Every dataset contains a sampling frame that captures temporal
organization:

``` r
# Access the sampling frame
sf <- dataset$sampling_frame
print(sf)
#> Sampling Frame
#> ==============
#> 
#> Structure:
#>   1 block
#>   Total scans: 100
#> 
#> Timing:
#>   TR: 2 s
#>   Precision: 0.1 s
#> 
#> Duration:
#>   Total time: 200.0 s

# Query temporal properties
cat("TR:", get_TR(sf), "seconds\n")
#> TR: 2 seconds
cat("Number of runs:", n_runs(sf), "\n")
#> Number of runs: 1
cat("Run lengths:", get_run_lengths(sf), "timepoints\n")
#> Run lengths: 100 timepoints
cat("Total duration:", get_total_duration(sf), "seconds\n")
#> Total duration: 200 seconds

# Get timing information
run_durations <- get_run_duration(sf)
cat("Run durations:", run_durations, "seconds\n")
#> Run durations: 200 seconds

# Access sample indices for each run
run_samples <- samples(sf)
cat("Run 1 samples:", head(run_samples[[1]]), "...\n")
#> Run 1 samples: 1 ...
cat("Run 2 samples:", head(run_samples[[2]]), "...\n")
#> Run 2 samples: 2 ...
```

The sampling frame provides all the temporal metadata needed for
sophisticated time series analyses.

#### Converting Between Time and Samples

Sampling frames enable easy conversion between time units and sample
indices:

``` r
# Convert event times to sample indices
event_onsets <- c(30, 75, 130, 175) # seconds
TR <- get_TR(dataset$sampling_frame)

# Manual conversion
sample_indices <- round(event_onsets / TR) + 1
cat("Event onsets at samples:", sample_indices, "\n")
#> Event onsets at samples: 16 39 66 89

# Use sampling frame for run-aware conversions
for (run_idx in 1:n_runs(dataset$sampling_frame)) {
  run_start <- samples(dataset$sampling_frame)[[run_idx]][1]
  cat("Run", run_idx, "starts at global sample", run_start, "\n")
}
#> Run 1 starts at global sample 1
```

This timing awareness is crucial for proper event-related analyses and
temporal alignment.

## Advanced Topics

Once you’re comfortable with basic dataset creation and access, these
advanced techniques help you handle complex scenarios and optimize
performance.

### Memory Management Strategies

Understanding when data is loaded helps you optimize memory usage for
large datasets:

``` r
# File datasets: creation is cheap, access triggers loading
file_dataset <- fmri_file_dataset(file_paths, mask_path, TR = 2.0)
cat("Created file dataset (no data loaded yet)\n")

# First access loads data
first_access <- get_data_matrix(file_dataset, run_id = 1)
cat("Loaded run 1:", object.size(first_access), "bytes\n")

# Subsequent accesses use cached data
second_access <- get_data_matrix(file_dataset, run_id = 1)
cat("Second access (cached)\n")

# Convert to matrix dataset to keep everything in memory
matrix_version <- as.matrix_dataset(file_dataset)
cat("Converted to matrix dataset\n")
```

Use file datasets for exploration and convert to matrix datasets when
you need repeated fast access to the same data.

### Custom Event Integration

Experimental design information integrates seamlessly with temporal
structure:

``` r
# Create detailed event table
events <- data.frame(
  onset = c(10, 30, 50, 70, 110, 130, 150, 170),
  duration = c(2, 2, 2, 2, 2, 2, 2, 2),
  trial_type = rep(c("faces", "houses"), 4),
  amplitude = c(1, 1, 1, 1, 1, 1, 1, 1),
  run = c(1, 1, 1, 1, 2, 2, 2, 2)
)

# Add to dataset
dataset$event_table <- events

# Analyze events in context of temporal structure
TR <- get_TR(dataset$sampling_frame)
for (i in 1:nrow(events)) {
  onset_sample <- round(events$onset[i] / TR) + 1
  run_id <- events$run[i]

  cat(
    "Event", i, ":", events$trial_type[i],
    "at sample", onset_sample, "in run", run_id, "\n"
  )
}
#> Event 1 : faces at sample 6 in run 1 
#> Event 2 : houses at sample 16 in run 1 
#> Event 3 : faces at sample 26 in run 1 
#> Event 4 : houses at sample 36 in run 1 
#> Event 5 : faces at sample 56 in run 2 
#> Event 6 : houses at sample 66 in run 2 
#> Event 7 : faces at sample 76 in run 2 
#> Event 8 : houses at sample 86 in run 2

# Extract data around events
extract_event_data <- function(dataset, event_row, window = c(-2, 8)) {
  TR <- get_TR(dataset$sampling_frame)
  onset_sample <- round(event_row$onset / TR) + 1
  run_id <- event_row$run

  # Get samples for this window
  samples <- (onset_sample + window[1]):(onset_sample + window[2])

  # Extract data for this run and time window
  run_data <- get_data_matrix(dataset, run_id = run_id)
  event_data <- run_data[samples, , drop = FALSE]

  return(event_data)
}

# Example: extract data around first event
event1_data <- extract_event_data(dataset, events[1, ])
cat("Event 1 data shape:", dim(event1_data), "\n")
#> Event 1 data shape: 11 1000
```

This integration of experimental design with temporal structure enables
sophisticated event-related analyses.

### Performance Optimization

Several strategies can significantly improve performance for large
datasets:

``` r
# Strategy 1: Use appropriate chunk sizes
small_chunks <- data_chunks(dataset, nchunks = 20) # Many small chunks
large_chunks <- data_chunks(dataset, nchunks = 4) # Fewer large chunks

# Strategy 2: Process runs separately when appropriate
run_chunks <- data_chunks(dataset, runwise = TRUE)

# Strategy 3: Use partial loading for file datasets
partial_data <- get_data_matrix(dataset, run_id = 1) # Load only one run

# Strategy 4: Monitor memory usage
if (requireNamespace("pryr", quietly = TRUE)) {
  cat("Memory before loading:", pryr::mem_used(), "\n")
  data <- get_data_matrix(dataset)
  cat("Memory after loading:", pryr::mem_used(), "\n")
}
```

Choose chunk sizes based on available memory and processing
requirements. More chunks mean less memory usage but more overhead.

## Tips and Best Practices

Here are practical guidelines learned from real-world usage that will
help you avoid common pitfalls and work more effectively.

### Memory Management

**Important**: For file-based datasets exceeding available RAM, use
partial loading by specifying `run_id` in
[`get_data_matrix()`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md).
Loading complete datasets should only occur when all runs are required
simultaneously for analysis.

When working with large datasets, memory management becomes crucial:

``` r
# Good: Process in chunks
analyze_large_dataset <- function(dataset) {
  chunks <- data_chunks(dataset, nchunks = 10)
  results <- list()

  for (chunk in chunks) {
    # Process chunk
    result <- your_analysis(chunk$data)
    results[[chunk$chunk_num]] <- result

    # Optionally force garbage collection
    if (chunk$chunk_num %% 5 == 0) {
      gc()
    }
  }

  return(do.call(rbind, results))
}

# Bad: Loading everything at once for large datasets
# big_data <- get_data_matrix(very_large_dataset)  # May exhaust memory
```

Monitor memory usage and adjust chunk sizes based on your system’s
capabilities.

### Data Validation

**Required Check**: Validate temporal structure using
[`get_run_lengths()`](https://bbuchsbaum.github.io/fmridataset/reference/get_run_lengths.md)
to ensure alignment with experimental design. Run length mismatches
indicate data organization errors that will propagate through subsequent
analyses.

### Chunking Configuration

**Run-Boundary Operations**: Configure
`data_chunks(dataset, runwise = TRUE)` for analyses requiring run
boundary preservation, including: - Temporal detrending - Motion
parameter regression - Run-level normalization - Temporal filtering

### Error Prevention

Common validation checks that prevent downstream errors:

``` r
validate_dataset <- function(dataset) {
  # Check temporal consistency
  sf <- dataset$sampling_frame
  expected_timepoints <- n_timepoints(sf)

  if (inherits(dataset, "matrix_dataset")) {
    actual_timepoints <- nrow(dataset$datamat)
    if (expected_timepoints != actual_timepoints) {
      stop(
        "Temporal structure mismatch: expected ", expected_timepoints,
        " timepoints, found ", actual_timepoints
      )
    }
  }

  # Check event table consistency
  if (nrow(dataset$event_table) > 0) {
    max_run <- max(dataset$event_table$run)
    n_runs_data <- n_runs(sf)

    if (max_run > n_runs_data) {
      stop(
        "Events reference run ", max_run,
        " but dataset only has ", n_runs_data, " runs"
      )
    }

    # Check event timing
    total_duration <- get_total_duration(sf)
    max_event_time <- max(dataset$event_table$onset + dataset$event_table$duration)

    if (max_event_time > total_duration) {
      warning(
        "Events extend beyond scan duration (",
        max_event_time, " > ", total_duration, " seconds)"
      )
    }
  }

  cat("Dataset validation passed\n")
}

# validate_dataset(dataset)
```

Early validation catches problems before they affect your analysis.

### Reproducibility

Document dataset creation for reproducible research:

``` r
# Create metadata record
create_dataset_record <- function(dataset) {
  list(
    timestamp = Sys.time(),
    r_version = R.version.string,
    fmridataset_version = packageVersion("fmridataset"),
    dataset_class = class(dataset)[1],
    temporal_structure = list(
      n_runs = n_runs(dataset$sampling_frame),
      run_lengths = get_run_lengths(dataset$sampling_frame),
      TR = get_TR(dataset$sampling_frame)
    ),
    spatial_structure = list(
      n_voxels = if (inherits(dataset, "matrix_dataset")) {
        ncol(dataset$datamat)
      } else {
        "unknown"
      }
    ),
    events = list(
      n_events = nrow(dataset$event_table),
      event_types = if (nrow(dataset$event_table) > 0) {
        unique(dataset$event_table$trial_type)
      } else {
        character(0)
      }
    )
  )
}

# Store metadata
# metadata <- create_dataset_record(dataset)
# saveRDS(metadata, "analysis_dataset_metadata.rds")
```

Good metadata practices make your analyses reproducible and help
collaborators understand your data structure.

## Troubleshooting

When things don’t work as expected, these solutions address the most
common issues encountered in real usage.

### Common Error Messages

- **“Error: Run lengths do not sum to number of timepoints”**:

  This occurs when the `run_length` parameter doesn’t match your actual
  data dimensions. Check that `sum(run_length)` equals the number of
  timepoints in your data matrix or files.

- **“Error: Cannot read file: \[filename\]”**:

  File path issues are common. Use
  [`file.exists()`](https://rdrr.io/r/base/files.html) to verify paths
  and ensure you’re using absolute paths or correct relative paths from
  your working directory.

- **“Warning: Events extend beyond scan duration”**: Your event table
  contains onsets or durations that exceed the total scan time. Verify
  event timing and units (seconds vs. TRs).

- **“Error: Mask dimensions do not match data dimensions”**: The spatial
  dimensions of your mask don’t match your data. For matrix datasets,
  the mask should have one element per column. For file datasets, mask
  spatial dimensions must match the data files.

### Performance Issues

If dataset operations are slow:

1.  **Check file paths**: Network drives or compressed files can be slow
    to read
2.  **Monitor memory**: Use [`gc()`](https://rdrr.io/r/base/gc.html) and
    consider smaller chunk sizes
3.  **Use run-specific access**: Load only needed runs with `run_id`
    parameter
4.  **Consider format conversion**: Convert frequently-accessed file
    datasets to matrix datasets

``` r
# Benchmark different access patterns
if (requireNamespace("microbenchmark", quietly = TRUE)) {
  # Compare full vs. partial loading
  mb <- microbenchmark::microbenchmark(
    full_data = get_data_matrix(dataset),
    run1_only = get_data_matrix(dataset, run_id = 1),
    times = 5
  )
  print(mb)
}
```

### Memory Issues

When you encounter memory problems:

``` r
# Monitor memory usage during operations
memory_profile <- function(dataset) {
  if (requireNamespace("pryr", quietly = TRUE)) {
    start_mem <- pryr::mem_used()
    cat("Starting memory:", start_mem, "\n")

    # Try loading data
    tryCatch(
      {
        data <- get_data_matrix(dataset, run_id = 1)
        loaded_mem <- pryr::mem_used()
        cat("After loading run 1:", loaded_mem, "\n")
        cat("Data size:", object.size(data), "\n")

        rm(data)
        gc()
        final_mem <- pryr::mem_used()
        cat("After cleanup:", final_mem, "\n")
      },
      error = function(e) {
        cat("Memory error:", conditionMessage(e), "\n")
      }
    )
  }
}

# memory_profile(dataset)
```

Use chunking for datasets that exceed available memory, and consider
processing runs separately.

## Integration with Other Vignettes

This introduction connects to several other topics in the fmridataset
ecosystem:

**Next Steps**: - [Architecture
Overview](https://bbuchsbaum.github.io/fmridataset/articles/architecture-overview.md) -
Understand the design principles and extensibility model - [Study-Level
Analysis](https://bbuchsbaum.github.io/fmridataset/articles/study-level-analysis.md) -
Scale from single subjects to multi-subject studies - [H5 Backend
Usage](https://bbuchsbaum.github.io/fmridataset/articles/h5-backend-usage.md) -
Use HDF5 for efficient storage of large datasets

**Advanced Topics**: - [Backend
Registry](https://bbuchsbaum.github.io/fmridataset/articles/backend-registry.md) -
Create custom backends for new data formats  
- [Extending
Backends](https://bbuchsbaum.github.io/fmridataset/articles/extending-backends.md) -
Deep dive into backend development

**Related Packages**: fmridataset integrates seamlessly with the broader
neuroimaging ecosystem: - **neuroim2**: Use NeuroVec objects directly
with memory datasets - **fmrireg**: Leverage temporal structure for
regression modeling - **DelayedArray**: Advanced array operations with
lazy evaluation

## Session Information

``` r
sessionInfo()
#> R version 4.5.2 (2025-10-31)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.3 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] fmridataset_0.8.9
#> 
#> loaded via a namespace (and not attached):
#>  [1] gtable_0.3.6          xfun_0.56             bslib_0.9.0          
#>  [4] ggplot2_4.0.1         lattice_0.22-7        bigassertr_0.1.7     
#>  [7] numDeriv_2016.8-1.1   vctrs_0.7.0           tools_4.5.2          
#> [10] generics_0.1.4        stats4_4.5.2          parallel_4.5.2       
#> [13] tibble_3.3.1          pkgconfig_2.0.3       Matrix_1.7-4         
#> [16] RColorBrewer_1.1-3    bigstatsr_1.6.2       S4Vectors_0.48.0     
#> [19] S7_0.2.1              desc_1.4.3            RcppParallel_5.1.11-1
#> [22] assertthat_0.2.1      lifecycle_1.0.5       compiler_4.5.2       
#> [25] neuroim2_0.8.3        farver_2.1.2          stringr_1.6.0        
#> [28] textshaping_1.0.4     RNifti_1.9.0          bigparallelr_0.3.2   
#> [31] codetools_0.2-20      htmltools_0.5.9       sass_0.4.10          
#> [34] yaml_2.3.12           deflist_0.2.0         pillar_1.11.1        
#> [37] pkgdown_2.2.0         crayon_1.5.3          jquerylib_0.1.4      
#> [40] RNiftyReg_2.8.4       cachem_1.1.0          DelayedArray_0.36.0  
#> [43] dbscan_1.2.4          iterators_1.0.14      abind_1.4-8          
#> [46] foreach_1.5.2         tidyselect_1.2.1      digest_0.6.39        
#> [49] stringi_1.8.7         dplyr_1.1.4           purrr_1.2.1          
#> [52] splines_4.5.2         cowplot_1.2.0         fastmap_1.2.0        
#> [55] grid_4.5.2            mmap_0.6-23           SparseArray_1.10.8   
#> [58] cli_3.6.5             magrittr_2.0.4        S4Arrays_1.10.1      
#> [61] fmrihrf_0.1.0.9000    scales_1.4.0          XVector_0.50.0       
#> [64] rmarkdown_2.30        matrixStats_1.5.0     rmio_0.4.0           
#> [67] ragg_1.5.0            memoise_2.0.1         evaluate_1.0.5       
#> [70] knitr_1.51            IRanges_2.44.0        doParallel_1.0.17    
#> [73] rlang_1.1.7           Rcpp_1.1.1            glue_1.8.0           
#> [76] BiocGenerics_0.56.0   jsonlite_2.0.0        R6_2.6.1             
#> [79] MatrixGenerics_1.22.0 systemfonts_1.3.1     fs_1.6.6             
#> [82] flock_0.7
```
