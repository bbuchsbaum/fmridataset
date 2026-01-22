# Motivation: Why Create Custom Backends?

Your research group has been using a custom MATLAB pipeline that exports
preprocessed fMRI data as JSON files with separate metadata. The data
format is optimized for your specific analyses, includes custom quality
metrics, and integrates with your lab’s database system. Rather than
converting all this data to standard formats or writing custom loading
code for each analysis, you can create a backend that makes this format
work seamlessly with fmridataset.

Creating a custom backend means your specialized data format immediately
gains access to all fmridataset features: unified interfaces, efficient
chunking, study-level operations, and compatibility with the entire
ecosystem. This vignette teaches you the essentials of backend
development through practical examples, showing you how to implement the
required interface and optimize for your specific use case.

# A Complete Example: JSON Backend

Let’s create a working backend for JSON-formatted fMRI data that
demonstrates all the essential concepts:

``` r
# Create a complete JSON backend implementation
json_backend <- function(json_file, metadata_file = NULL) {
  # Input validation
  if (!file.exists(json_file)) {
    stop("JSON file not found: ", json_file)
  }
  
  # Initialize backend structure
  backend <- list(
    json_file = json_file,
    metadata_file = metadata_file,
    data_cache = NULL,
    dims_cache = NULL,
    is_open = FALSE
  )
  
  class(backend) <- c("json_backend", "storage_backend")
  backend
}

# Implement the open method
backend_open.json_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)  # Already open
  }
  
  # Simulate reading JSON data (in practice, use jsonlite::fromJSON)
  # For demonstration, create synthetic data
  set.seed(123)
  n_time <- 100
  n_voxels <- 500
  
  backend$data_cache <- matrix(
    rnorm(n_time * n_voxels),
    nrow = n_time,
    ncol = n_voxels
  )
  
  backend$dims_cache <- list(
    spatial = c(n_voxels, 1, 1),  # Flat spatial structure
    time = n_time
  )
  
  backend$is_open <- TRUE
  backend
}

# Implement the close method
backend_close.json_backend <- function(backend) {
  backend$data_cache <- NULL
  backend$dims_cache <- NULL
  backend$is_open <- FALSE
  invisible(backend)
}

# Implement dimension query
backend_get_dims.json_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened first")
  }
  backend$dims_cache
}

# Implement data access
backend_get_data.json_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) {
    stop("Backend must be opened first")
  }
  
  data <- backend$data_cache
  
  # Handle subsetting
  if (!is.null(rows)) {
    data <- data[rows, , drop = FALSE]
  }
  if (!is.null(cols)) {
    data <- data[, cols, drop = FALSE]
  }
  
  data
}

# Implement mask generation
backend_get_mask.json_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened first")
  }
  
  # All voxels are valid in our JSON format
  rep(TRUE, backend$dims_cache$spatial[1])
}

# Test the backend
json_file <- tempfile(fileext = ".json")
writeLines("{}", json_file)  # Create dummy file

backend <- json_backend(json_file)
backend <- backend_open(backend)

dims <- backend_get_dims(backend)
cat("Backend dimensions - Time:", dims$time, "Spatial:", dims$spatial[1], "\n")
#> Backend dimensions - Time: 100 Spatial: 500

# Get some data
sample_data <- backend_get_data(backend, rows = 1:10, cols = 1:50)
cat("Retrieved data shape:", dim(sample_data), "\n")
#> Retrieved data shape: 10 50

backend_close(backend)
```

> **💡 Key Insight**: A backend only needs to implement five methods to
> work with the entire fmridataset ecosystem. This simple interface
> provides tremendous power and flexibility.

# Understanding the Backend Contract

The backend contract defines the minimal interface that all backends
must implement. Understanding this contract is essential for creating
compatible backends.

## Required Methods

Every backend must implement these five S3 methods:

### 1. backend_open()

Opens the backend and acquires resources (file handles, connections,
memory). This method should be idempotent - calling it multiple times on
an open backend should be safe.

### 2. backend_close()

Releases all resources and cleans up. After closing, the backend should
not hold any external resources.

### 3. backend_get_dims()

Returns dimensions as a list with `spatial` (3-element vector) and
`time` (single integer) elements. This must work without loading all
data.

### 4. backend_get_data()

Returns data in timepoints × voxels orientation. Must support optional
row and column subsetting for efficient partial loading.

### 5. backend_get_mask()

Returns a logical vector indicating valid voxels. Length must equal the
product of spatial dimensions.

## Optional Methods

These methods enhance functionality but aren’t required:

``` r
# Optional: Metadata extraction
backend_get_metadata.json_backend <- function(backend) {
  if (!backend$is_open) {
    stop("Backend must be opened first")
  }
  
  list(
    format = "json",
    compression = "none",
    creation_date = Sys.Date(),
    custom_metrics = list(
      quality_score = 0.95,
      motion_level = "low"
    )
  )
}

# Optional: Validation
backend_validate.json_backend <- function(backend) {
  # Check data integrity
  if (!backend$is_open) {
    return(FALSE)
  }
  
  # Validate dimensions
  dims <- backend$dims_cache
  expected_size <- dims$time * dims$spatial[1]
  actual_size <- length(backend$data_cache)
  
  if (expected_size != actual_size) {
    warning("Data size mismatch")
    return(FALSE)
  }
  
  TRUE
}

# Test optional methods
backend <- backend_open(json_backend(json_file))
metadata <- backend_get_metadata(backend)
cat("Format:", metadata$format, "\n")
#> Format: json
cat("Quality score:", metadata$custom_metrics$quality_score, "\n")
#> Quality score: 0.95

is_valid <- backend_validate(backend)
#> Error in backend_validate(backend): could not find function "backend_validate"
cat("Backend valid:", is_valid, "\n")
#> Error: object 'is_valid' not found
backend_close(backend)
```

# Deep Dive: Implementation Patterns

Let’s explore common patterns that make backends robust and efficient.

## State Management Pattern

Backends need to track whether they’re open and manage resources
appropriately:

``` r
# Robust state management example
stateful_backend <- function(source) {
  backend <- list(
    source = source,
    # State flags
    is_open = FALSE,
    is_validated = FALSE,
    has_error = FALSE,
    # Resource tracking
    resources = list(),
    # Error information
    last_error = NULL
  )
  
  class(backend) <- c("stateful_backend", "storage_backend")
  backend
}

# Safe resource acquisition
backend_open.stateful_backend <- function(backend) {
  if (backend$is_open) {
    return(backend)  # Idempotent
  }
  
  tryCatch({
    # Acquire resources
    backend$resources$data <- matrix(rnorm(1000), 100, 10)
    backend$is_open <- TRUE
    backend$has_error <- FALSE
  }, error = function(e) {
    backend$has_error <- TRUE
    backend$last_error <- conditionMessage(e)
    stop("Failed to open backend: ", conditionMessage(e))
  })
  
  backend
}

# Safe resource cleanup
backend_close.stateful_backend <- function(backend) {
  if (!backend$is_open) {
    return(invisible(backend))  # Already closed
  }
  
  # Release all resources
  backend$resources <- list()
  backend$is_open <- FALSE
  
  invisible(backend)
}

# Implement other required methods...
backend_get_dims.stateful_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  list(spatial = c(10, 1, 1), time = 100)
}

backend_get_data.stateful_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) stop("Backend not open")
  data <- backend$resources$data
  if (!is.null(rows)) data <- data[rows, , drop = FALSE]
  if (!is.null(cols)) data <- data[, cols, drop = FALSE]
  data
}

backend_get_mask.stateful_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  rep(TRUE, 10)
}

# Test state management
backend <- stateful_backend("dummy_source")
cat("Initial state - is_open:", backend$is_open, "\n")
#> Initial state - is_open: FALSE

backend <- backend_open(backend)
cat("After open - is_open:", backend$is_open, "\n")
#> After open - is_open: TRUE

backend <- backend_close(backend)
cat("After close - is_open:", backend$is_open, "\n")
#> After close - is_open: FALSE
```

## Lazy Loading Pattern

Implement lazy loading to defer expensive operations:

``` r
# Lazy loading backend
lazy_backend <- function(data_source) {
  backend <- list(
    data_source = data_source,
    is_open = FALSE,
    # Lazy caches
    dims_cache = NULL,
    data_cache = NULL,
    mask_cache = NULL
  )
  
  class(backend) <- c("lazy_backend", "storage_backend")
  backend
}

backend_open.lazy_backend <- function(backend) {
  backend$is_open <- TRUE
  # Don't load data yet!
  backend
}

backend_get_dims.lazy_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  
  # Load dimensions only when first requested
  if (is.null(backend$dims_cache)) {
    # In practice, read just headers/metadata
    backend$dims_cache <- list(
      spatial = c(100, 1, 1),
      time = 50
    )
  }
  
  backend$dims_cache
}

backend_get_data.lazy_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) stop("Backend not open")
  
  # Load data only when first accessed
  if (is.null(backend$data_cache)) {
    cat("Loading data (lazy)...\n")
    backend$data_cache <- matrix(rnorm(5000), 50, 100)
  }
  
  data <- backend$data_cache
  if (!is.null(rows)) data <- data[rows, , drop = FALSE]
  if (!is.null(cols)) data <- data[, cols, drop = FALSE]
  data
}

backend_get_mask.lazy_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  
  if (is.null(backend$mask_cache)) {
    dims <- backend_get_dims(backend)
    backend$mask_cache <- rep(TRUE, dims$spatial[1])
  }
  
  backend$mask_cache
}

backend_close.lazy_backend <- function(backend) {
  backend$dims_cache <- NULL
  backend$data_cache <- NULL
  backend$mask_cache <- NULL
  backend$is_open <- FALSE
  invisible(backend)
}

# Demonstrate lazy loading
backend <- lazy_backend("source")
backend <- backend_open(backend)

cat("Getting dimensions...\n")
#> Getting dimensions...
dims <- backend_get_dims(backend)  # No data loading

cat("\nGetting mask...\n")
#> 
#> Getting mask...
mask <- backend_get_mask(backend)  # Still no data loading

cat("\nGetting data...\n")
#> 
#> Getting data...
data <- backend_get_data(backend, rows = 1:10)  # NOW data loads
#> Loading data (lazy)...

backend_close(backend)
```

## Validation Pattern

Implement validation to ensure data integrity:

``` r
# Create validation utilities
validate_backend_contract <- function(backend_class) {
  required_methods <- c(
    "backend_open",
    "backend_close", 
    "backend_get_dims",
    "backend_get_data",
    "backend_get_mask"
  )
  
  missing_methods <- character()
  
  for (method in required_methods) {
    full_method <- paste0(method, ".", backend_class)
    if (!exists(full_method)) {
      missing_methods <- c(missing_methods, method)
    }
  }
  
  if (length(missing_methods) > 0) {
    stop("Backend class '", backend_class, "' missing required methods: ",
         paste(missing_methods, collapse = ", "))
  }
  
  cat("✓ Backend class '", backend_class, "' implements all required methods\n")
  TRUE
}

# Test our backends
validate_backend_contract("json_backend")
#> ✓ Backend class ' json_backend ' implements all required methods
#> [1] TRUE
validate_backend_contract("lazy_backend")
#> ✓ Backend class ' lazy_backend ' implements all required methods
#> [1] TRUE
```

# Advanced Topics

## Caching Strategies

Implement intelligent caching for better performance:

``` r
# Advanced caching backend
cached_backend <- function(source, cache_size_mb = 100) {
  backend <- list(
    source = source,
    cache_size_mb = cache_size_mb,
    is_open = FALSE,
    # Multi-level cache
    cache = list(
      dims = NULL,
      mask = NULL,
      data_blocks = list(),
      access_times = list()
    ),
    # Cache statistics
    stats = list(
      hits = 0,
      misses = 0,
      evictions = 0
    )
  )
  
  class(backend) <- c("cached_backend", "storage_backend")
  backend
}

# Implement cache management
cache_get_or_load <- function(backend, key, loader_fn) {
  if (!is.null(backend$cache[[key]])) {
    backend$stats$hits <- backend$stats$hits + 1
    cat("Cache hit for", key, "\n")
    return(backend$cache[[key]])
  }
  
  backend$stats$misses <- backend$stats$misses + 1
  cat("Cache miss for", key, "- loading...\n")
  
  value <- loader_fn()
  backend$cache[[key]] <- value
  backend$cache$access_times[[key]] <- Sys.time()
  
  value
}

backend_open.cached_backend <- function(backend) {
  backend$is_open <- TRUE
  backend
}

backend_get_dims.cached_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  
  cache_get_or_load(backend, "dims", function() {
    list(spatial = c(100, 1, 1), time = 50)
  })
}

backend_get_data.cached_backend <- function(backend, rows = NULL, cols = NULL) {
  if (!backend$is_open) stop("Backend not open")
  
  # Create cache key based on request
  cache_key <- paste0("data_", 
                     paste(range(rows %||% 1:50), collapse = "_"),
                     "_",
                     paste(range(cols %||% 1:100), collapse = "_"))
  
  data <- cache_get_or_load(backend, cache_key, function() {
    matrix(rnorm(5000), 50, 100)
  })
  
  if (!is.null(rows)) data <- data[rows, , drop = FALSE]
  if (!is.null(cols)) data <- data[, cols, drop = FALSE]
  data
}

backend_get_mask.cached_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  
  cache_get_or_load(backend, "mask", function() {
    rep(TRUE, 100)
  })
}

backend_close.cached_backend <- function(backend) {
  # Report cache statistics
  cat("\nCache statistics:\n")
  cat("  Hits:", backend$stats$hits, "\n")
  cat("  Misses:", backend$stats$misses, "\n")
  cat("  Hit rate:", 
      round(100 * backend$stats$hits / 
            (backend$stats$hits + backend$stats$misses), 1), "%\n")
  
  backend$cache <- list()
  backend$is_open <- FALSE
  invisible(backend)
}

# Demonstrate caching
`%||%` <- function(x, y) if (is.null(x)) y else x

backend <- cached_backend("source")
backend <- backend_open(backend)

# First access - cache miss
data1 <- backend_get_data(backend, rows = 1:10)
#> Cache miss for data_1_10_1_100 - loading...

# Second access - cache hit
data2 <- backend_get_data(backend, rows = 1:10)
#> Cache miss for data_1_10_1_100 - loading...

# Different subset - cache miss
data3 <- backend_get_data(backend, rows = 11:20)
#> Cache miss for data_11_20_1_100 - loading...

backend_close(backend)
#> 
#> Cache statistics:
#>   Hits: 0 
#>   Misses: 0 
#>   Hit rate: NaN %
```

## Error Handling

Robust error handling makes backends production-ready:

``` r
# Create a backend with comprehensive error handling
robust_backend <- function(source) {
  backend <- list(
    source = source,
    is_open = FALSE,
    error_log = list()
  )
  
  class(backend) <- c("robust_backend", "storage_backend")
  backend
}

# Helper to log errors
log_error <- function(backend, operation, error) {
  backend$error_log[[length(backend$error_log) + 1]] <- list(
    timestamp = Sys.time(),
    operation = operation,
    message = conditionMessage(error)
  )
  backend
}

backend_open.robust_backend <- function(backend) {
  tryCatch({
    if (backend$is_open) {
      warning("Backend already open")
      return(backend)
    }
    
    # Simulate potential failures
    if (runif(1) > 0.8) {
      stop("Simulated connection failure")
    }
    
    backend$is_open <- TRUE
    cat("Successfully opened backend\n")
    backend
    
  }, error = function(e) {
    backend <- log_error(backend, "open", e)
    stop("Failed to open backend: ", conditionMessage(e))
  })
}

backend_get_data.robust_backend <- function(backend, rows = NULL, cols = NULL) {
  tryCatch({
    if (!backend$is_open) {
      stop("Backend not open")
    }
    
    # Validate indices
    if (!is.null(rows) && any(rows < 1)) {
      stop("Invalid row indices")
    }
    
    if (!is.null(cols) && any(cols < 1)) {
      stop("Invalid column indices")
    }
    
    # Return data
    matrix(rnorm(5000), 50, 100)[rows %||% 1:50, cols %||% 1:100, drop = FALSE]
    
  }, error = function(e) {
    backend <- log_error(backend, "get_data", e)
    stop("Data access failed: ", conditionMessage(e))
  })
}

# Implement other methods...
backend_get_dims.robust_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  list(spatial = c(100, 1, 1), time = 50)
}

backend_get_mask.robust_backend <- function(backend) {
  if (!backend$is_open) stop("Backend not open")
  rep(TRUE, 100)
}

backend_close.robust_backend <- function(backend) {
  if (length(backend$error_log) > 0) {
    cat("\nError log:\n")
    for (error in backend$error_log) {
      cat("  -", error$operation, "at", format(error$timestamp), 
          ":", error$message, "\n")
    }
  }
  backend$is_open <- FALSE
  invisible(backend)
}

# Test error handling
set.seed(123)
backend <- robust_backend("source")

# May fail randomly
result <- tryCatch({
  backend <- backend_open(backend)
  data <- backend_get_data(backend, rows = 1:10)
  cat("Data retrieved successfully\n")
  backend_close(backend)
}, error = function(e) {
  cat("Caught error:", conditionMessage(e), "\n")
})
#> Successfully opened backend
#> Data retrieved successfully
```

# Tips and Best Practices

Here are essential guidelines for creating robust, efficient backends.

> **⚠️ Performance Tip**: Always implement lazy loading for large
> datasets. Load metadata and dimensions quickly, but defer data loading
> until actually needed.

> **🛡️ Best Practice**: Make your backend methods idempotent. Opening an
> already-open backend or closing an already-closed backend should be
> safe operations.

> **⚡ Pro Tip**: Cache computed values like masks and dimensions. These
> are often queried multiple times but rarely change.

## Backend Development Checklist

Before considering your backend complete:

``` r
backend_checklist <- function() {
  cat("Backend Development Checklist:\n\n")
  
  cat("Required Functionality:\n")
  cat("  ☐ Implements all 5 required methods\n")
  cat("  ☐ Returns correct data orientations\n")
  cat("  ☐ Handles NULL rows/cols in get_data\n")
  cat("  ☐ Returns valid dimension structure\n")
  cat("  ☐ Mask length matches spatial dimensions\n\n")
  
  cat("Robustness:\n")
  cat("  ☐ Validates inputs in constructor\n")
  cat("  ☐ Checks is_open state in all methods\n")
  cat("  ☐ Handles errors gracefully\n")
  cat("  ☐ Cleans up resources in close\n")
  cat("  ☐ Methods are idempotent\n\n")
  
  cat("Performance:\n")
  cat("  ☐ Implements lazy loading\n")
  cat("  ☐ Caches frequently accessed values\n")
  cat("  ☐ Minimizes memory footprint\n")
  cat("  ☐ Supports partial data loading\n\n")
  
  cat("Documentation:\n")
  cat("  ☐ Constructor documented\n")
  cat("  ☐ Error messages are informative\n")
  cat("  ☐ Usage examples provided\n")
  cat("  ☐ Performance characteristics noted\n")
}

backend_checklist()
#> Backend Development Checklist:
#> 
#> Required Functionality:
#>   ☐ Implements all 5 required methods
#>   ☐ Returns correct data orientations
#>   ☐ Handles NULL rows/cols in get_data
#>   ☐ Returns valid dimension structure
#>   ☐ Mask length matches spatial dimensions
#> 
#> Robustness:
#>   ☐ Validates inputs in constructor
#>   ☐ Checks is_open state in all methods
#>   ☐ Handles errors gracefully
#>   ☐ Cleans up resources in close
#>   ☐ Methods are idempotent
#> 
#> Performance:
#>   ☐ Implements lazy loading
#>   ☐ Caches frequently accessed values
#>   ☐ Minimizes memory footprint
#>   ☐ Supports partial data loading
#> 
#> Documentation:
#>   ☐ Constructor documented
#>   ☐ Error messages are informative
#>   ☐ Usage examples provided
#>   ☐ Performance characteristics noted
```

## Testing Your Backend

Comprehensive testing ensures reliability:

``` r
# Test suite for backends
test_backend <- function(backend_constructor, test_source) {
  cat("Testing backend implementation...\n\n")
  
  # Test construction
  cat("Testing construction...")
  backend <- backend_constructor(test_source)
  cat(" ✓\n")
  
  # Test opening
  cat("Testing open...")
  backend <- backend_open(backend)
  cat(" ✓\n")
  
  # Test dimensions
  cat("Testing dimensions...")
  dims <- backend_get_dims(backend)
  stopifnot(is.list(dims))
  stopifnot(all(c("spatial", "time") %in% names(dims)))
  stopifnot(length(dims$spatial) == 3)
  cat(" ✓\n")
  
  # Test mask
  cat("Testing mask...")
  mask <- backend_get_mask(backend)
  stopifnot(is.logical(mask))
  stopifnot(length(mask) == prod(dims$spatial))
  cat(" ✓\n")
  
  # Test data access
  cat("Testing data access...")
  data <- backend_get_data(backend)
  stopifnot(is.matrix(data))
  stopifnot(nrow(data) == dims$time)
  cat(" ✓\n")
  
  # Test subsetting
  cat("Testing subsetting...")
  subset_data <- backend_get_data(backend, rows = 1:10, cols = 1:20)
  stopifnot(dim(subset_data)[1] == 10)
  stopifnot(dim(subset_data)[2] == 20)
  cat(" ✓\n")
  
  # Test closing
  cat("Testing close...")
  backend_close(backend)
  cat(" ✓\n")
  
  cat("\n✅ All tests passed!\n")
}

# Test our JSON backend
test_backend(json_backend, json_file)
#> Testing backend implementation...
#> 
#> Testing construction... ✓
#> Testing open... ✓
#> Testing dimensions... ✓
#> Testing mask... ✓
#> Testing data access... ✓
#> Testing subsetting... ✓
#> Testing close... ✓
#> 
#> ✅ All tests passed!
```

# Troubleshooting

Common issues when developing backends and their solutions.

## Dimension Mismatches

**Problem**: “Error: Mask length does not match spatial dimensions”

**Solution**: Ensure `length(backend_get_mask(backend))` equals
`prod(backend_get_dims(backend)$spatial)`

## Memory Issues

**Problem**: Large datasets cause memory errors

**Solution**: Implement lazy loading and support partial data access
through row/column subsetting

## State Management

**Problem**: “Error: Backend not open” in methods

**Solution**: Always check `is_open` flag and provide informative error
messages

# Integration with Other Vignettes

This backend development guide connects to:

**Prerequisites**: - [Getting
Started](https://bbuchsbaum.github.io/fmridataset/fmridataset-intro.md) -
Understand how backends fit into the ecosystem - [Architecture
Overview](https://bbuchsbaum.github.io/fmridataset/architecture-overview.md) -
Learn the design principles

**Next Steps**: - [Backend
Registry](https://bbuchsbaum.github.io/fmridataset/backend-registry.md) -
Register your backend for automatic selection - [Advanced Backend
Patterns](https://bbuchsbaum.github.io/fmridataset/extending-backends.md) -
Sophisticated techniques for production backends

**Applications**: - [H5 Backend
Usage](https://bbuchsbaum.github.io/fmridataset/h5-backend-usage.md) -
See a production backend in action

# Session Information

``` r
sessionInfo()
#> R version 4.3.2 (2023-10-31)
#> Platform: aarch64-apple-darwin20 (64-bit)
#> Running under: macOS Sonoma 14.3
#> 
#> Matrix products: default
#> BLAS:   /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/lib/libRblas.0.dylib 
#> LAPACK: /Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.11.0
#> 
#> locale:
#> [1] en_CA.UTF-8/en_CA.UTF-8/en_CA.UTF-8/C/en_CA.UTF-8/en_CA.UTF-8
#> 
#> time zone: America/Toronto
#> tzcode source: internal
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] fmridataset_0.8.9
#> 
#> loaded via a namespace (and not attached):
#>   [1] checklist_0.4.2       remotes_2.5.0         rlang_1.1.6          
#>   [4] magrittr_2.0.3        hunspell_3.0.6        matrixStats_1.5.0    
#>   [7] compiler_4.3.2        callr_3.7.6           vctrs_0.6.5          
#>  [10] stringr_1.5.1         profvis_0.4.0         pkgconfig_2.0.3      
#>  [13] crayon_1.5.3          fastmap_1.2.0         backports_1.5.0      
#>  [16] XVector_0.42.0        ellipsis_0.3.2        promises_1.3.3       
#>  [19] rmarkdown_2.29        sessioninfo_1.2.3     ps_1.9.1             
#>  [22] fmrihrf_0.1.0         purrr_1.0.4           xfun_0.52            
#>  [25] gert_2.1.5            zlibbioc_1.48.2       cachem_1.1.0         
#>  [28] rmio_0.4.0            neuroim2_0.8.1        jsonlite_2.0.0       
#>  [31] later_1.4.2           DelayedArray_0.28.0   parallel_4.3.2       
#>  [34] prettyunits_1.2.0     R6_2.6.1              stringi_1.8.7        
#>  [37] RColorBrewer_1.1-3    pkgload_1.4.0         numDeriv_2016.8-1.1  
#>  [40] Rcpp_1.1.0            assertthat_0.2.1      iterators_1.0.14     
#>  [43] knitr_1.50            usethis_3.1.0         IRanges_2.36.0       
#>  [46] tidyselect_1.2.1      httpuv_1.6.16         Matrix_1.6-5         
#>  [49] splines_4.3.2         abind_1.4-8           yaml_2.3.10          
#>  [52] doParallel_1.0.17     codetools_0.2-19      miniUI_0.1.2         
#>  [55] curl_6.4.0            processx_3.8.6        pkgbuild_1.4.8       
#>  [58] lattice_0.21-9        tibble_3.3.0          shiny_1.10.0         
#>  [61] withr_3.0.2           askpass_1.2.1         evaluate_1.0.4       
#>  [64] desc_1.4.3            RcppParallel_5.1.10   urlchecker_1.0.1     
#>  [67] xml2_1.3.8            pillar_1.11.0         MatrixGenerics_1.14.0
#>  [70] RNifti_1.8.0          foreach_1.5.2         stats4_4.3.2         
#>  [73] rex_1.2.1             bigassertr_0.1.7      mmap_0.6-22          
#>  [76] dbscan_1.2.2          generics_0.1.4        pingr_2.0.5          
#>  [79] rprojroot_2.0.4       credentials_2.0.2     xopen_1.0.1          
#>  [82] S4Vectors_0.40.2      ggplot2_3.5.2         codemetar_0.3.5      
#>  [85] scales_1.4.0          xtable_1.8-4          glue_1.8.0           
#>  [88] lazyeval_0.2.2        tools_4.3.2           deflist_0.2.0        
#>  [91] sys_3.4.3             fs_1.6.6              cowplot_1.2.0        
#>  [94] grid_4.3.2            gh_1.5.0              colorspace_2.1-1     
#>  [97] lintr_3.2.0           devtools_2.4.5        flock_0.7            
#> [100] RNiftyReg_2.8.4       cli_3.6.5             bigparallelr_0.3.2   
#> [103] rcmdcheck_1.4.0       S4Arrays_1.2.1        dplyr_1.1.4          
#> [106] gtable_0.3.6          digest_0.6.37         BiocGenerics_0.48.1  
#> [109] SparseArray_1.2.4     htmlwidgets_1.6.4     farver_2.1.2         
#> [112] memoise_2.0.1         htmltools_0.5.8.1     pkgdown_2.1.3        
#> [115] lifecycle_1.0.4       httr_1.4.7            bigstatsr_1.6.1      
#> [118] mime_0.13             openssl_2.3.3
```
