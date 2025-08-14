#' Backend Registry System Example
#' 
#' This example demonstrates how to create and register custom backends
#' using the fmridataset backend registry system.

library(fmridataset)

# ============================================================================
# Example 1: Simple Custom Backend
# ============================================================================

#' Create a simple backend factory for list-based data
simple_list_backend <- function(data_list, mask = NULL, ...) {
  # Convert list of vectors to matrix (timepoints x voxels)
  data_matrix <- do.call(rbind, data_list)
  
  # Create default mask if not provided
  if (is.null(mask)) {
    mask <- rep(TRUE, ncol(data_matrix))
  }
  
  # Create backend object
  backend <- list(
    data_matrix = data_matrix,
    mask = mask,
    spatial_dims = c(ncol(data_matrix), 1, 1),  # Flat 3D space
    metadata = list(format = "simple_list", created = Sys.time())
  )
  
  class(backend) <- c("simple_list_backend", "storage_backend")
  backend
}

# Implement required S3 methods
backend_open.simple_list_backend <- function(backend) {
  # No resources to open for in-memory backend
  backend
}

backend_close.simple_list_backend <- function(backend) {
  # No resources to close
  invisible(NULL)
}

backend_get_dims.simple_list_backend <- function(backend) {
  list(
    spatial = backend$spatial_dims,
    time = nrow(backend$data_matrix)
  )
}

backend_get_mask.simple_list_backend <- function(backend) {
  backend$mask
}

backend_get_data.simple_list_backend <- function(backend, rows = NULL, cols = NULL) {
  # Apply mask first
  data <- backend$data_matrix[, backend$mask, drop = FALSE]
  
  # Apply row subsetting
  if (!is.null(rows)) {
    data <- data[rows, , drop = FALSE]
  }
  
  # Apply column subsetting  
  if (!is.null(cols)) {
    data <- data[, cols, drop = FALSE]
  }
  
  data
}

backend_get_metadata.simple_list_backend <- function(backend) {
  backend$metadata
}

# Register the backend
register_backend(
  name = "simple_list",
  factory = simple_list_backend,
  description = "Simple backend for list-based time series data"
)

# Test the backend
data_list <- list(
  rnorm(10),  # timepoint 1: 10 voxels
  rnorm(10),  # timepoint 2: 10 voxels  
  rnorm(10),  # timepoint 3: 10 voxels
  rnorm(10)   # timepoint 4: 10 voxels
)

# Create backend instance
backend <- create_backend("simple_list", data_list = data_list)
print(backend)

# Test backend functionality
cat("Backend dimensions:", toString(backend_get_dims(backend)), "\n")
cat("Mask length:", length(backend_get_mask(backend)), "\n")
cat("Data shape:", toString(dim(backend_get_data(backend))), "\n")

# Use in fMRI dataset
dataset <- fmri_dataset(backend, TR = 2.0, run_length = 4)
print(dataset)

# ============================================================================
# Example 2: CSV Backend with Custom Validation
# ============================================================================

#' CSV backend with file I/O and custom validation
csv_backend <- function(csv_file, mask_file = NULL, transpose = FALSE, ...) {
  # Validate file exists
  if (!file.exists(csv_file)) {
    stop("CSV file does not exist: ", csv_file)
  }
  
  # Read data
  data_matrix <- as.matrix(read.csv(csv_file, header = FALSE))
  
  # Transpose if requested (useful if CSV has voxels in rows)
  if (transpose) {
    data_matrix <- t(data_matrix)
  }
  
  # Read or create mask
  if (!is.null(mask_file) && file.exists(mask_file)) {
    mask <- as.logical(read.csv(mask_file, header = FALSE)[[1]])
  } else {
    mask <- rep(TRUE, ncol(data_matrix))
  }
  
  # Validate dimensions
  if (length(mask) != ncol(data_matrix)) {
    stop("Mask length (", length(mask), ") doesn't match number of columns (", 
         ncol(data_matrix), ")")
  }
  
  backend <- list(
    csv_file = csv_file,
    mask_file = mask_file,
    data_matrix = data_matrix,
    mask = mask,
    spatial_dims = c(ncol(data_matrix), 1, 1),
    metadata = list(
      format = "CSV",
      source_file = csv_file,
      transpose = transpose
    )
  )
  
  class(backend) <- c("csv_backend", "storage_backend")
  backend
}

# Implement S3 methods for CSV backend
backend_open.csv_backend <- function(backend) backend
backend_close.csv_backend <- function(backend) invisible(NULL)

backend_get_dims.csv_backend <- function(backend) {
  list(
    spatial = backend$spatial_dims,
    time = nrow(backend$data_matrix)
  )
}

backend_get_mask.csv_backend <- function(backend) backend$mask

backend_get_data.csv_backend <- function(backend, rows = NULL, cols = NULL) {
  data <- backend$data_matrix[, backend$mask, drop = FALSE]
  if (!is.null(rows)) data <- data[rows, , drop = FALSE]
  if (!is.null(cols)) data <- data[, cols, drop = FALSE]
  data
}

backend_get_metadata.csv_backend <- function(backend) backend$metadata

# Custom validation function
csv_validator <- function(backend) {
  # Check data is numeric
  if (!is.numeric(backend$data_matrix)) {
    stop("CSV data must be numeric")
  }
  
  # Check minimum timepoints
  if (nrow(backend$data_matrix) < 5) {
    stop("CSV backend requires at least 5 timepoints, got ", 
         nrow(backend$data_matrix))
  }
  
  # Check for missing values
  if (any(is.na(backend$data_matrix))) {
    stop("CSV data contains missing values")
  }
  
  TRUE
}

# Register CSV backend with validation
register_backend(
  name = "csv",
  factory = csv_backend,
  description = "CSV file backend with custom validation",
  validate_function = csv_validator
)

# Create example CSV file for testing
temp_csv <- tempfile(fileext = ".csv")
test_data <- matrix(rnorm(50 * 20), 50, 20)  # 50 timepoints, 20 voxels
write.table(test_data, temp_csv, sep = ",", 
            row.names = FALSE, col.names = FALSE)

# Test CSV backend
csv_backend_instance <- create_backend("csv", csv_file = temp_csv)
cat("CSV backend created successfully!\n")
cat("Dimensions:", toString(backend_get_dims(csv_backend_instance)), "\n")

# Clean up
unlink(temp_csv)

# ============================================================================
# Registry Management Examples
# ============================================================================

# List all registered backends
cat("\nRegistered backends:\n")
print(list_backend_names())

# Get detailed information about backends
cat("\nBackend registry information:\n")
registry <- get_backend_registry()
class(registry) <- "backend_registry"
print(registry)

# Check if specific backends are registered
cat("\nBackend availability:\n")
cat("Matrix backend:", is_backend_registered("matrix"), "\n")
cat("CSV backend:", is_backend_registered("csv"), "\n")
cat("Simple list backend:", is_backend_registered("simple_list"), "\n")
cat("Nonexistent backend:", is_backend_registered("nonexistent"), "\n")

# Get information about a specific backend
nifti_info <- get_backend_registry("nifti")
cat("\nNIfTI backend info:\n")
cat("Description:", nifti_info$description, "\n")
cat("Registered at:", format(nifti_info$registered_at), "\n")

# ============================================================================
# Cleanup
# ============================================================================

# Unregister custom backends (optional - they'll be cleaned up when R exits)
unregister_backend("simple_list")
unregister_backend("csv")

cat("\nBackend registry example completed successfully!\n")