# Demonstration of performance improvement with read_header optimization
library(fmridataset)
library(neuroim2)
library(microbenchmark)

# Use example NIfTI file
test_file <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")

cat("Performance comparison: read_header vs read_vec\n")
cat("================================================\n\n")

# Method 1: Using read_header (new optimized approach)
read_header_approach <- function(files) {
  dims_list <- lapply(files, function(f) {
    h <- neuroim2::read_header(f)
    d <- dim(h)  # Clean usage with dim() method
    list(spatial = d[1:3], time = d[4])
  })
  
  # Aggregate dimensions
  list(
    spatial = dims_list[[1]]$spatial,
    time = sum(sapply(dims_list, function(d) d$time))
  )
}

# Method 2: Using read_vec (old approach)
read_vec_approach <- function(files) {
  dims_list <- lapply(files, function(f) {
    v <- neuroim2::read_vec(f)
    d <- dim(v)
    list(spatial = d[1:3], time = d[4])
  })
  
  # Aggregate dimensions
  list(
    spatial = dims_list[[1]]$spatial,
    time = sum(sapply(dims_list, function(d) d$time))
  )
}

# Test with single file
cat("Single file performance:\n")
single_file_bench <- microbenchmark(
  read_header = read_header_approach(test_file),
  read_vec = read_vec_approach(test_file),
  times = 20
)
print(single_file_bench)

# Test with multiple files (simulating multiple runs)
cat("\n\nMultiple files (3 runs) performance:\n")
multi_files <- rep(test_file, 3)
multi_file_bench <- microbenchmark(
  read_header = read_header_approach(multi_files),
  read_vec = read_vec_approach(multi_files),
  times = 10
)
print(multi_file_bench)

# Show the improvement
cat("\n\nPerformance improvement:\n")
single_median_header <- median(single_file_bench[single_file_bench$expr == "read_header", "time"])
single_median_vec <- median(single_file_bench[single_file_bench$expr == "read_vec", "time"])
single_speedup <- single_median_vec / single_median_header

multi_median_header <- median(multi_file_bench[multi_file_bench$expr == "read_header", "time"])
multi_median_vec <- median(multi_file_bench[multi_file_bench$expr == "read_vec", "time"])
multi_speedup <- multi_median_vec / multi_median_header

cat(sprintf("Single file: %.1fx faster\n", single_speedup))
cat(sprintf("Multiple files: %.1fx faster\n", multi_speedup))

# Demonstrate usage in fmri_dataset
cat("\n\nUsing optimized backend in fmri_dataset:\n")
cat("=========================================\n")

# Create dataset - this now uses read_header internally
backend <- nifti_backend(
  source = multi_files,
  mask_source = test_file,
  preload = FALSE
)

# Getting dimensions is now fast
system.time({
  dims <- backend_get_dims(backend)
})

cat("\nDimensions extracted efficiently:\n")
cat(sprintf("  Spatial: %s\n", paste(dims$spatial, collapse = " x ")))
cat(sprintf("  Time: %d volumes\n", dims$time))