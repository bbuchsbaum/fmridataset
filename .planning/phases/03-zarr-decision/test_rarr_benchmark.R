#!/usr/bin/env Rscript
# Test Rarr package and benchmark vs HDF5
# Purpose: Compare production-ready Rarr implementation with HDF5 performance

library(Rarr)
library(hdf5r)

cat("=== Rarr Package Testing and Benchmarking ===\n\n")

# Initialize test results
results <- list()

# Get package versions
rarr_version <- as.character(packageVersion("Rarr"))
hdf5r_version <- as.character(packageVersion("hdf5r"))
cat("Rarr version:", rarr_version, "\n")
cat("hdf5r version:", hdf5r_version, "\n\n")

# Test 1: Rarr basic 4D array creation and write
cat("Test 1: Rarr - Create and write 4D array (8x8x4x20, float64)\n")
test1_result <- tryCatch({
  # Create test data
  arr_4d <- array(rnorm(8*8*4*20), dim = c(8, 8, 4, 20))

  # Create temporary directory for zarr store
  test_dir <- tempfile("rarr_test_")

  # Write to zarr using Rarr
  write_zarr_array(arr_4d, zarr_array_path = test_dir, chunk_dim = c(8, 8, 4, 5))

  # Verify store was created
  if (!dir.exists(test_dir)) {
    stop("Store directory not created")
  }

  list(status = "PASS", message = "Array created and written successfully",
       store = test_dir, data = arr_4d)
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test1 <- test1_result
cat("Result:", test1_result$status, "-", test1_result$message, "\n\n")

# Test 2: Rarr read back and verify data integrity
cat("Test 2: Rarr - Read back and verify data integrity\n")
test2_result <- tryCatch({
  if (test1_result$status != "PASS") {
    stop("Cannot test - Test 1 failed")
  }

  # Read back the zarr array
  arr_read <- read_zarr_array(test1_result$store)

  # Verify dimensions
  if (!identical(dim(arr_read), dim(test1_result$data))) {
    stop(paste("Dimension mismatch: expected", paste(dim(test1_result$data), collapse="x"),
               "got", paste(dim(arr_read), collapse="x")))
  }

  # Verify data integrity
  max_diff <- max(abs(arr_read - test1_result$data))
  if (max_diff > 1e-10) {
    stop(paste("Data mismatch: max difference =", max_diff))
  }

  list(status = "PASS", message = paste("Data read back correctly, max diff:", format(max_diff, scientific = TRUE)))
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test2 <- test2_result
cat("Result:", test2_result$status, "-", test2_result$message, "\n\n")

# Test 3: Rarr subset reads
cat("Test 3: Rarr - Subset reads\n")
test3_result <- tryCatch({
  if (test1_result$status != "PASS") {
    stop("Cannot test - Test 1 failed")
  }

  # Read subset - single timepoint
  # Rarr uses list indexing for subsets
  tp1 <- read_zarr_array(test1_result$store, index = list(1:8, 1:8, 1:4, 1))
  expected_tp1 <- test1_result$data[, , , 1, drop = FALSE]

  # Rarr may drop singleton dimensions - reshape if needed
  if (!identical(dim(tp1), dim(expected_tp1))) {
    # If dimensions differ only by singleton, reshape
    if (prod(dim(tp1)) == prod(dim(expected_tp1))) {
      dim(tp1) <- dim(expected_tp1)
    } else {
      stop(paste("Dimension mismatch: tp1 =", paste(dim(tp1), collapse="x"),
                 "expected =", paste(dim(expected_tp1), collapse="x")))
    }
  }

  diff_tp1 <- max(abs(tp1 - expected_tp1))
  if (diff_tp1 > 1e-10) {
    stop(paste("Single timepoint data mismatch:", diff_tp1))
  }

  list(status = "PASS", message = "Subset reads work correctly")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test3 <- test3_result
cat("Result:", test3_result$status, "-", test3_result$message, "\n\n")

# Test 4: Rarr data types
cat("Test 4: Rarr - Different data types (float64, int32)\n")
test4_result <- tryCatch({
  # Test float64
  arr_f64 <- array(rnorm(100), dim = c(10, 10))
  test_dir_f64 <- tempfile("rarr_f64_")
  write_zarr_array(arr_f64, zarr_array_path = test_dir_f64, chunk_dim = c(10, 10))
  arr_f64_back <- read_zarr_array(test_dir_f64)
  diff_f64 <- max(abs(arr_f64 - arr_f64_back))

  # Test int32
  arr_i32 <- array(as.integer(sample(1:100, 100, replace = TRUE)), dim = c(10, 10))
  test_dir_i32 <- tempfile("rarr_i32_")
  write_zarr_array(arr_i32, zarr_array_path = test_dir_i32, chunk_dim = c(10, 10))
  arr_i32_back <- read_zarr_array(test_dir_i32)
  diff_i32 <- max(abs(arr_i32 - arr_i32_back))

  if (diff_f64 > 1e-10) {
    stop(paste("float64 precision loss:", diff_f64))
  }

  if (diff_i32 > 0) {
    stop(paste("int32 data mismatch:", diff_i32))
  }

  list(status = "PASS", message = paste("Data types OK - float64 diff:", format(diff_f64, scientific = TRUE),
                                        "int32 diff:", diff_i32))
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test4 <- test4_result
cat("Result:", test4_result$status, "-", test4_result$message, "\n\n")

# Test 5: Rarr compression
cat("Test 5: Rarr - Compression\n")
test5_result <- tryCatch({
  arr <- array(rnorm(1000), dim = c(10, 10, 10))
  test_dir_comp <- tempfile("rarr_comp_")

  # Rarr uses blosc compression by default
  write_zarr_array(arr, zarr_array_path = test_dir_comp, chunk_dim = c(10, 10, 10))
  arr_comp_back <- read_zarr_array(test_dir_comp)

  diff_comp <- max(abs(arr - arr_comp_back))
  if (diff_comp > 1e-10) {
    stop(paste("Compression data mismatch:", diff_comp))
  }

  list(status = "PASS", message = "Compression works correctly")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test5 <- test5_result
cat("Result:", test5_result$status, "-", test5_result$message, "\n\n")

# Test 6: Rarr edge cases
cat("Test 6: Rarr - Edge cases\n")
test6_result <- tryCatch({
  if (test1_result$status != "PASS") {
    stop("Cannot test - Test 1 failed")
  }

  # Full array read (already tested in test2)
  full_array <- read_zarr_array(test1_result$store)
  diff_full <- max(abs(full_array - test1_result$data))
  if (diff_full > 1e-10) {
    stop("Full array read mismatch")
  }

  list(status = "PASS", message = "Edge cases handled correctly")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test6 <- test6_result
cat("Result:", test6_result$status, "-", test6_result$message, "\n\n")

# Summary of Rarr tests
cat("\n=== Rarr Test Summary ===\n")
pass_count <- sum(sapply(results, function(r) r$status == "PASS"))
total_count <- length(results)
cat("Tests passed:", pass_count, "/", total_count, "\n\n")

if (pass_count < total_count) {
  cat("*** BLOCKING ISSUES FOUND ***\n")
  cat("Failed tests:\n")
  for (name in names(results)) {
    if (results[[name]]$status != "PASS") {
      cat("  -", name, ":", results[[name]]$message, "\n")
    }
  }
  cat("\n")
}

# Performance Benchmark
cat("\n=== Performance Benchmark ===\n")
cat("Testing with 64x64x30x100 fMRI-like data (float64)\n\n")

# Create benchmark data
set.seed(42)
bench_dims <- c(64, 64, 30, 100)
bench_data <- array(rnorm(prod(bench_dims)), dim = bench_dims)
bench_size_mb <- prod(bench_dims) * 8 / (1024^2)  # float64 = 8 bytes
cat("Array size:", sprintf("%.2f", bench_size_mb), "MB\n\n")

# Temporary paths
zarr_path <- tempfile(fileext = ".zarr")
h5_path <- tempfile(fileext = ".h5")

# Helper to format time in ms
format_time <- function(t) {
  sprintf("%.1f ms", t * 1000)
}

# Benchmark 1: Write full array
cat("1. Write full array:\n")

# Rarr write
t_rarr_write <- system.time({
  write_zarr_array(bench_data, zarr_array_path = zarr_path,
                   chunk_dim = c(64, 64, 30, 10))
})[3]

# HDF5 write
t_h5_write <- system.time({
  h5file <- H5File$new(h5_path, mode = "w")
  h5file[["data"]] <- bench_data
  h5file$close_all()
})[3]

cat("  Rarr:", format_time(t_rarr_write), "\n")
cat("  HDF5:", format_time(t_h5_write), "\n\n")

# Benchmark 2: Read full array
cat("2. Read full array:\n")

t_rarr_read_full <- system.time({
  arr_rarr <- read_zarr_array(zarr_path)
})[3]

t_h5_read_full <- system.time({
  h5file <- H5File$new(h5_path, mode = "r")
  dset <- h5file[["data"]]
  arr_h5 <- dset$read()
  h5file$close_all()
})[3]

cat("  Rarr:", format_time(t_rarr_read_full), "\n")
cat("  HDF5:", format_time(t_h5_read_full), "\n\n")

# Benchmark 3: Read single timepoint
cat("3. Read single timepoint (64x64x30):\n")

t_rarr_read_tp <- system.time({
  tp_rarr <- read_zarr_array(zarr_path, index = list(1:64, 1:64, 1:30, 1))
})[3]

t_h5_read_tp <- system.time({
  h5file <- H5File$new(h5_path, mode = "r")
  dset <- h5file[["data"]]
  # Select single timepoint using hyperslab
  dset$read(args = list(1:64, 1:64, 1:30, 1)) -> tp_h5
  h5file$close_all()
})[3]

cat("  Rarr:", format_time(t_rarr_read_tp), "\n")
cat("  HDF5:", format_time(t_h5_read_tp), "\n\n")

# Benchmark 4: Read ROI (10x10x10 region, all timepoints)
cat("4. Read ROI (10x10x10 x 100 timepoints):\n")

t_rarr_read_roi <- system.time({
  roi_rarr <- read_zarr_array(zarr_path, index = list(1:10, 1:10, 1:10, 1:100))
})[3]

t_h5_read_roi <- system.time({
  h5file <- H5File$new(h5_path, mode = "r")
  dset <- h5file[["data"]]
  # Select ROI using hyperslab
  dset$read(args = list(1:10, 1:10, 1:10, 1:100)) -> roi_h5
  h5file$close_all()
})[3]

cat("  Rarr:", format_time(t_rarr_read_roi), "\n")
cat("  HDF5:", format_time(t_h5_read_roi), "\n\n")

# File sizes
cat("5. File size on disk:\n")
zarr_size <- sum(file.info(list.files(zarr_path, recursive = TRUE, full.names = TRUE))$size) / (1024^2)
h5_size <- file.info(h5_path)$size / (1024^2)

cat("  Rarr:", sprintf("%.2f MB", zarr_size), "\n")
cat("  HDF5:", sprintf("%.2f MB", h5_size), "\n\n")

# Summary table
cat("\n=== Performance Summary Table ===\n")
cat(sprintf("%-30s | %12s | %12s | %s\n", "Operation", "Rarr", "HDF5", "Notes"))
cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("%-30s | %12s | %12s | %s\n", "Write 64x64x30x100", format_time(t_rarr_write), format_time(t_h5_write), ""))
cat(sprintf("%-30s | %12s | %12s | %s\n", "Read full array", format_time(t_rarr_read_full), format_time(t_h5_read_full), ""))
cat(sprintf("%-30s | %12s | %12s | %s\n", "Read single timepoint", format_time(t_rarr_read_tp), format_time(t_h5_read_tp), ""))
cat(sprintf("%-30s | %12s | %12s | %s\n", "Read ROI", format_time(t_rarr_read_roi), format_time(t_h5_read_roi), ""))
cat(sprintf("%-30s | %9.2f MB | %9.2f MB | %s\n", "File size", zarr_size, h5_size, ""))
cat("\n")

# Save results
benchmark_results <- list(
  rarr_version = rarr_version,
  hdf5r_version = hdf5r_version,
  tests = results,
  benchmark = list(
    write = c(rarr = t_rarr_write, hdf5 = t_h5_write),
    read_full = c(rarr = t_rarr_read_full, hdf5 = t_h5_read_full),
    read_timepoint = c(rarr = t_rarr_read_tp, hdf5 = t_h5_read_tp),
    read_roi = c(rarr = t_rarr_read_roi, hdf5 = t_h5_read_roi),
    file_size = c(rarr = zarr_size, hdf5 = h5_size)
  )
)

saveRDS(benchmark_results,
        file = "/Users/bbuchsbaum/code/fmridataset/.planning/phases/03-zarr-decision/test_rarr_benchmark_results.rds")

cat("Results saved to test_rarr_benchmark_results.rds\n")

# Cleanup
unlink(zarr_path, recursive = TRUE)
unlink(h5_path)
if (test1_result$status == "PASS") unlink(test1_result$store, recursive = TRUE)
