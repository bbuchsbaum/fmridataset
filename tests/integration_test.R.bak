#!/usr/bin/env Rscript

# Integration test for fmridataset package
# Demonstrates core functionality working together

cat(paste(rep("=", 60), collapse=""), "\n")
cat("fmridataset Package Integration Test\n")
cat(paste(rep("=", 60), collapse=""), "\n")

# Load the package
suppressMessages({
  # Try to load the installed package first, fallback to development mode
  if (!require("fmridataset", quietly = TRUE)) {
    # Development mode - try to load source files directly
    if (file.exists("R/all_generic.R")) {
      source("R/all_generic.R")
      source("R/sampling_frame.R") 
      source("R/dataset_constructors.R")
      source("R/data_access.R")
      source("R/data_chunks.R")
      source("R/conversions.R")
      source("R/print_methods.R")
    } else {
      stop("Cannot load fmridataset package or source files")
    }
  }
  library(tibble, quietly = TRUE)
  library(methods, quietly = TRUE)
})

cat("\n1. Testing sampling_frame creation...\n")
sf <- sampling_frame(run_length = c(100, 80), TR = 2.0)
cat("   ✓ Created sampling frame with", n_runs(sf), "runs\n")
cat("   ✓ Total timepoints:", n_timepoints(sf), "\n")

cat("\n2. Testing matrix_dataset creation...\n")
set.seed(123)
test_matrix <- matrix(rnorm(1800), nrow = 180, ncol = 10)  # 180 timepoints, 10 voxels

dataset <- matrix_dataset(
  datamat = test_matrix,
  TR = 2.0,
  run_length = c(100, 80)
)
cat("   ✓ Created", class(dataset)[1], "dataset\n")
cat("   ✓ Dataset has", ncol(dataset$datamat), "voxels\n")
cat("   ✓ Dataset has", n_timepoints(dataset$sampling_frame), "timepoints\n")

cat("\n3. Testing data access...\n")
data_matrix <- get_data_matrix(dataset)
cat("   ✓ Retrieved data matrix with dimensions:", nrow(data_matrix), "×", ncol(data_matrix), "\n")

# Test run-specific access
run1_data <- get_data_matrix(dataset, run_id = 1)
cat("   ✓ Retrieved run 1 data:", nrow(run1_data), "timepoints\n")

cat("\n4. Testing with events...\n")
events <- data.frame(
  onset = c(10, 50, 90, 130),
  duration = c(2, 2, 2, 2),
  trial_type = c("A", "B", "A", "B")
)

dataset_with_events <- matrix_dataset(
  datamat = test_matrix,
  TR = 2.0,
  run_length = c(100, 80),
  event_table = events
)

cat("   ✓ Created dataset with", nrow(dataset_with_events$event_table), "events\n")

cat("\n5. Testing data chunking...\n")
chunks <- data_chunks(dataset, nchunks = 3)
chunk1 <- chunks$nextElem()
cat("   ✓ Created chunks, first chunk has", ncol(chunk1$data), "voxels\n")

# Test runwise chunking
run_chunks <- data_chunks(dataset, runwise = TRUE)
cat("   ✓ Created", run_chunks$nchunks, "run-wise chunks\n")

cat("\n6. Testing print method...\n")
print(dataset_with_events)

cat("\n7. Testing conversions...\n")
matrix_version <- as.matrix_dataset(dataset)
cat("   ✓ Converted to matrix dataset with", ncol(matrix_version$datamat), "voxels\n")

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("Integration test completed successfully!\n")
cat("Core fmridataset functionality is working.\n")
cat(paste(rep("=", 60), collapse=""), "\n") 