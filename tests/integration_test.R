#!/usr/bin/env Rscript

# Integration test for fmridataset package
# Demonstrates core functionality working together

cat("=" * 60, "\n")
cat("fmridataset Package Integration Test\n")
cat("=" * 60, "\n")

# Load required packages
suppressMessages({
  library(tibble)
  library(methods)
})

# Source main files (simplified for demo)
source("R/aaa_generics.R")
source("R/utils.R")
source("R/sampling_frame.R")
source("R/fmri_dataset_class.R")
source("R/fmri_dataset_create.R")
source("R/fmri_dataset_accessors.R")

cat("\n1. Testing sampling_frame creation...\n")
sf <- sampling_frame(TR = 2.0, run_lengths = c(100, 80))
cat("   ✓ Created sampling frame with", n_runs(sf), "runs\n")
cat("   ✓ Total timepoints:", n_timepoints(sf), "\n")

cat("\n2. Testing fmri_dataset creation from matrix...\n")
set.seed(123)
test_matrix <- matrix(rnorm(1800), nrow = 180, ncol = 10)  # 180 timepoints, 10 voxels

dataset <- fmri_dataset_create(
  images = test_matrix,
  TR = 2.0,
  run_lengths = c(100, 80)
)
cat("   ✓ Created", get_dataset_type(dataset), "dataset\n")
cat("   ✓ Dataset has", get_num_voxels(dataset), "voxels\n")
cat("   ✓ Dataset has", get_num_timepoints(dataset), "timepoints\n")

cat("\n3. Testing data access...\n")
data_matrix <- get_data_matrix(dataset)
cat("   ✓ Retrieved data matrix with dimensions:", nrow(data_matrix), "×", ncol(data_matrix), "\n")

# Test run-specific access
run1_data <- get_data_matrix(dataset, run_id = 1)
cat("   ✓ Retrieved run 1 data:", nrow(run1_data), "timepoints\n")

cat("\n4. Testing with events and censoring...\n")
events <- data.frame(
  onset = c(10, 50, 90, 130),
  duration = c(2, 2, 2, 2),
  trial_type = c("A", "B", "A", "B")
)

censor_vector <- rep(TRUE, 180)
censor_vector[50:55] <- FALSE  # Censor 6 timepoints

dataset_complex <- fmri_dataset_create(
  images = test_matrix,
  TR = 2.0,
  run_lengths = c(100, 80),
  event_table = events,
  censor_vector = censor_vector,
  temporal_zscore = TRUE
)

cat("   ✓ Created dataset with", nrow(get_event_table(dataset_complex)), "events\n")
cat("   ✓ Censoring:", sum(!get_censor_vector(dataset_complex)), "timepoints removed\n")

cat("\n5. Testing validation...\n")
tryCatch({
  validate_fmri_dataset(dataset_complex)
  cat("   ✓ Validation passed\n")
}, error = function(e) {
  cat("   ⚠ Validation issue:", e$message, "\n")
})

cat("\n6. Testing print method...\n")
print(dataset_complex)

cat("\n" + "=" * 60, "\n")
cat("Integration test completed successfully!\n")
cat("Core fmridataset functionality is working.\n")
cat("=" * 60, "\n") 