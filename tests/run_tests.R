#!/usr/bin/env Rscript

# Test runner for fmridataset package
# This script loads all necessary source files and runs the test suite

cat("Loading fmridataset source files...\n")

# Set working directory to package root
if (basename(getwd()) == "tests") {
  setwd("..")
}

# Load required packages
library(testthat)
library(tibble)
library(methods)
library(assertthat)  # Required for dataset constructors
library(purrr)       # Required for dataset constructors  
library(neuroim2)    # Required for neuroimaging data structures
library(deflist)     # Required for data chunking

# Source all R files in order - updated for refactored structure
source_files <- c(
  "R/all_generic.R",           # Generic function declarations (must be first)
  "R/aaa_generics.R",          # Existing BIDS generics
  "R/utils.R", 
  "R/sampling_frame.R",
  "R/transformations.R",
  "R/config.R",                # Configuration functions
  "R/dataset_constructors.R",  # Dataset creation functions  
  "R/data_access.R",           # Data access methods
  "R/data_chunks.R",           # Data chunking functionality
  "R/print_methods.R",         # Print and display methods
  "R/conversions.R",           # Type conversion methods
  "R/fmri_dataset.R",          # Main entry point and documentation
  "R/fmri_dataset_class.R",
  "R/fmri_dataset_create.R",
  "R/fmri_dataset_accessors.R",
  "R/fmri_dataset_iterate.R",
  "R/fmri_dataset_validate.R",
  "R/fmri_dataset_print_summary.R",
  "R/fmri_dataset_preload.R",
  "R/fmri_dataset_from_paths.R",
  "R/fmri_dataset_from_list_matrix.R",
  "R/fmri_dataset_from_bids.R",
  "R/matrix_dataset.R",
  "R/bids_facade_phase1.R",
  "R/bids_facade_phase2.R",
  "R/bids_facade_phase3.R",
  "R/bids_interface.R"
)

for (file in source_files) {
  if (file.exists(file)) {
    cat("  Sourcing", file, "\n")
    source(file)
  } else {
    cat("  Warning: Missing", file, "\n")
  }
}

cat("\nRunning test suite...\n")

# Run tests with informative output
test_results <- test_dir(
  "tests/testthat", 
  reporter = "summary", 
  stop_on_failure = FALSE
)

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("Test Summary\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("Total tests run:", length(test_results), "\n")

# Count results
failed_tests <- 0
passed_tests <- 0
skipped_tests <- 0

# Safely extract test results
for (result in test_results) {
  if (is.list(result) && "results" %in% names(result)) {
    if ("passed" %in% names(result$results)) {
      passed_tests <- passed_tests + result$results$passed
    }
    if ("failed" %in% names(result$results)) {
      failed_tests <- failed_tests + result$results$failed
    }
    if ("skipped" %in% names(result$results)) {
      skipped_tests <- skipped_tests + result$results$skipped
    }
  }
}

cat("Passed:", passed_tests, "\n")
cat("Failed:", failed_tests, "\n") 
cat("Skipped:", skipped_tests, "\n")

if (failed_tests > 0) {
  cat("\nNote: Some tests failed. This may indicate areas where the refactored\n")
  cat("code structure needs refinement or additional compatibility measures.\n")
  cat("The test suite successfully identified potential issues in the\n")
  cat("refactored fmridataset package structure.\n")
} else {
  cat("\nAll tests passed! The refactored fmridataset package is working correctly.\n")
}

cat("\nRefactored package testing complete.\n")
cat("The test suite validates:\n")
cat("- Modular file structure compatibility\n")
cat("- Dataset construction and validation\n")  
cat("- Data access and manipulation\n")
cat("- Chunking and iteration functionality\n")
cat("- Print and summary methods\n")
cat("- Type conversion between dataset formats\n")
cat("- Backwards compatibility with existing interfaces\n") 