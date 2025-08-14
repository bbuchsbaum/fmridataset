#!/usr/bin/env Rscript

# Script to generate golden test data for fmridataset package
# Run this script from the package root directory

# Check if we're in the package root
if (!file.exists("DESCRIPTION")) {
  stop("This script must be run from the fmridataset package root directory")
}

# Load the package
devtools::load_all()

# Source the helper file directly
source("tests/testthat/helper-golden.R")

# Create golden data directory if it doesn't exist
golden_dir <- "tests/testthat/golden"
if (!dir.exists(golden_dir)) {
  dir.create(golden_dir, recursive = TRUE)
  cat("Created golden data directory:", golden_dir, "\n")
}

# Generate all golden data
cat("Generating golden test data...\n")
cat("This will create reference data files in:", golden_dir, "\n\n")

tryCatch({
  # Generate the data
  generate_all_golden_data()
  
  # List generated files
  files <- list.files(golden_dir, pattern = "\\.rds$", full.names = FALSE)
  cat("\nGenerated files:\n")
  for (f in files) {
    size <- file.size(file.path(golden_dir, f))
    cat(sprintf("  - %s (%.1f KB)\n", f, size / 1024))
  }
  
  cat("\nGolden test data generation complete!\n")
  cat("\nTo use these tests:\n")
  cat("1. Run: devtools::test(filter = 'golden')\n")
  cat("2. Or run individual test files:\n")
  cat("   - testthat::test_file('tests/testthat/test-golden-datasets.R')\n")
  cat("   - testthat::test_file('tests/testthat/test-golden-fmriseries.R')\n")
  cat("   - testthat::test_file('tests/testthat/test-golden-backends.R')\n")
  cat("   - testthat::test_file('tests/testthat/test-golden-sampling-frame.R')\n")
  cat("   - testthat::test_file('tests/testthat/test-golden-snapshots.R')\n")
  
}, error = function(e) {
  cat("\nError generating golden test data:\n")
  cat(e$message, "\n")
  quit(status = 1)
})

cat("\nNote: Snapshot tests will create snapshot files on first run.\n")
cat("Review the snapshots in tests/testthat/_snaps/ and commit them to git.\n")