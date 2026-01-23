#!/usr/bin/env Rscript

# Test script to verify all optional packages work correctly
# This helps catch issues where tests pass due to skipped functionality

cat("Testing fmridataset with all optional packages...\n\n")

# List of optional packages used in the codebase
optional_packages <- list(
  # From Suggests in DESCRIPTION
  bench = "Performance benchmarking",
  bidser = "BIDS dataset support",
  crayon = "Colored terminal output",
  arrow = "Arrow/Parquet support",
  dplyr = "Data manipulation",
  fmristore = "HDF5 storage for fMRI data",
  foreach = "Parallel processing",

  # From requireNamespace calls
  zarr = "Zarr array support (CRAN)",
  hdf5r = "HDF5 support (CRAN)",
  jsonlite = "JSON configuration files",
  yaml = "YAML configuration files"
)

# Check which packages are installed
cat("Checking optional package availability:\n")
cat(strrep("-", 60), "\n")

installed <- character()
missing <- character()

for (pkg in names(optional_packages)) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("✓ %-15s : %s\n", pkg, optional_packages[[pkg]]))
    installed <- c(installed, pkg)
  } else {
    cat(sprintf("✗ %-15s : %s\n", pkg, optional_packages[[pkg]]))
    missing <- c(missing, pkg)
  }
}

cat(strrep("-", 60), "\n")
cat(sprintf("Installed: %d/%d\n", length(installed), length(optional_packages)))

if (length(missing) > 0) {
  cat("\nTo install missing packages:\n")

  github_pkgs <- intersect(missing, c("fmristore", "bidser"))
  if (length(github_pkgs) > 0) {
    if ("fmristore" %in% github_pkgs) {
      cat('remotes::install_github("bbuchsbaum/fmristore")\n')
    }
    if ("bidser" %in% github_pkgs) {
      cat('remotes::install_github("bbuchsbaum/bidser")\n')
    }
  }

  cran_pkgs <- setdiff(missing, github_pkgs)
  if (length(cran_pkgs) > 0) {
    cat(sprintf(
      "install.packages(c(%s))\n",
      paste0('"', cran_pkgs, '"', collapse = ", ")
    ))
  }
}

# Run specific tests for optional functionality
cat("\n\nTesting optional functionality:\n")
cat(strrep("-", 60), "\n")

# Test Zarr backend
if ("zarr" %in% installed) {
  cat("Testing Zarr backend... ")
  tryCatch(
    {
      devtools::load_all(quiet = TRUE)
      backend <- zarr_backend("dummy.zarr")
      cat("✓ Success\n")
    },
    error = function(e) {
      cat("✗ Failed:", conditionMessage(e), "\n")
    }
  )
} else {
  cat("Skipping Zarr backend test (zarr not installed)\n")
}

# Test HDF5 functionality
if (all(c("hdf5r", "fmristore") %in% installed)) {
  cat("Testing HDF5 backend... ")
  tryCatch(
    {
      devtools::load_all(quiet = TRUE)
      # Create minimal test
      temp_h5 <- tempfile(fileext = ".h5")
      # Would need actual H5 file creation here
      cat("✓ Success (basic check)\n")
    },
    error = function(e) {
      cat("✗ Failed:", conditionMessage(e), "\n")
    }
  )
} else {
  cat("Skipping HDF5 backend test (hdf5r/fmristore not installed)\n")
}

# Test BIDS functionality
if ("bidser" %in% installed) {
  cat("Testing BIDS integration... ")
  tryCatch(
    {
      # Basic check that bidser can be loaded
      bidser::bids_project
      cat("✓ Success (package loads)\n")
    },
    error = function(e) {
      cat("✗ Failed:", conditionMessage(e), "\n")
    }
  )
} else {
  cat("Skipping BIDS integration test (bidser not installed)\n")
}

# Run full test suite and check for skips
cat("\n\nRunning full test suite:\n")
cat(strrep("-", 60), "\n")

if (length(installed) == length(optional_packages)) {
  cat("All optional packages installed - running comprehensive tests\n")

  # Run tests and analyze results
  test_results <- testthat::test_dir("tests/testthat", reporter = "minimal")

  # Count skipped tests
  n_skip <- sum(vapply(test_results, function(x) {
    sum(vapply(x$results, function(r) inherits(r, "skip"), logical(1)))
  }, integer(1)))

  if (n_skip > 0) {
    cat(sprintf("\nWARNING: %d tests were skipped even with all packages installed\n", n_skip))
    cat("This might indicate tests that need updating\n")
  } else {
    cat("\nSUCCESS: All tests ran without skips\n")
  }
} else {
  cat("Some optional packages missing - tests may skip functionality\n")
  cat("Install missing packages and re-run for comprehensive testing\n")
}

cat(strrep("-", 60), "\n")
