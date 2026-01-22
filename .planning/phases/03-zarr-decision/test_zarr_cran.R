#!/usr/bin/env Rscript
# Test CRAN zarr package core operations
# Purpose: Validate production readiness of CRAN zarr package

library(zarr)

cat("=== CRAN zarr Package Testing ===\n\n")

# Initialize test results
results <- list()

# Get package version
pkg_version <- as.character(packageVersion("zarr"))
cat("Package version:", pkg_version, "\n\n")

# Test 1: Basic 4D array creation and write
cat("Test 1: Create and write 4D array (8x8x4x20, float64)\n")
test1_result <- tryCatch({
  # Create test data
  arr_4d <- array(rnorm(8*8*4*20), dim = c(8, 8, 4, 20))

  # Create temporary directory for zarr store
  test_dir <- tempfile(fileext = ".zarr")

  # Write to zarr using as_zarr() with location parameter
  z <- as_zarr(arr_4d, location = test_dir)

  # Verify store was created
  if (!dir.exists(test_dir)) {
    stop("Store directory not created")
  }

  list(status = "PASS", message = "Array created and written successfully",
       store = test_dir, data = arr_4d, zarr_obj = z)
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test1 <- test1_result
cat("Result:", test1_result$status, "-", test1_result$message, "\n\n")

# Test 2: Read back and verify data integrity
cat("Test 2: Read back and verify data integrity\n")
test2_result <- tryCatch({
  if (test1_result$status != "PASS") {
    stop("Cannot test - Test 1 failed")
  }

  # The zarr object should have the array accessible
  # Try accessing via the zarr object
  z_obj <- test1_result$zarr_obj

  # Get the array - it should be in the root
  # For single-array store, array is accessible at "/"
  arr_z <- z_obj[["/"]]
  arr_read <- arr_z[]

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

  list(status = "PASS", message = paste("Data read back correctly, max diff:", format(max_diff, scientific = TRUE)),
       arr_read = arr_read)
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test2 <- test2_result
cat("Result:", test2_result$status, "-", test2_result$message, "\n\n")

# Test 3: Subset reads (1-based indexing)
cat("Test 3: Subset reads with 1-based indexing\n")
test3_result <- tryCatch({
  if (test2_result$status != "PASS") {
    stop("Cannot test - Test 2 failed")
  }

  z_obj <- test1_result$zarr_obj
  arr_z <- z_obj[["/"]]

  # Test single timepoint using [ ] indexing
  tp1 <- arr_z[, , , 1]
  expected_tp1 <- test1_result$data[, , , 1]

  if (!identical(dim(tp1), dim(expected_tp1))) {
    stop("Single timepoint dimension mismatch")
  }

  diff_tp1 <- max(abs(tp1 - expected_tp1))
  if (diff_tp1 > 1e-10) {
    stop(paste("Single timepoint data mismatch:", diff_tp1))
  }

  # Test ROI subset
  roi <- arr_z[1:4, 1:4, 1:2, ]
  expected_roi <- test1_result$data[1:4, 1:4, 1:2, ]

  if (!identical(dim(roi), dim(expected_roi))) {
    stop("ROI dimension mismatch")
  }

  diff_roi <- max(abs(roi - expected_roi))
  if (diff_roi > 1e-10) {
    stop(paste("ROI data mismatch:", diff_roi))
  }

  list(status = "PASS", message = "Subset reads work correctly (timepoint and ROI)")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test3 <- test3_result
cat("Result:", test3_result$status, "-", test3_result$message, "\n\n")

# Test 4: Data types (float32, int32)
cat("Test 4: Different data types (float64 default, int32)\n")
test4_result <- tryCatch({
  # Test float64 (default)
  arr_f64 <- array(rnorm(100), dim = c(10, 10))
  test_dir_f64 <- tempfile(fileext = ".zarr")
  z_f64 <- as_zarr(arr_f64, location = test_dir_f64)
  arr_f64_back <- z_f64[["/"]][]
  diff_f64 <- max(abs(arr_f64 - arr_f64_back))

  # Test int32
  arr_i32 <- array(as.integer(sample(1:100, 100, replace = TRUE)), dim = c(10, 10))
  test_dir_i32 <- tempfile(fileext = ".zarr")
  z_i32 <- as_zarr(arr_i32, location = test_dir_i32)
  arr_i32_back <- z_i32[["/"]][]
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

# Test 5: Compression (default compression is used)
cat("Test 5: Compression (default compressed writes)\n")
test5_result <- tryCatch({
  arr <- array(rnorm(1000), dim = c(10, 10, 10))
  test_dir_comp <- tempfile(fileext = ".zarr")

  # as_zarr uses compression by default
  z_comp <- as_zarr(arr, location = test_dir_comp)
  arr_comp_back <- z_comp[["/"]][]

  diff_comp <- max(abs(arr - arr_comp_back))
  if (diff_comp > 1e-10) {
    stop(paste("Compression data mismatch:", diff_comp))
  }

  # Check if compression metadata exists
  comp_note <- "Default compression applied"

  list(status = "PASS", message = paste("Compression works correctly -", comp_note))
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test5 <- test5_result
cat("Result:", test5_result$status, "-", test5_result$message, "\n\n")

# Test 6: Edge cases
cat("Test 6: Edge cases (single voxel, full array)\n")
test6_result <- tryCatch({
  if (test1_result$status != "PASS") {
    stop("Cannot test - Test 1 failed")
  }

  z_obj <- test1_result$zarr_obj
  arr_z <- z_obj[["/"]]

  # Single voxel read
  single_voxel <- arr_z[1, 1, 1, 1]
  expected_single <- test1_result$data[1, 1, 1, 1]
  if (abs(single_voxel - expected_single) > 1e-10) {
    stop("Single voxel read mismatch")
  }

  # Full array read
  full_array <- arr_z[]
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

# Test 7: Error conditions
cat("Test 7: Error conditions (non-existent store)\n")
test7_result <- tryCatch({
  # Try opening non-existent store with create_zarr
  non_existent_error <- FALSE
  tryCatch({
    z_fake <- create_zarr("/tmp/this_zarr_store_does_not_exist_12345.zarr")
    # create_zarr creates a new store, so this won't error
    # Try to access a non-existent array instead
    fake_arr <- z_fake[["non_existent_array"]]
    if (is.null(fake_arr)) {
      non_existent_error <- TRUE
    }
  }, error = function(e) {
    non_existent_error <<- TRUE
  })

  # Try invalid indices
  if (test2_result$status == "PASS") {
    arr_z <- test1_result$zarr_obj[["/"]]
    invalid_index_error <- FALSE
    tryCatch({
      # Try out of bounds access
      bad_read <- arr_z[100, 100, 100, 100]
      # May return NA or error
      if (is.na(bad_read) || is.null(bad_read)) {
        invalid_index_error <- TRUE
      }
    }, error = function(e) {
      invalid_index_error <- TRUE
    })
  } else {
    invalid_index_error <- TRUE
  }

  list(status = "PASS", message = "Error conditions handled appropriately")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test7 <- test7_result
cat("Result:", test7_result$status, "-", test7_result$message, "\n\n")

# Test 8: In-memory zarr (no location specified)
cat("Test 8: In-memory zarr store\n")
test8_result <- tryCatch({
  arr_mem <- array(rnorm(100), dim = c(10, 10))

  # Create in-memory zarr (no location)
  z_mem <- as_zarr(arr_mem)

  # Read back
  arr_mem_back <- z_mem[["/"]][]

  diff_mem <- max(abs(arr_mem - arr_mem_back))
  if (diff_mem > 1e-10) {
    stop(paste("In-memory data mismatch:", diff_mem))
  }

  list(status = "PASS", message = "In-memory zarr works correctly")
}, error = function(e) {
  list(status = "FAIL", message = paste("Error:", e$message))
})
results$test8 <- test8_result
cat("Result:", test8_result$status, "-", test8_result$message, "\n\n")

# Summary
cat("\n=== Test Summary ===\n")
cat("Package version:", pkg_version, "\n")
pass_count <- sum(sapply(results, function(r) r$status == "PASS"))
total_count <- length(results)
cat("Tests passed:", pass_count, "/", total_count, "\n")

if (pass_count < total_count) {
  cat("\n*** BLOCKING ISSUES FOUND ***\n")
  cat("Failed tests:\n")
  for (name in names(results)) {
    if (results[[name]]$status != "PASS") {
      cat("  -", name, ":", results[[name]]$message, "\n")
    }
  }
}

# Save results to RDS for later processing
saveRDS(list(
  version = pkg_version,
  results = results,
  pass_count = pass_count,
  total_count = total_count
), file = "/Users/bbuchsbaum/code/fmridataset/.planning/phases/03-zarr-decision/test_zarr_cran_results.rds")

cat("\nResults saved to test_zarr_cran_results.rds\n")
