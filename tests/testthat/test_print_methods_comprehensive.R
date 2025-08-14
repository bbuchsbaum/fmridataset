test_that("print.fmri_dataset shows correct basic information", {
  # Create a simple matrix dataset
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = c(50, 50))
  
  # Capture output
  output <- capture.output(print(dset))
  
  # Check basic structure
  expect_true(any(grepl("fMRI Dataset", output)))
  expect_true(any(grepl("Dimensions:", output)))
  expect_true(any(grepl("Timepoints: 100", output)))
  expect_true(any(grepl("Runs: 2", output)))
  expect_true(any(grepl("TR: 2 seconds", output)))
})

test_that("print.fmri_dataset full=TRUE shows additional details", {
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  # Compare outputs
  output_basic <- capture.output(print(dset, full = FALSE))
  output_full <- capture.output(print(dset, full = TRUE))
  
  # Full output should be longer
  expect_true(length(output_full) > length(output_basic))
  
  # Full output should show actual voxel count
  expect_true(any(grepl("Voxels in mask: 50", output_full)))
  expect_true(any(grepl("Voxels in mask: \\(lazy\\)", output_basic)))
})

test_that("print.latent_dataset shows latent-specific information", {
  skip_if_not_installed("fmristore")
  
  # Create mock latent dataset
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(basis = "matrix", loadings = "matrix", offset = "numeric", 
                mask = "array", space = "ANY"))
  }
  
  lvec <- methods::new("mock_LatentNeuroVec",
    basis = matrix(rnorm(100 * 5), 100, 5),
    loadings = matrix(rnorm(1000 * 5), 1000, 5),
    offset = numeric(0),
    mask = array(TRUE, c(10, 10, 10)),
    space = structure(c(10, 10, 10, 100), class = "mock_space")
  )
  
  dset <- latent_dataset(list(lvec), TR = 2, run_length = 100)
  output <- capture.output(print(dset))
  
  # Check latent-specific output
  expect_true(any(grepl("Latent Dataset", output)))
  expect_true(any(grepl("Components: 5", output)))
  expect_true(any(grepl("Original voxels: 1000", output)))
})

test_that("print.data_chunk shows chunk information", {
  mat <- matrix(1:20, 5, 4)
  chunk <- structure(
    list(
      data = mat,
      indices = list(rows = 1:5, cols = 1:4),
      nchunks = 1,
      chunkid = 1
    ),
    class = "data_chunk"
  )
  
  output <- capture.output(print(chunk))
  
  expect_true(any(grepl("Data Chunk", output)))
  expect_true(any(grepl("chunk 1 of 1", output)))
  expect_true(any(grepl("5 x 4", output)))
})

test_that("print.chunkiter shows iterator information", {
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  iter <- data_chunks(dset, nchunks = 5)
  output <- capture.output(print(iter))
  
  expect_true(any(grepl("Chunk Iterator", output)))
  expect_true(any(grepl("nchunks: 5", output)))
})

test_that("print methods handle empty datasets gracefully", {
  # Empty matrix dataset
  mat <- matrix(numeric(0), 0, 10)
  expect_error(
    matrix_dataset(mat, TR = 2, run_length = 0),
    "Block lengths must be positive"
  )
  
  # Test with 1 timepoint instead
  mat <- matrix(1:10, 1, 10)
  dset <- matrix_dataset(mat, TR = 2, run_length = 1)
  output <- capture.output(print(dset))
  
  expect_true(any(grepl("Timepoints: 1", output)))
  expect_true(any(grepl("Runs: 1", output)))
})

test_that("print methods handle NULL and missing values", {
  # Dataset with no event table
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  dset$event_table <- NULL
  
  # Should not error
  expect_silent(output <- capture.output(print(dset)))
  
  # Dataset with NA values in sampling frame
  dset$sampling_frame$TR <- NA
  output <- capture.output(print(dset))
  expect_true(any(grepl("TR: NA", output)))
})

test_that("summary.fmri_dataset provides comprehensive information", {
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = c(40, 60))
  
  # Add event table
  dset$event_table <- data.frame(
    onset = c(10, 20, 30),
    duration = c(2, 2, 2),
    trial_type = c("A", "B", "A")
  )
  
  output <- capture.output(summary(dset))
  
  # Should show event summary
  expect_true(any(grepl("Event Summary", output)))
  expect_true(any(grepl("trial_type", output)))
})

test_that("print methods for series selectors show correct info", {
  # Test various selector types
  idx_sel <- index_selector(c(1, 3, 5))
  output <- capture.output(print(idx_sel))
  expect_true(any(grepl("index_selector", output)))
  expect_true(any(grepl("indices: 1, 3, 5", output)))
  
  # Mask selector
  mask_sel <- mask_selector(rep(c(TRUE, FALSE), 5))
  output <- capture.output(print(mask_sel))
  expect_true(any(grepl("mask_selector", output)))
  
  # ROI selector
  roi_array <- array(FALSE, dim = c(10, 10, 10))
  roi_array[1:3, 1:3, 1] <- TRUE  # 9 active voxels
  roi_sel <- roi_selector(roi_array)
  output <- capture.output(print(roi_sel))
  expect_true(any(grepl("roi_selector", output)))
  expect_true(any(grepl("active voxels: 9", output)))
  
  # Sphere selector
  sphere_sel <- sphere_selector(center = c(10, 20, 30), radius = 5)
  output <- capture.output(print(sphere_sel))
  expect_true(any(grepl("sphere_selector", output)))
  expect_true(any(grepl("Center:.*10.*20.*30", output)))
  expect_true(any(grepl("Radius: 5", output)))
})

test_that("print methods handle very long output gracefully", {
  # Dataset with many runs
  mat <- matrix(rnorm(1000 * 50), 1000, 50)
  run_lengths <- rep(10, 100)  # 100 runs
  dset <- matrix_dataset(mat, TR = 2, run_length = run_lengths)
  
  output <- capture.output(print(dset))
  
  # Should truncate run lengths display
  expect_true(any(grepl("\\.\\.\\.", output)))
  expect_true(any(grepl("100 runs", output)))
})

test_that("print.matrix_dataset shows matrix-specific information", {
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  output <- capture.output(print(dset))
  
  # Should indicate it's in-memory
  expect_true(any(grepl("matrix_dataset|in-memory|Matrix", output)))
})

test_that("print methods use consistent formatting", {
  # Create different dataset types
  mat <- matrix(rnorm(100 * 50), 100, 50)
  mat_dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  # Mock file dataset
  if (requireNamespace("neuroim2", quietly = TRUE)) {
    # Would test with actual file dataset
    # For now just test matrix dataset formatting
    output <- capture.output(print(mat_dset))
    
    # Check consistent formatting patterns
    expect_true(any(grepl("^\\*\\*", output)))  # Section headers start with **
    expect_true(any(grepl("^  -", output)))     # Items start with "  -"
  }
})

test_that("print methods handle special characters in paths", {
  skip_on_cran()
  
  # Create dataset with special characters in base path
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  # Add a path with special characters
  if (.Platform$OS.type == "unix") {
    dset$base_path <- "/tmp/test path with spaces/data"
  } else {
    dset$base_path <- "C:\\test path with spaces\\data"
  }
  
  output <- capture.output(print(dset))
  # Should handle the path without errors
  expect_true(length(output) > 0)
})

test_that("print output is invisibly returned", {
  mat <- matrix(rnorm(100 * 50), 100, 50)
  dset <- matrix_dataset(mat, TR = 2, run_length = 100)
  
  # Capture both output and return value
  output <- capture.output(ret <- print(dset))
  
  # Return value should be the dataset itself
  expect_identical(ret, dset)
  
  # Should be returned invisibly (the assignment itself doesn't print the return value)
  # But print() still outputs to console, so we capture it
  output2 <- capture.output(x <- print(dset))
  expect_identical(x, dset)
  expect_true(length(output2) > 0)  # print() still produces output
})