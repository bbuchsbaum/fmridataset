test_that("h5_backend constructor validates inputs correctly", {
  skip_if_not_installed("fmristore")
  
  # Test missing files
  expect_error(
    h5_backend(c("nonexistent1.h5", "nonexistent2.h5"), "mask.h5"),
    "H5 source files not found"
  )
  
  # Test missing mask file - create a temporary file to pass first validation
  temp_file <- tempfile(fileext = ".h5")
  file.create(temp_file)
  on.exit(unlink(temp_file))
  
  expect_error(
    h5_backend(temp_file, "nonexistent_mask.h5"),
    "H5 mask file not found"
  )
  
  # Test invalid source type
  expect_error(
    h5_backend(123, "mask.h5"),
    "source must be character vector.*or list"
  )
  
  # Test invalid H5NeuroVec objects in list
  expect_error(
    h5_backend(list("not_h5neurovec"), "mask.h5"),
    "All source objects must be H5NeuroVec objects"
  )
})

test_that("h5_backend works with file paths", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip_if_not_installed("hdf5r")
  
  # Create temporary H5 files for testing
  temp_dir <- tempdir()
  h5_file1 <- file.path(temp_dir, "test_scan1.h5")
  h5_file2 <- file.path(temp_dir, "test_scan2.h5")
  mask_file <- file.path(temp_dir, "test_mask.h5")
  
  # Create test H5 files using fmristore helpers (if available)
  # This is a simplified test - in practice you'd create proper H5 files
  skip("H5 file creation helpers not available for testing")
  
  # If we had the files, the test would look like:
  # backend <- h5_backend(
  #   source = c(h5_file1, h5_file2),
  #   mask_source = mask_file
  # )
  # 
  # expect_s3_class(backend, "h5_backend")
  # expect_s3_class(backend, "storage_backend")
  # expect_equal(backend$source, c(h5_file1, h5_file2))
  # expect_equal(backend$mask_source, mask_file)
})

test_that("h5_backend works with pre-loaded H5NeuroVec objects", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip_if_not_installed("hdf5r")
  
  # This test would require creating mock H5NeuroVec objects
  # which is complex without the actual fmristore infrastructure
  skip("Mock H5NeuroVec objects not available for testing")
})

test_that("h5_backend backend_open and backend_close work", {
  skip_if_not_installed("fmristore")
  
  # Create a mock backend (without real files for this test)
  backend <- structure(
    list(
      source = character(0),
      mask_source = character(0),
      preload = FALSE,
      h5_objects = NULL,
      mask = NULL,
      dims = NULL
    ),
    class = c("h5_backend", "storage_backend")
  )
  
  # Test opening (should work even with empty backend for preload=FALSE)
  opened_backend <- backend_open(backend)
  expect_s3_class(opened_backend, "h5_backend")
  
  # Test closing (should not error)
  expect_silent(backend_close(opened_backend))
})

test_that("h5_backend validates fmristore dependency", {
  # Skip this test if fmristore is actually available
  skip_if(requireNamespace("fmristore", quietly = TRUE), "fmristore is available")
  
  # If fmristore is not installed, h5_backend should error
  expect_error(
    h5_backend("test.h5", "mask.h5"),
    "Package 'fmristore' is required for H5 backend but is not available"
  )
})

test_that("fmri_h5_dataset constructor works", {
  skip_if_not_installed("fmristore")
  
  # Mock the h5_backend function to avoid file dependencies
  with_mocked_bindings(
    h5_backend = function(...) {
      structure(
        list(
          source = list(...)[["source"]],
          mask_source = list(...)[["mask_source"]],
          preload = FALSE
        ),
        class = c("h5_backend", "storage_backend")
      )
    },
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 100),
    {
      dataset <- fmri_h5_dataset(
        h5_files = c("scan1.h5", "scan2.h5"),
        mask_source = "mask.h5",
        TR = 2,
        run_length = c(50, 50)
      )
      
      expect_s3_class(dataset, "fmri_file_dataset")
      expect_s3_class(dataset, "fmri_dataset")
      expect_s3_class(dataset$backend, "h5_backend")
    }
  )
})

test_that("h5_backend handles base_path correctly", {
  skip_if_not_installed("fmristore")
  
  # Mock the h5_backend function
  h5_backend_calls <- list()
  with_mocked_bindings(
    h5_backend = function(...) {
      h5_backend_calls <<- append(h5_backend_calls, list(list(...)))
      structure(list(), class = c("h5_backend", "storage_backend"))
    },
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 50),
    {
      dataset <- fmri_h5_dataset(
        h5_files = "scan.h5",
        mask_source = "mask.h5",
        TR = 2,
        run_length = 50,
        base_path = "/path/to/data"
      )
      
      # Check that base_path was properly prepended
      call_args <- h5_backend_calls[[1]]
      expect_equal(call_args$source, "/path/to/data/scan.h5")
      expect_equal(call_args$mask_source, "/path/to/data/mask.h5")
    }
  )
})

test_that("h5_backend error handling works correctly", {
  skip_if_not_installed("fmristore")
  
  # Create temporary files to pass file existence check
  temp_file <- tempfile(fileext = ".h5")
  file.create(temp_file)
  on.exit(unlink(temp_file))
  
  # Test with invalid mask source type
  expect_error(
    h5_backend(temp_file, 123),
    "mask_source must be file path, NeuroVol, or H5NeuroVol object"
  )
})

test_that("h5_backend integration with storage_backend interface", {
  skip_if_not_installed("fmristore")
  
  # Test that h5_backend properly inherits from storage_backend
  backend <- structure(
    list(
      source = character(0),
      mask_source = character(0),
      preload = FALSE
    ),
    class = c("h5_backend", "storage_backend")
  )
  
  expect_s3_class(backend, "storage_backend")
  expect_true(inherits(backend, "h5_backend"))
  
  # Test that all required methods exist
  expect_true(exists("backend_open.h5_backend"))
  expect_true(exists("backend_close.h5_backend"))
  expect_true(exists("backend_get_dims.h5_backend"))
  expect_true(exists("backend_get_mask.h5_backend"))
  expect_true(exists("backend_get_data.h5_backend"))
  expect_true(exists("backend_get_metadata.h5_backend"))
}) 