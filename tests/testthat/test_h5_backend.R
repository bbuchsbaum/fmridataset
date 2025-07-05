# Tests for H5 Backend functionality

# Mock methods for H5NeuroVec and H5NeuroVol objects
dim.H5NeuroVec <- function(x) {
  x$space$dims # This should already be a vector
}

dim.H5NeuroVol <- function(x) {
  x$space$dims # This should already be a vector
}

close.H5NeuroVec <- function(con, ...) {
  invisible(NULL)
}

close.H5NeuroVol <- function(con, ...) {
  invisible(NULL)
}

space.H5NeuroVec <- function(x) {
  x$space
}

space.H5NeuroVol <- function(x) {
  x$space
}

as.array.H5NeuroVol <- function(x, ...) {
  x$h5obj[["data/elements"]]
}

as.logical.mock_h5_dataset <- function(x, ...) {
  as.logical(as.vector(x))
}

# Mock neuroim2 series function
series.H5NeuroVec <- function(x, i, ...) {
  data_arr <- x$obj[[x$dataset_name]]
  if (missing(i)) {
    # Return all data as matrix (time x voxels)
    dims <- dim(data_arr)
    matrix(as.vector(data_arr), nrow = dims[4], ncol = prod(dims[1:3]))
  } else {
    # Return data for specific voxel indices
    dims <- dim(data_arr)
    n_time <- dims[4]
    n_voxels <- length(i)

    # Create matrix with time x voxels
    result_matrix <- matrix(0, nrow = n_time, ncol = n_voxels)

    # Fill in data for each voxel index
    for (v in seq_along(i)) {
      voxel_idx <- i[v]
      # Convert linear index to 3D coordinates
      coords <- arrayInd(voxel_idx, dims[1:3])
      # Extract time series for this voxel
      result_matrix[, v] <- data_arr[coords[1], coords[2], coords[3], ]
    }

    result_matrix
  }
}

# Mock methods for NeuroSpace objects
trans.NeuroSpace <- function(x) {
  x$trans
}

spacing.NeuroSpace <- function(x) {
  x$spacing
}

origin.NeuroSpace <- function(x) {
  x$origin
}

dim.NeuroSpace <- function(x) {
  x$dims
}

# Helper function to create mock H5NeuroVec objects
create_mock_h5neurovec <- function(dims = c(10, 10, 5, 50), dataset_name = "data") {
  # Ensure dims is a vector
  dims <- as.numeric(dims)

  # Create a mock H5File object
  mock_h5file <- list(
    `[[` = function(name) {
      if (name == dataset_name) {
        # Return a mock dataset that acts like an array
        structure(
          array(rnorm(prod(dims)), dim = dims),
          class = "mock_h5_dataset"
        )
      } else if (name == "space/dim") {
        structure(dims, class = "mock_h5_attr")
      } else if (name == "space/origin") {
        structure(c(0, 0, 0), class = "mock_h5_attr")
      } else if (name == "space/trans") {
        structure(diag(4), class = "mock_h5_attr")
      }
    },
    exists = function(name) name %in% c(dataset_name, "space/dim", "space/origin", "space/trans"),
    is_valid = TRUE
  )

  # Create mock NeuroSpace - ensure dims is a vector
  mock_space <- structure(
    list(
      dims = dims, # Keep as vector
      origin = c(0, 0, 0),
      trans = diag(4),
      spacing = c(1, 1, 1)
    ),
    class = "NeuroSpace"
  )

  # Create mock H5NeuroVec
  structure(
    list(
      space = mock_space,
      obj = mock_h5file,
      dataset_name = dataset_name
    ),
    class = "H5NeuroVec"
  )
}

# Helper function to create mock H5NeuroVol objects
create_mock_h5neurovol <- function(dims = c(10, 10, 5)) {
  # Ensure dims is a vector
  dims <- as.numeric(dims)

  # Create a mock H5File object
  mock_h5file <- list(
    `[[` = function(name) {
      if (name == "data/elements") {
        structure(
          array(runif(prod(dims)), dim = dims),
          class = "mock_h5_dataset"
        )
      } else if (name == "space/dim") {
        structure(dims, class = "mock_h5_attr")
      } else if (name == "space/origin") {
        structure(c(0, 0, 0), class = "mock_h5_attr")
      } else if (name == "space/trans") {
        structure(diag(4), class = "mock_h5_attr")
      }
    },
    exists = function(name) name %in% c("data/elements", "space/dim", "space/origin", "space/trans"),
    is_valid = TRUE
  )

  # Create mock NeuroSpace - ensure dims is a vector
  mock_space <- structure(
    list(
      dims = dims, # Keep as vector
      origin = c(0, 0, 0),
      trans = diag(4),
      spacing = c(1, 1, 1)
    ),
    class = "NeuroSpace"
  )

  # Create mock H5NeuroVol
  structure(
    list(
      space = mock_space,
      h5obj = mock_h5file
    ),
    class = "H5NeuroVol"
  )
}

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

test_that("h5_backend constructor sets parameters correctly", {
  skip_if_not_installed("fmristore")

  # Create temporary files for validation
  temp_file1 <- tempfile(fileext = ".h5")
  temp_file2 <- tempfile(fileext = ".h5")
  mask_file <- tempfile(fileext = ".h5")
  file.create(c(temp_file1, temp_file2, mask_file))
  on.exit(unlink(c(temp_file1, temp_file2, mask_file)))

  backend <- h5_backend(
    source = c(temp_file1, temp_file2),
    mask_source = mask_file,
    data_dataset = "custom_data",
    mask_dataset = "custom_mask",
    preload = TRUE
  )

  expect_s3_class(backend, "h5_backend")
  expect_s3_class(backend, "storage_backend")
  expect_equal(backend$source, c(temp_file1, temp_file2))
  expect_equal(backend$mask_source, mask_file)
  expect_equal(backend$data_dataset, "custom_data")
  expect_equal(backend$mask_dataset, "custom_mask")
  expect_true(backend$preload)
})

test_that("h5_backend handles custom dataset paths", {
  skip_if_not_installed("fmristore")

  # Create temporary files for validation
  temp_file <- tempfile(fileext = ".h5")
  mask_file <- tempfile(fileext = ".h5")
  file.create(c(temp_file, mask_file))
  on.exit(unlink(c(temp_file, mask_file)))

  backend <- h5_backend(
    source = temp_file,
    mask_source = mask_file,
    data_dataset = "scan_data",
    mask_dataset = "brain_mask"
  )

  expect_equal(backend$data_dataset, "scan_data")
  expect_equal(backend$mask_dataset, "brain_mask")
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

test_that("h5_backend ignores base_path for absolute paths", {
  skip_if_not_installed("fmristore")

  h5_backend_calls <- list()
  with_mocked_bindings(
    h5_backend = function(...) {
      h5_backend_calls <<- append(h5_backend_calls, list(list(...)))
      structure(list(), class = c("h5_backend", "storage_backend"))
    },
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(10,10,5), time = 50),
    {
      dataset <- fmri_h5_dataset(
        h5_files = "/abs/scan.h5",
        mask_source = "/abs/mask.h5",
        TR = 2,
        run_length = 50,
        base_path = "/ignored"
      )

      call_args <- h5_backend_calls[[1]]
      expect_equal(call_args$source, "/abs/scan.h5")
      expect_equal(call_args$mask_source, "/abs/mask.h5")
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

test_that("h5_backend structure validation", {
  skip_if_not_installed("fmristore")

  # Create temporary files
  temp_file <- tempfile(fileext = ".h5")
  mask_file <- tempfile(fileext = ".h5")
  file.create(c(temp_file, mask_file))
  on.exit(unlink(c(temp_file, mask_file)))

  backend <- h5_backend(
    source = temp_file,
    mask_source = mask_file
  )

  # Test that backend has all required fields
  expect_true("source" %in% names(backend))
  expect_true("mask_source" %in% names(backend))
  expect_true("preload" %in% names(backend))
  expect_true("data_dataset" %in% names(backend))
  expect_true("mask_dataset" %in% names(backend))
  expect_true("h5_objects" %in% names(backend))
  expect_true("mask" %in% names(backend))
  expect_true("dims" %in% names(backend))

  # Test default values
  expect_equal(backend$data_dataset, "data")
  expect_equal(backend$mask_dataset, "data/elements")
  expect_false(backend$preload)
  expect_null(backend$h5_objects)
  expect_null(backend$mask)
  expect_null(backend$dims)
})

test_that("h5_backend handles multiple source files", {
  skip_if_not_installed("fmristore")

  # Create temporary files
  temp_files <- paste0(tempfile(), c("_1.h5", "_2.h5", "_3.h5"))
  mask_file <- tempfile(fileext = ".h5")
  file.create(c(temp_files, mask_file))
  on.exit(unlink(c(temp_files, mask_file)))

  backend <- h5_backend(
    source = temp_files,
    mask_source = mask_file,
    preload = FALSE
  )

  expect_equal(length(backend$source), 3)
  expect_equal(backend$source, temp_files)
})

test_that("fmri_h5_dataset validates parameters", {
  skip_if_not_installed("fmristore")

  # Mock dependencies to focus on parameter validation
  with_mocked_bindings(
    h5_backend = function(...) structure(list(), class = c("h5_backend", "storage_backend")),
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 100),
    {
      # Test that function validates TR
      expect_error(
        fmri_h5_dataset(
          h5_files = "scan.h5",
          mask_source = "mask.h5",
          TR = -1, # Invalid TR
          run_length = 100
        ),
        "TR values must be positive"
      )

      # Test that valid parameters work
      expect_silent({
        result <- fmri_h5_dataset(
          h5_files = c("scan1.h5", "scan2.h5"),
          mask_source = "mask.h5",
          TR = 2,
          run_length = c(50, 50) # Total matches mock time dimension
        )
      })
    }
  )
})

test_that("h5_backend basic functionality works", {
  skip_if_not_installed("fmristore")

  # Create temporary files
  temp_file <- tempfile(fileext = ".h5")
  mask_file <- tempfile(fileext = ".h5")
  file.create(c(temp_file, mask_file))
  on.exit(unlink(c(temp_file, mask_file)))

  # Test basic backend creation and method existence
  backend <- h5_backend(
    source = temp_file,
    mask_source = mask_file
  )

  # Test that we can call basic methods without errors
  expect_silent(backend_close(backend))

  # Test backend configuration
  expect_false(backend$preload)
  expect_equal(backend$source, temp_file)
  expect_equal(backend$mask_source, mask_file)
})
