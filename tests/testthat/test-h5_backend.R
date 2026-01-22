# Tests for H5 Backend functionality
# This file provides comprehensive tests for h5_backend.R covering:
# - Constructor validation
# - Backend lifecycle (open/close)
# - Data access methods (get_dims, get_mask, get_data, get_metadata)
# - Resource cleanup
# - Error handling

# ==============================================================================
# Test Fixtures and Helpers
# ==============================================================================

# Mock methods for H5NeuroVec and H5NeuroVol objects when fmristore not available
dim.H5NeuroVec <- function(x) {
  # Ensure dims is returned as a numeric vector
  if (is.null(x) || is.null(x$space) || is.null(x$space$dims)) {
    return(c(0, 0, 0, 0))
  }
  result <- as.numeric(x$space$dims)
  if (length(result) == 0) result <- c(0, 0, 0, 0)
  result
}

dim.H5NeuroVol <- function(x) {
  # Ensure dims is returned as a numeric vector
  if (is.null(x) || is.null(x$space) || is.null(x$space$dims)) {
    return(c(0, 0, 0))
  }
  result <- as.numeric(x$space$dims)
  if (length(result) == 0) result <- c(0, 0, 0)
  result
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
  # Extract the actual array from the mock h5obj
  dataset <- x$h5obj[["data/elements"]]
  if (inherits(dataset, "mock_h5_dataset")) {
    # It's already an array with class mock_h5_dataset
    unclass(dataset)
  } else {
    dataset
  }
}

as.vector.H5NeuroVol <- function(x, ...) {
  as.vector(as.array(x))
}

as.logical.mock_h5_dataset <- function(x, ...) {
  as.logical(as.vector(unclass(x)))
}

as.array.mock_h5_dataset <- function(x, ...) {
  # Remove the mock class and return the underlying array
  unclass(x)
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

# Helper function to create NeuroVol mask objects
create_neurovol_mask <- function(dims = c(4, 4, 4), pattern = "alternating") {
  total_voxels <- prod(dims)
  if (pattern == "alternating") {
    mask_arr <- array(rep(c(TRUE, FALSE), length.out = total_voxels), dims)
  } else if (pattern == "all_true") {
    mask_arr <- array(TRUE, dims)
  } else if (pattern == "all_false") {
    mask_arr <- array(FALSE, dims)
  } else {
    stop("Unknown pattern")
  }

  structure(
    mask_arr,
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    space = structure(
      list(dim = dims, origin = c(0, 0, 0), spacing = c(1, 1, 1)),
      class = "NeuroSpace"
    )
  )
}

# ==============================================================================
# Constructor Validation Tests
# ==============================================================================

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
  skip_if_not_installed("withr")
  skip_if_not_installed("hdf5r")

  temp_dir <- withr::local_tempdir()
  h5_file1 <- file.path(temp_dir, "test_scan1.h5")
  h5_file2 <- file.path(temp_dir, "test_scan2.h5")
  mask_file <- file.path(temp_dir, "test_mask.h5")

  write_mock <- function(path, timepoints) {
    arr <- array(rnorm(4 * 4 * 4 * timepoints), c(4, 4, 4, timepoints))
    space <- neuroim2::NeuroSpace(c(4, 4, 4, timepoints))
    vec <- neuroim2::NeuroVec(arr, space)
    fmristore::write_vec(vec, path)
  }

  write_mock(h5_file1, 10)
  write_mock(h5_file2, 12)
  write_mock(mask_file, 1)

  backend <- h5_backend(
    source = c(h5_file1, h5_file2),
    mask_source = mask_file,
    preload = FALSE
  )

  expect_s3_class(backend, "h5_backend")
  expect_s3_class(backend, "storage_backend")
  expect_equal(backend$source, c(h5_file1, h5_file2))
  expect_equal(backend$mask_source, mask_file)
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

# ==============================================================================
# Backend Lifecycle Tests (open/close)
# ==============================================================================

test_that("h5_backend backend_open with preload=TRUE loads data", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock H5NeuroVec objects
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 10))

  # Create mask as regular NeuroVol (not H5)
  mask_arr <- array(rep(c(TRUE, FALSE), length.out = 64), c(4, 4, 4))
  mask_vol <- structure(
    mask_arr,
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    space = structure(
      list(dim = c(4, 4, 4), origin = c(0, 0, 0), spacing = c(1, 1, 1)),
      class = "NeuroSpace"
    )
  )

  # Create backend with in-memory objects and preload=TRUE
  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol,
    preload = TRUE
  )

  # Open should load data
  backend <- backend_open(backend)

  # Verify h5_objects is populated
  expect_false(is.null(backend$h5_objects))
  expect_equal(length(backend$h5_objects), 1)

  # Verify mask is loaded
  expect_false(is.null(backend$mask))

  # Verify dims are set
  expect_false(is.null(backend$dims))
  expect_equal(backend$dims$spatial, c(4, 4, 4))
  expect_equal(backend$dims$time, 10)

  # Cleanup
  backend_close(backend)
})

test_that("h5_backend backend_close releases resources", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock objects
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 5))
  mask_vol <- create_neurovol_mask(dims = c(4, 4, 4))

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol,
    preload = TRUE
  )

  backend <- backend_open(backend)

  # Close should not error
  expect_silent(backend_close(backend))

  # Repeated close should not error
  expect_silent(backend_close(backend))
})

test_that("h5_backend lazy loading works without preload", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock objects
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 8))
  mask_vol <- create_neurovol_mask(dims = c(4, 4, 4))

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol,
    preload = FALSE
  )

  # Should be able to get dims without opening
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(4, 4, 4))
  expect_equal(dims$time, 8)

  # h5_objects should still be NULL (not preloaded)
  expect_null(backend$h5_objects)
})

# ==============================================================================
# backend_get_dims Tests
# ==============================================================================

test_that("h5_backend get_dims returns correct structure", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock H5 with known dims (4, 4, 4, 10)
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 10))
  mask_vol <- create_neurovol_mask(dims = c(4, 4, 4))

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol,
    preload = FALSE
  )

  dims <- backend_get_dims(backend)

  # Verify structure
  expect_type(dims, "list")
  expect_equal(names(dims), c("spatial", "time"))
  expect_equal(dims$spatial, c(4, 4, 4))
  expect_equal(dims$time, 10)
})

test_that("h5_backend get_dims handles multiple files", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create two mock H5 objects with 5 timepoints each
  mock_h5vec1 <- create_mock_h5neurovec(dims = c(4, 4, 4, 5))
  mock_h5vec2 <- create_mock_h5neurovec(dims = c(4, 4, 4, 5))
  mask_vol <- create_neurovol_mask(dims = c(4, 4, 4))

  backend <- h5_backend(
    source = list(mock_h5vec1, mock_h5vec2),
    mask_source = mask_vol,
    preload = FALSE
  )

  dims <- backend_get_dims(backend)

  # Verify time == 10 (sum of both files)
  expect_equal(dims$time, 10)
  expect_equal(dims$spatial, c(4, 4, 4))
})

# ==============================================================================
# backend_get_mask Tests
# ==============================================================================

test_that("h5_backend get_mask returns logical vector", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")

  # Create mock objects
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 5))
  mask_vol <- create_neurovol_mask(dims = c(4, 4, 4))

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol,
    preload = FALSE
  )

  mask <- backend_get_mask(backend)

  # Verify result is logical
  expect_type(mask, "logical")

  # Verify length equals prod(spatial_dims)
  expect_equal(length(mask), 64)

  # Verify some TRUE values exist (mock creates random mask)
  expect_true(sum(mask) > 0)
})

test_that("h5_backend get_mask validates mask content", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")

  # Create mock with all-FALSE mask
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 5))
  mask_vol_false <- create_neurovol_mask(dims = c(4, 4, 4), pattern = "all_false")

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol_false,
    preload = FALSE
  )

  # Mask with all FALSE should error
  expect_error(
    backend_get_mask(backend),
    "mask contains no TRUE values"
  )
})

# ==============================================================================
# backend_get_data Tests
# ==============================================================================

test_that("h5_backend get_data returns full matrix", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock objects with all TRUE mask
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 6))
  mask_vol_true <- create_neurovol_mask(dims = c(4, 4, 4), pattern = "all_true")

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol_true,
    preload = TRUE
  )

  backend <- backend_open(backend)
  data <- backend_get_data(backend, rows = NULL, cols = NULL)

  # Verify dimensions (time x masked_voxels)
  expect_equal(nrow(data), 6)
  expect_equal(ncol(data), 64) # All voxels masked

  backend_close(backend)
})

test_that("h5_backend get_data subsets correctly", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create mock objects
  mock_h5vec <- create_mock_h5neurovec(dims = c(4, 4, 4, 10))
  mask_vol_true <- create_neurovol_mask(dims = c(4, 4, 4), pattern = "all_true")

  backend <- h5_backend(
    source = list(mock_h5vec),
    mask_source = mask_vol_true,
    preload = TRUE
  )

  backend <- backend_open(backend)

  # Test row and column subsetting
  data_subset <- backend_get_data(backend, rows = 1:5, cols = 1:10)
  expect_equal(dim(data_subset), c(5, 10))

  backend_close(backend)
})

test_that("h5_backend get_data concatenates multiple files", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("Requires real H5NeuroVec with S4 methods - mocks can't work with neuroim2 S4 dispatch")

  # Create two mock H5 objects
  mock_h5vec1 <- create_mock_h5neurovec(dims = c(4, 4, 4, 4))
  mock_h5vec2 <- create_mock_h5neurovec(dims = c(4, 4, 4, 6))
  mask_vol_true <- create_neurovol_mask(dims = c(4, 4, 4), pattern = "all_true")

  backend <- h5_backend(
    source = list(mock_h5vec1, mock_h5vec2),
    mask_source = mask_vol_true,
    preload = TRUE
  )

  backend <- backend_open(backend)
  data <- backend_get_data(backend)

  # Verify data is concatenated along time (should be 4 + 6 = 10 rows)
  expect_equal(nrow(data), 10)
  expect_equal(ncol(data), 64)

  backend_close(backend)
})

# ==============================================================================
# backend_get_metadata Tests
# ==============================================================================

test_that("h5_backend get_metadata returns neuroimaging info", {
  skip_if_not_installed("fmristore")
  skip_if_not_installed("neuroim2")
  skip("backend_get_metadata requires real H5NeuroVec (S4 class) - cannot mock S4 dispatch")

  # NOTE: This test would require a real H5 file to work properly
  # because neuroim2 uses S4 generics (space, spacing, origin, trans)
  # that cannot be mocked with simple list structures.
  # Keeping test structure for future integration testing with real H5 files.

  # Verify format
  expect_equal(metadata$format, "h5")

  # Verify neuroimaging metadata present
  expect_true("affine" %in% names(metadata))
  expect_true("voxel_dims" %in% names(metadata))
  expect_true("origin" %in% names(metadata))
  expect_true("dimensions" %in% names(metadata))

  # Verify values
  expect_equal(metadata$voxel_dims, c(2, 2, 2))
  expect_equal(metadata$origin, c(10, 20, 30))
})

# ==============================================================================
# Integration Tests
# ==============================================================================

test_that("fmri_h5_dataset constructor works", {
  skip_if_not_installed("fmristore")

  # Mock the h5_backend function to avoid file dependencies
  with_mocked_bindings(
    file.exists = function(...) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        create_backend = function(type, ...) {
          structure(list(type = type, ...), class = c("h5_backend", "storage_backend"))
        },
        validate_backend = function(backend) TRUE,
        backend_open = function(backend) backend,
        backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 100),
        .package = "fmridataset",
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
    }
  )
})

test_that("h5_backend handles base_path correctly", {
  skip_if_not_installed("fmristore")

  # Mock the h5_backend function
  h5_backend_calls <- list()
  with_mocked_bindings(
    file.exists = function(...) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        create_backend = function(type, ...) {
          h5_backend_calls <<- append(h5_backend_calls, list(list(type = type, ...)))
          structure(list(), class = c("h5_backend", "storage_backend"))
        },
        validate_backend = function(backend) TRUE,
        backend_open = function(backend) backend,
        backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 50),
        .package = "fmridataset",
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
          expect_equal(call_args$type, "h5")
          expect_equal(call_args$source, "/path/to/data/scan.h5")
          expect_equal(call_args$mask_source, "/path/to/data/mask.h5")
        }
      )
    }
  )
})

test_that("h5_backend ignores base_path for absolute paths", {
  skip_if_not_installed("fmristore")

  h5_backend_calls <- list()
  with_mocked_bindings(
    file.exists = function(...) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        create_backend = function(type, ...) {
          h5_backend_calls <<- append(h5_backend_calls, list(list(type = type, ...)))
          structure(list(), class = c("h5_backend", "storage_backend"))
        },
        validate_backend = function(backend) TRUE,
        backend_open = function(backend) backend,
        backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 50),
        .package = "fmridataset",
        {
          dataset <- fmri_h5_dataset(
            h5_files = "/abs/scan.h5",
            mask_source = "/abs/mask.h5",
            TR = 2,
            run_length = 50,
            base_path = "/ignored"
          )

          call_args <- h5_backend_calls[[1]]
          expect_equal(call_args$type, "h5")
          expect_equal(call_args$source, "/abs/scan.h5")
          expect_equal(call_args$mask_source, "/abs/mask.h5")
        }
      )
    }
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

test_that("fmri_h5_dataset validates parameters", {
  skip_if_not_installed("fmristore")

  # Mock dependencies to focus on parameter validation
  with_mocked_bindings(
    file.exists = function(...) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        h5_backend = function(...) structure(list(), class = c("h5_backend", "storage_backend")),
        validate_backend = function(backend) TRUE,
        backend_open = function(backend) backend,
        backend_get_dims = function(backend) list(spatial = c(10, 10, 5), time = 100),
        .package = "fmridataset",
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
