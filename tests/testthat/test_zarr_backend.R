test_that("zarr_backend validates inputs correctly", {
  # Test invalid source
  expect_error(
    zarr_backend(NULL),
    "source must be a single character string"
  )

  expect_error(
    zarr_backend(c("path1", "path2")),
    "source must be a single character string"
  )

  # Test that constructor returns correct class (skip if Rarr not installed)
  skip_if_not_installed("Rarr")
  backend <- zarr_backend("dummy.zarr")
  expect_s3_class(backend, "zarr_backend")
  expect_s3_class(backend, "storage_backend")
})

test_that("zarr_backend requires Rarr package", {
  # Mock the requireNamespace function to simulate missing package
  with_mocked_bindings(
    requireNamespace = function(...) FALSE,
    .package = "base",
    {
      expect_error(
        zarr_backend("dummy.zarr"),
        "The Rarr package is required"
      )
    }
  )
})

test_that("zarr_backend handles missing files gracefully", {
  skip_if_not_installed("Rarr")

  backend <- zarr_backend("/nonexistent/path.zarr")
  expect_error(
    backend_open(backend),
    "Zarr store not found"
  )
})

test_that("zarr_backend works with mock data", {
  skip_if_not_installed("Rarr")

  # Create a temporary directory for mock zarr store
  temp_dir <- tempdir()
  zarr_path <- file.path(temp_dir, "test_mock.zarr")

  # Mock Rarr functions for testing
  mock_array_info <- list(
    dimension = c(10, 10, 10, 100),
    chunk = c(5, 5, 5, 50),
    compressor = "gzip",
    attributes = list(TR = 2.0)
  )

  mock_data <- array(rnorm(10 * 10 * 10 * 100), c(10, 10, 10, 100))
  mock_mask <- array(TRUE, c(10, 10, 10))

  # Create backend with mocked functions
  with_mocked_bindings(
    read_zarr_array = function(store, path, subset = NULL) {
      if (path == "data") {
        if (!is.null(subset)) {
          # Handle subsetting
          return(mock_data[subset[[1]] %||% TRUE,
            subset[[2]] %||% TRUE,
            subset[[3]] %||% TRUE,
            subset[[4]] %||% TRUE,
            drop = FALSE
          ])
        }
        return(mock_data)
      } else if (path == "mask") {
        return(mock_mask)
      }
      stop("Unknown path")
    },
    zarr_overview = function(array) mock_array_info,
    .package = "Rarr",
    {
      # Create and open backend
      backend <- zarr_backend(zarr_path, data_key = "data", mask_key = "mask")
      backend <- backend_open(backend)

      # Test dimensions
      dims <- backend_get_dims(backend)
      expect_equal(dims$spatial, c(10, 10, 10))
      expect_equal(dims$time, 100)

      # Test mask
      mask <- backend_get_mask(backend)
      expect_equal(length(mask), 1000) # 10*10*10
      expect_true(all(mask))

      # Test full data read
      data <- backend_get_data(backend)
      expect_equal(dim(data), c(100, 1000)) # time x voxels

      # Test subset read
      subset_data <- backend_get_data(backend, rows = 1:10, cols = 1:50)
      expect_equal(dim(subset_data), c(10, 50))

      # Test metadata
      metadata <- backend_get_metadata(backend)
      expect_equal(metadata$storage_format, "zarr")
      expect_equal(metadata$TR, 2.0)

      # Test close
      expect_silent(backend_close(backend))
    }
  )
})

test_that("zarr_backend handles preload option", {
  skip_if_not_installed("Rarr")

  mock_data <- array(1:24, c(2, 3, 2, 2))
  called_count <- 0

  with_mocked_bindings(
    read_zarr_array = function(store, path, subset = NULL) {
      called_count <<- called_count + 1
      if (!is.null(subset) && all(sapply(subset, is.null))) {
        # Full read
        return(mock_data)
      }
      return(mock_data)
    },
    zarr_overview = function(array) {
      list(dimension = c(2, 3, 2, 2))
    },
    .package = "Rarr",
    {
      # Test with preload = TRUE
      backend <- zarr_backend("dummy.zarr", preload = TRUE)
      backend <- backend_open(backend)

      # Should have loaded data during open
      initial_calls <- called_count

      # Multiple data accesses should not increase call count
      data1 <- backend_get_data(backend)
      data2 <- backend_get_data(backend)

      expect_equal(called_count, initial_calls) # No additional calls
    }
  )
})

test_that("zarr_backend validates array dimensions", {
  skip_if_not_installed("Rarr")

  # Test with wrong number of dimensions
  with_mocked_bindings(
    read_zarr_array = function(...) array(1:8, c(2, 2, 2)),
    zarr_overview = function(...) list(dimension = c(2, 2, 2)), # 3D instead of 4D
    .package = "Rarr",
    {
      backend <- zarr_backend("dummy.zarr")
      expect_error(
        backend_open(backend),
        "Expected 4D array, got 3D"
      )
    }
  )
})

test_that("zarr_backend handles missing mask gracefully", {
  skip_if_not_installed("Rarr")

  mock_data <- array(1:16, c(2, 2, 2, 2))

  with_mocked_bindings(
    read_zarr_array = function(store, path, ...) {
      if (path == "data") {
        return(mock_data)
      }
      stop("Mask not found") # Simulate missing mask
    },
    zarr_overview = function(...) list(dimension = c(2, 2, 2, 2)),
    .package = "Rarr",
    {
      backend <- zarr_backend("dummy.zarr", mask_key = "mask")
      expect_warning(
        backend <- backend_open(backend),
        "Could not load mask"
      )

      # Should still work with default mask
      mask <- backend_get_mask(backend)
      expect_equal(length(mask), 8) # 2*2*2
      expect_true(all(mask))
    }
  )
})

test_that("zarr_backend handles remote URLs", {
  skip_if_not_installed("Rarr")

  # Test S3 URL
  backend <- zarr_backend("s3://bucket/path/data.zarr")
  expect_equal(backend$source, "s3://bucket/path/data.zarr")

  # Test HTTPS URL
  backend <- zarr_backend("https://example.com/data.zarr")
  expect_equal(backend$source, "https://example.com/data.zarr")
})

test_that("zarr_backend integrates with fmri_dataset", {
  skip_if_not_installed("Rarr")

  mock_data <- array(rnorm(8 * 10), c(2, 2, 2, 10))

  with_mocked_bindings(
    read_zarr_array = function(store, path, subset = NULL) {
      if (path == "data") {
        if (!is.null(subset)) {
          return(mock_data[subset[[1]] %||% TRUE,
            subset[[2]] %||% TRUE,
            subset[[3]] %||% TRUE,
            subset[[4]] %||% TRUE,
            drop = FALSE
          ])
        }
        return(mock_data)
      }
      array(TRUE, c(2, 2, 2)) # mask
    },
    zarr_overview = function(...) list(dimension = dim(mock_data)),
    .package = "Rarr",
    {
      # Create backend
      backend <- zarr_backend("test.zarr")

      # Create dataset
      dataset <- fmri_dataset(
        backend,
        TR = 2,
        run_length = 10
      )

      expect_s3_class(dataset, "fmri_dataset")
      expect_equal(n_timepoints(dataset), 10)

      # Test data access
      data_mat <- get_data_matrix(dataset)
      expect_equal(dim(data_mat), c(10, 8))
    }
  )
})

test_that("zarr_backend chooses optimal I/O strategy", {
  skip_if_not_installed("Rarr")

  # Large array with chunks
  large_dims <- c(64, 64, 40, 200)
  chunk_dims <- c(32, 32, 20, 50)

  mock_data <- array(seq_len(prod(large_dims)), large_dims)
  read_strategy <- NULL

  with_mocked_bindings(
    read_zarr_array = function(store, path, subset = NULL) {
      if (path == "data") {
        if (is.null(subset)) {
          read_strategy <<- "full"
          return(mock_data)
        } else if (all(sapply(subset, is.null))) {
          read_strategy <<- "full"
          return(mock_data)
        } else {
          # Track if we're doing chunk or voxel reads
          if (length(subset[[1]]) > 1 || length(subset[[2]]) > 1 ||
            length(subset[[3]]) > 1) {
            read_strategy <<- "chunk"
          } else {
            read_strategy <<- "voxel"
          }
          return(mock_data[subset[[1]] %||% TRUE,
            subset[[2]] %||% TRUE,
            subset[[3]] %||% TRUE,
            subset[[4]] %||% TRUE,
            drop = FALSE
          ])
        }
      }
      array(TRUE, large_dims[1:3]) # mask
    },
    zarr_overview = function(...) {
      list(
        dimension = large_dims,
        chunk = chunk_dims
      )
    },
    .package = "Rarr",
    {
      backend <- zarr_backend("test.zarr")
      backend <- backend_open(backend)

      # Test 1: Large subset (>50%) should use full read
      read_strategy <- NULL
      n_voxels <- prod(large_dims[1:3])
      large_cols <- seq_len(round(n_voxels * 0.6))
      data <- backend_get_data(backend, rows = 1:150, cols = large_cols)
      expect_equal(read_strategy, "full")

      # Test 2: Small subset should use chunk or voxel strategy
      read_strategy <- NULL
      small_cols <- 1:10
      small_rows <- 1:5
      data <- backend_get_data(backend, rows = small_rows, cols = small_cols)
      expect_true(read_strategy %in% c("chunk", "voxel"))
    }
  )
})

test_that("zarr_backend helper functions work correctly", {
  skip_if_not_installed("Rarr")

  # Test estimate_zarr_chunks_needed
  chunk_shape <- c(32, 32, 20, 50)
  array_shape <- c(64, 64, 40, 200)

  # All data - should need all chunks
  rows <- 1:200
  cols <- 1:prod(array_shape[1:3])
  n_chunks <- estimate_zarr_chunks_needed(chunk_shape, array_shape, rows, cols)
  expect_equal(n_chunks, prod(array_shape / chunk_shape))

  # Single chunk
  rows <- 1:50
  cols <- 1:1024 # First 32x32x1 voxels
  n_chunks <- estimate_zarr_chunks_needed(chunk_shape, array_shape, rows, cols)
  expect_true(n_chunks >= 1)
})
