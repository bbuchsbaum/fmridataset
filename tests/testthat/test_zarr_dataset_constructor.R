test_that("fmri_zarr_dataset creates valid dataset", {
  skip_if_not_installed("Rarr")

  # Mock Zarr data
  mock_data <- array(rnorm(2 * 2 * 2 * 10), c(2, 2, 2, 10))

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
    zarr_overview = function(...) {
      list(
        dimension = dim(mock_data),
        chunk = c(2, 2, 2, 5),
        compressor = "gzip",
        attributes = list(description = "Test data")
      )
    },
    .package = "Rarr",
    {
      # Create dataset using constructor
      dataset <- fmri_zarr_dataset(
        "test.zarr",
        TR = 2,
        run_length = 10
      )

      # Verify dataset properties
      expect_s3_class(dataset, "fmri_dataset")
      expect_s3_class(dataset$backend, "zarr_backend")
      expect_equal(n_timepoints(dataset), 10)
      expect_equal(get_TR(dataset), 2)

      # Test data access
      data_mat <- get_data_matrix(dataset)
      expect_equal(dim(data_mat), c(10, 8)) # 10 timepoints, 2*2*2 voxels

      # Test mask
      mask <- get_mask(dataset)
      expect_equal(length(mask), 8)
      expect_true(all(mask))
    }
  )
})

test_that("fmri_zarr_dataset handles custom keys", {
  skip_if_not_installed("Rarr")

  mock_data <- array(1:16, c(2, 2, 1, 4))

  with_mocked_bindings(
    read_zarr_array = function(store, path, subset = NULL) {
      if (path == "custom/bold") {
        return(mock_data)
      } else if (path == "custom/brain_mask") {
        return(array(c(TRUE, FALSE, TRUE, TRUE), c(2, 2, 1)))
      }
      stop("Unknown path: ", path)
    },
    zarr_overview = function(...) list(dimension = dim(mock_data)),
    .package = "Rarr",
    {
      dataset <- fmri_zarr_dataset(
        "test.zarr",
        data_key = "custom/bold",
        mask_key = "custom/brain_mask",
        TR = 1,
        run_length = 4
      )

      # Check mask was loaded correctly
      mask <- get_mask(dataset)
      expect_equal(sum(mask), 3) # 3 TRUE values
    }
  )
})

test_that("fmri_zarr_dataset works without mask", {
  skip_if_not_installed("Rarr")

  mock_data <- array(1:8, c(2, 1, 1, 4))

  with_mocked_bindings(
    read_zarr_array = function(store, path, ...) {
      if (path == "data") {
        return(mock_data)
      }
      stop("Mask not found")
    },
    zarr_overview = function(...) list(dimension = dim(mock_data)),
    .package = "Rarr",
    {
      expect_warning(
        dataset <- fmri_zarr_dataset(
          "test.zarr",
          mask_key = "nonexistent",
          TR = 2,
          run_length = 4
        ),
        "Could not load mask"
      )

      # Should work with default mask
      mask <- get_mask(dataset)
      expect_true(all(mask))
    }
  )
})

test_that("fmri_zarr_dataset validates run_length", {
  skip_if_not_installed("Rarr")

  mock_data <- array(1:16, c(2, 2, 2, 2)) # Only 2 timepoints

  with_mocked_bindings(
    read_zarr_array = function(...) mock_data,
    zarr_overview = function(...) list(dimension = dim(mock_data)),
    .package = "Rarr",
    {
      # Run length doesn't match time dimension
      expect_error(
        fmri_zarr_dataset(
          "test.zarr",
          TR = 2,
          run_length = 10 # But data only has 2 timepoints
        ),
        "Sum of run_length"
      )
    }
  )
})
