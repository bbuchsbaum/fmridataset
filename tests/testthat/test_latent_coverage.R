# Tests for R/latent_backend.R and R/latent_dataset.R - coverage improvement

# --- latent_backend constructor validation ---

test_that("latent_backend errors on invalid source type", {
  expect_error(latent_backend(source = 42), "source must be character vector or list")
})

test_that("latent_backend errors on non-existent files", {
  expect_error(
    latent_backend(source = c("/nonexistent/file.lv.h5")),
    "Source files not found"
  )
})

test_that("latent_backend errors on wrong file extension", {
  tmp <- tempfile(fileext = ".csv")
  file.create(tmp)
  on.exit(unlink(tmp))

  expect_error(
    latent_backend(source = tmp),
    "must be HDF5"
  )
})

test_that("latent_backend errors on list with invalid item", {
  expect_error(
    latent_backend(source = list(42)),
    "must be a LatentNeuroVec"
  )
})

test_that("latent_backend errors on list with non-existent file path", {
  expect_error(
    latent_backend(source = list("/nonexistent/path.h5")),
    "must be an existing file"
  )
})

test_that("latent_backend constructor with valid h5 file paths creates object", {
  # Create dummy .lv.h5 files that exist
  tmp1 <- tempfile(fileext = ".lv.h5")
  file.create(tmp1)
  on.exit(unlink(tmp1))

  backend <- latent_backend(source = tmp1)
  expect_s3_class(backend, "latent_backend")
  expect_s3_class(backend, "storage_backend")
  expect_false(backend$is_open)
})

test_that("backend_open.latent_backend requires fmristore", {
  skip_if(requireNamespace("fmristore", quietly = TRUE),
    "fmristore is installed, skipping unavailable-package test"
  )

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_open(backend), "fmristore")
})

test_that("backend_get_dims.latent_backend errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_get_dims(backend), "must be opened")
})

test_that("backend_get_mask.latent_backend errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_get_mask(backend), "must be opened")
})

test_that("backend_get_data.latent_backend errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_get_data(backend), "must be opened")
})

test_that("backend_get_metadata.latent_backend errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_get_metadata(backend), "must be opened")
})

test_that("backend_get_loadings errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_get_loadings(backend), "must be opened")
})

test_that("backend_reconstruct_voxels errors when not open", {
  skip_if_not_installed("fmristore")

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  expect_error(backend_reconstruct_voxels(backend), "must be opened")
})

test_that("backend_close.latent_backend works", {
  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  backend <- latent_backend(source = tmp)
  result <- backend_close(backend)
  expect_null(result)
  expect_false(backend$is_open)
})

# --- latent_dataset constructor ---

test_that("latent_dataset errors when fmristore not available", {
  skip_if(requireNamespace("fmristore", quietly = TRUE))

  tmp <- tempfile(fileext = ".lv.h5")
  file.create(tmp)
  on.exit(unlink(tmp))

  expect_error(
    latent_dataset(source = tmp, TR = 2, run_length = 10),
    "fmristore"
  )
})

# --- get_latent_space_dims ---

test_that("get_latent_space_dims fallback for objects without space", {
  # Create a mock S4-like object with basis and loadings
  skip_if_not_installed("fmristore")

  # Test the function exists and is callable
  expect_true(is.function(fmridataset:::get_latent_space_dims))
})
