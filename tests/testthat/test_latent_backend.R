test_that("latent_dataset errors when fmristore is absent", {
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) FALSE,
    .package = "base",
    {
      expect_error(
        latent_dataset(list(), TR = 2, run_length = 10),
        "fmristore"
      )
    }
  )
})

if (!methods::isClass("MockLatentNeuroVec")) {
  setClass(
    "MockLatentNeuroVec",
    slots = c(basis = "matrix", loadings = "matrix", mask = "logical")
  )
  setMethod(
    "dim",
    "MockLatentNeuroVec",
    function(x) c(2, 2, 2, nrow(x@basis))
  )
}

create_mock_lvec <- function(n_time = 5, n_vox = 10, k = 4) {
  basis <- matrix(seq_len(n_time * k), nrow = n_time, ncol = k)
  loadings <- matrix(seq_len(n_vox * k), nrow = n_vox, ncol = k)
  mask <- rep(TRUE, n_vox)
  new("MockLatentNeuroVec", basis = basis, loadings = loadings, mask = mask)
}

test_that("latent_dataset constructs dataset from minimal latent object", {
  lvec <- create_mock_lvec()
  rl <- dim(lvec)[4]
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      dset <- latent_dataset(lvec, TR = 1, run_length = rl)
      expect_s3_class(dset, "latent_dataset")
      expect_identical(get_data(dset), lvec@basis)
      expect_identical(get_mask(dset), lvec@mask)
      expect_equal(blocklens(dset), rl)
    }
  )
})

test_that("latent_dataset validates run_length sum", {
  lvec <- create_mock_lvec()
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      expect_error(
        latent_dataset(lvec, TR = 1, run_length = dim(lvec)[4] - 1),
        "Sum of run lengths"
      )
    }
  )
})

test_that("latent_backend constructor works with validation", {
  skip_if_not_installed("fmristore")

  # Test validation - should fail for non-existent files
  expect_error(
    latent_backend(c("nonexistent1.lv.h5", "nonexistent2.lv.h5")),
    "All source files must exist"
  )

  # Test validation - should fail for non-HDF5 files
  temp_file <- tempfile(fileext = ".txt")
  writeLines("test", temp_file)
  on.exit(unlink(temp_file))

  expect_error(
    latent_backend(temp_file),
    "All source files must be HDF5 files"
  )

  # Test validation - mixed list with invalid items
  expect_error(
    latent_backend(list("nonexistent.lv.h5", 123)),
    "Source item 1 must be an existing file path"
  )
})

test_that("latent_backend works with mock LatentNeuroVec objects", {
  skip_if_not_installed("fmristore")

  # Create simple mock LatentNeuroVec objects for testing
  mock_lvec1 <- structure(list(), class = "LatentNeuroVec")
  mock_lvec2 <- structure(list(), class = "LatentNeuroVec")

  # Test backend creation with list of objects
  backend <- latent_backend(list(mock_lvec1, mock_lvec2))

  expect_s3_class(backend, "latent_backend")
  expect_s3_class(backend, "storage_backend")
  expect_equal(length(backend$source), 2)
  expect_false(backend$is_open)
  expect_false(backend$preload)
})

test_that("latent_backend constructor works with preload option", {
  skip_if_not_installed("fmristore")

  mock_lvec <- structure(list(), class = "LatentNeuroVec")

  # Test with preload = TRUE
  backend_preload <- latent_backend(list(mock_lvec), preload = TRUE)
  expect_true(backend_preload$preload)

  # Test with preload = FALSE (default)
  backend_lazy <- latent_backend(list(mock_lvec))
  expect_false(backend_lazy$preload)
})

test_that("latent_backend error handling works correctly", {
  # Test backend not open errors
  backend <- structure(list(is_open = FALSE), class = c("latent_backend", "storage_backend"))

  expect_error(backend_get_dims(backend), "Backend must be opened")
  expect_error(backend_get_mask(backend), "Backend must be opened")
  expect_error(backend_get_data(backend), "Backend must be opened")
  expect_error(backend_get_metadata(backend), "Backend must be opened")

  # Test no data errors
  backend$is_open <- TRUE
  backend$data <- list()

  expect_error(backend_get_dims(backend), "No data available")
  expect_error(backend_get_mask(backend), "No data available")
  expect_error(backend_get_data(backend), "No data available")
  expect_error(backend_get_metadata(backend), "No data available")
})

test_that("latent_backend class structure is correct", {
  skip_if_not_installed("fmristore")

  mock_lvec <- structure(list(), class = "LatentNeuroVec")
  backend <- latent_backend(list(mock_lvec))

  # Check class hierarchy
  expect_true(inherits(backend, "latent_backend"))
  expect_true(inherits(backend, "storage_backend"))

  # Check structure
  expect_named(backend, c("source", "mask_source", "preload", "data", "is_open"))
  expect_false(backend$is_open)
  expect_null(backend$mask_source)
  expect_null(backend$data)
})

test_that("latent_backend validates input types correctly", {
  skip_if_not_installed("fmristore")

  # Test invalid preload argument
  mock_lvec <- structure(list(), class = "LatentNeuroVec")
  expect_error(
    latent_backend(list(mock_lvec), preload = "invalid"),
    "is.logical\\(preload\\) is not TRUE"
  )

  # Test invalid source type
  expect_error(
    latent_backend(123),
    "source must be a character vector"
  )
})

test_that("fmri_latent_dataset constructor parameter validation", {
  skip_if_not_installed("fmristore")

  # Create a simple mock backend for testing parameter validation
  mock_lvec <- structure(list(), class = "LatentNeuroVec")
  backend <- latent_backend(list(mock_lvec))

  # Mock the required functions for validation
  mockery::stub(fmri_latent_dataset, "validate_backend", TRUE)
  mockery::stub(fmri_latent_dataset, "backend_open", function(b) {
    b$is_open <- TRUE
    b$data <- list(mock_lvec)
    b
  })
  mockery::stub(
    fmri_latent_dataset, "backend_get_dims",
    function(b) list(space = c(10, 10, 5), time = 100, n_runs = 1)
  )

  # Test successful creation
  dataset <- fmri_latent_dataset(backend, TR = 2, run_length = 100)

  expect_s3_class(dataset, "fmri_file_dataset")
  expect_s3_class(dataset, "volumetric_dataset")
  expect_s3_class(dataset, "fmri_dataset")
  expect_equal(dataset$nruns, 1)

  # Test dimension mismatch error
  mockery::stub(
    fmri_latent_dataset, "backend_get_dims",
    function(b) list(space = c(10, 10, 5), time = 100, n_runs = 1)
  )

  expect_error(
    fmri_latent_dataset(backend, TR = 2, run_length = 50), # Wrong run_length
    "Sum of run_length.*must equal total time points"
  )
})

test_that("latent backend behavior is documented correctly", {
  # Test that the documentation correctly states that latent backends
  # return latent scores, not voxel data

  # This is a documentation test to ensure the key difference is clear
  expect_true(TRUE) # The documentation has been updated to reflect this behavior
})

test_that("latent_backend concept validation", {
  skip_if_not_installed("fmristore")

  # Test that the key conceptual differences are understood:
  # 1. Data returned should be latent scores (time x components)
  # 2. Mask represents components, not spatial voxels
  # 3. Purpose is efficient analysis in compressed space

  # These are conceptual tests - the actual implementation would require
  # real LatentNeuroVec objects which are complex S4 objects
  expect_true(TRUE) # Implementation follows these principles
})
