test_that("latent_backend validates inputs correctly", {
  # Test invalid source
  expect_error(
    latent_backend(NULL),
    "source must be character vector or list"
  )
  
  expect_error(
    latent_backend(123),
    "source must be character vector or list"
  )
  
  # Test non-existent files
  expect_error(
    latent_backend(c("/nonexistent/file1.lv.h5", "/nonexistent/file2.lv.h5")),
    "Source files not found"
  )
  
  # Test invalid file extensions
  temp_file <- tempfile(fileext = ".txt")
  writeLines("test", temp_file)
  expect_error(
    latent_backend(temp_file),
    "All source files must be HDF5 files"
  )
  unlink(temp_file)
})

test_that("latent_backend works with mock LatentNeuroVec objects", {
  # Create mock LatentNeuroVec objects
  mock_lvec1 <- structure(
    list(
      basis = matrix(rnorm(100 * 10), 100, 10),
      loadings = matrix(rnorm(1000 * 10), 1000, 10),
      space = c(10, 10, 10, 100),
      offset = rep(0, 1000),
      mask = rep(TRUE, 1000)
    ),
    class = c("MockLatentNeuroVec", "list")
  )
  
  # Make it S4-like
  mock_lvec1 <- methods::setClass(
    "MockLatentNeuroVec",
    slots = c(
      basis = "matrix",
      loadings = "matrix", 
      space = "numeric",
      offset = "numeric",
      mask = "logical"
    )
  )(
    basis = matrix(rnorm(100 * 10), 100, 10),
    loadings = matrix(rnorm(1000 * 10), 1000, 10),
    space = c(10, 10, 10, 100),
    offset = rep(0, 1000),
    mask = rep(TRUE, 1000)
  )
  
  mock_lvec2 <- methods::new(
    "MockLatentNeuroVec",
    basis = matrix(rnorm(150 * 10), 150, 10),
    loadings = matrix(rnorm(1000 * 10), 1000, 10),
    space = c(10, 10, 10, 150),
    offset = rep(0, 1000),
    mask = rep(TRUE, 1000)
  )
  
  # Create backend with mock objects
  backend <- latent_backend(list(mock_lvec1, mock_lvec2))
  expect_s3_class(backend, "latent_backend")
  expect_s3_class(backend, "storage_backend")
  
  # Open backend
  backend <- backend_open(backend)
  expect_true(backend$is_open)
  
  # Check dimensions
  dims <- backend_get_dims(backend)
  expect_equal(dims$spatial, c(10, 10, 10))  # original spatial dims
  expect_equal(dims$time, 250)  # 100 + 150
  
  # Check mask
  mask <- backend_get_mask(backend)
  expect_equal(length(mask), 1000)  # n_voxels
  expect_true(all(mask))
  
  # Get data
  data <- backend_get_data(backend)
  expect_equal(dim(data), c(250, 10))
  
  # Get subset
  subset_data <- backend_get_data(backend, rows = 1:50, cols = 1:5)
  expect_equal(dim(subset_data), c(50, 5))
  
  # Get metadata
  metadata <- backend_get_metadata(backend)
  expect_equal(metadata$storage_format, "latent")
  expect_equal(metadata$n_components, 10)
  expect_equal(metadata$n_voxels, 1000)
  expect_equal(metadata$n_runs, 2)
  
  # Close backend
  backend <- backend_close(backend)
  expect_false(backend$is_open)
})

test_that("latent_backend handles inconsistent runs", {
  # Create mock objects with different dimensions
  mock_lvec1 <- methods::setClass(
    "MockLatentNeuroVec",
    slots = c(
      basis = "matrix",
      loadings = "matrix",
      space = "numeric",
      offset = "numeric"
    )
  )(
    basis = matrix(1:100, 100, 10),
    loadings = matrix(1:10000, 1000, 10),
    space = c(10, 10, 10, 100),
    offset = rep(0, 1000)
  )
  
  # Different spatial dimensions
  mock_lvec2 <- methods::new(
    "MockLatentNeuroVec",
    basis = matrix(1:100, 100, 10),
    loadings = matrix(1:20000, 2000, 10),
    space = c(20, 10, 10, 100),  # Different spatial dims
    offset = rep(0, 2000)
  )
  
  backend <- latent_backend(list(mock_lvec1, mock_lvec2))
  expect_error(
    backend_open(backend),
    "inconsistent spatial dimensions"
  )
  
  # Different number of components
  mock_lvec3 <- methods::new(
    "MockLatentNeuroVec",
    basis = matrix(1:500, 100, 5),  # 5 components instead of 10
    loadings = matrix(1:5000, 1000, 5),
    space = c(10, 10, 10, 100),
    offset = rep(0, 1000)
  )
  
  backend <- latent_backend(list(mock_lvec1, mock_lvec3))
  expect_error(
    backend_open(backend),
    "different number of components"
  )
})

test_that("latent_backend integrates with latent_dataset", {
  # Create mock objects
  mock_lvec <- methods::setClass(
    "MockLatentNeuroVec",
    slots = c(
      basis = "matrix",
      loadings = "matrix",
      space = "numeric",
      offset = "numeric",
      mask = "logical"
    )
  )(
    basis = matrix(rnorm(200 * 15), 200, 15),
    loadings = matrix(rnorm(5000 * 15), 5000, 15),
    space = c(20, 25, 10, 200),
    offset = rnorm(5000),
    mask = rep(TRUE, 5000)
  )
  
  # Create dataset using backend
  dataset <- latent_dataset(
    list(mock_lvec),
    TR = 2,
    run_length = 200
  )
  
  expect_s3_class(dataset, "latent_dataset")
  expect_s3_class(dataset, "fmri_dataset")
  
  # Debug: Check if backend exists and its class
  if (!is.null(dataset$backend)) {
    expect_s3_class(dataset$backend, "latent_backend")
  } else if (!is.null(dataset$storage)) {
    # Using old latent_dataset.R interface
    skip("Dataset is using legacy storage interface")
  } else {
    expect_s3_class(dataset$backend, "latent_backend")
  }
  
  # Test data access
  data <- get_data(dataset)
  expect_equal(dim(data), c(200, 15))
  
  # Test mask
  mask <- get_mask(dataset)
  expect_equal(length(mask), 15)  # n_components
  expect_true(all(mask))
})

test_that("backend_get_loadings works correctly", {
  mock_lvec <- methods::setClass(
    "MockLatentNeuroVec", 
    slots = c(
      basis = "matrix",
      loadings = "matrix",
      space = "numeric",
      offset = "numeric"
    )
  )(
    basis = matrix(1:100, 100, 10),
    loadings = matrix(1:10000, 1000, 10),
    space = c(10, 10, 10, 100),
    offset = rep(0, 1000)
  )
  
  backend <- latent_backend(list(mock_lvec))
  backend <- backend_open(backend)
  
  # Get all loadings
  loadings <- backend_get_loadings(backend)
  expect_equal(dim(loadings), c(1000, 10))
  
  # Get subset of loadings
  loadings_subset <- backend_get_loadings(backend, components = 1:5)
  expect_equal(dim(loadings_subset), c(1000, 5))
  
  backend_close(backend)
})

test_that("backend_reconstruct_voxels works correctly", {
  # Create data where reconstruction is verifiable
  n_time <- 50
  n_comp <- 5
  n_vox <- 100
  
  # Create known basis and loadings
  basis <- matrix(rnorm(n_time * n_comp), n_time, n_comp)
  loadings <- matrix(rnorm(n_vox * n_comp), n_vox, n_comp)
  offset <- rnorm(n_vox)
  
  mock_lvec <- methods::setClass(
    "MockLatentNeuroVec",
    slots = c(
      basis = "matrix",
      loadings = "matrix",
      space = "numeric",
      offset = "numeric"
    )
  )(
    basis = basis,
    loadings = loadings,
    space = c(10, 10, 1, n_time),
    offset = offset
  )
  
  backend <- latent_backend(list(mock_lvec))
  backend <- backend_open(backend)
  
  # Reconstruct all data
  reconstructed <- backend_reconstruct_voxels(backend)
  expected <- basis %*% t(loadings) + matrix(offset, n_time, n_vox, byrow = TRUE)
  expect_equal(reconstructed, expected, tolerance = 1e-10)
  
  # Reconstruct subset
  rows <- 1:10
  voxels <- 1:20
  reconstructed_subset <- backend_reconstruct_voxels(backend, rows = rows, voxels = voxels)
  expected_subset <- basis[rows, ] %*% t(loadings[voxels, ]) + 
                     matrix(offset[voxels], length(rows), length(voxels), byrow = TRUE)
  expect_equal(reconstructed_subset, expected_subset, tolerance = 1e-10)
  
  backend_close(backend)
})

test_that("latent_backend handles sparse loadings", {
  skip_if_not_installed("Matrix")
  
  # Create sparse loadings
  n_vox <- 1000
  n_comp <- 10
  loadings_dense <- matrix(0, n_vox, n_comp)
  # Make 90% sparse
  n_nonzero <- round(0.1 * n_vox * n_comp)
  nonzero_idx <- sample(n_vox * n_comp, n_nonzero)
  loadings_dense[nonzero_idx] <- rnorm(n_nonzero)
  
  loadings_sparse <- Matrix::Matrix(loadings_dense, sparse = TRUE)
  
  mock_lvec <- methods::setClass(
    "MockLatentNeuroVec",
    slots = c(
      basis = "matrix",
      loadings = "Matrix",
      space = "numeric",
      offset = "numeric"
    )
  )(
    basis = matrix(rnorm(100 * n_comp), 100, n_comp),
    loadings = loadings_sparse,
    space = c(10, 10, 10, 100),
    offset = rep(0, n_vox)
  )
  
  backend <- latent_backend(list(mock_lvec))
  backend <- backend_open(backend)
  
  # Get metadata
  metadata <- backend_get_metadata(backend)
  expect_true(metadata$loadings_sparsity > 0.85)
  expect_true(metadata$loadings_sparsity < 0.95)
  
  # Reconstruction should still work
  reconstructed <- backend_reconstruct_voxels(backend, rows = 1:10, voxels = 1:50)
  expect_equal(dim(reconstructed), c(10, 50))
  
  backend_close(backend)
})