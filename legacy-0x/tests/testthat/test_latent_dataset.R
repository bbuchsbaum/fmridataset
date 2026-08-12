test_that("latent_dataset creates proper objects", {
  skip_if_not_installed("fmristore")

  # Define mock class once
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  # Create mock LatentNeuroVec objects
  create_mock_lvec <- function(n_time = 100, n_comp = 10, n_voxels = 1000) {
    basis <- matrix(rnorm(n_time * n_comp), n_time, n_comp)
    loadings <- matrix(rnorm(n_voxels * n_comp), n_voxels, n_comp)
    offset <- rnorm(n_voxels)

    # Mock space object
    space <- structure(
      c(10, 10, 10, n_time), # dims
      class = "mock_space"
    )

    methods::new("mock_LatentNeuroVec",
      basis = basis,
      loadings = loadings,
      offset = offset,
      mask = array(TRUE, c(10, 10, 10)),
      space = space
    )
  }

  # Create test objects
  lvec1 <- create_mock_lvec(100, 10, 1000)
  lvec2 <- create_mock_lvec(100, 10, 1000)

  # Test creation
  dataset <- latent_dataset(
    source = list(lvec1, lvec2),
    TR = 2,
    run_length = c(100, 100)
  )

  expect_s3_class(dataset, "latent_dataset")
  expect_s3_class(dataset, "fmri_dataset")

  # Test dimensions
  expect_equal(n_timepoints(dataset), 200)
  expect_equal(n_runs(dataset), 2)
  expect_equal(get_TR(dataset), 2)
})

test_that("get_latent_scores returns correct dimensions", {
  skip_if_not_installed("fmristore")

  # Define mock class if needed
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  # Create mock data
  create_mock_lvec <- function(n_time = 100, n_comp = 10, n_voxels = 1000) {
    basis <- matrix(seq_len(n_time * n_comp), n_time, n_comp)
    loadings <- matrix(rnorm(n_voxels * n_comp), n_voxels, n_comp)

    space <- structure(
      c(10, 10, 10, n_time),
      class = "mock_space"
    )

    methods::new("mock_LatentNeuroVec",
      basis = basis,
      loadings = loadings,
      offset = numeric(0),
      mask = array(TRUE, c(10, 10, 10)),
      space = space
    )
  }

  lvec <- create_mock_lvec(50, 5, 1000)

  dataset <- latent_dataset(
    source = list(lvec),
    TR = 2,
    run_length = 50
  )

  # Get all scores
  scores <- get_latent_scores(dataset)
  expect_equal(dim(scores), c(50, 5))

  # Get subset
  scores_subset <- get_latent_scores(dataset, rows = 1:10, cols = 1:3)
  expect_equal(dim(scores_subset), c(10, 3))
})

test_that("get_spatial_loadings works correctly", {
  skip_if_not_installed("fmristore")

  # Define mock class if needed
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  # Create mock data
  n_voxels <- 1000
  n_comp <- 5
  loadings_mat <- matrix(rnorm(n_voxels * n_comp), n_voxels, n_comp)

  lvec <- methods::new("mock_LatentNeuroVec",
    basis = matrix(rnorm(100 * n_comp), 100, n_comp),
    loadings = loadings_mat,
    offset = numeric(0),
    mask = array(TRUE, c(10, 10, 10)),
    space = structure(c(10, 10, 10, 100), class = "mock_space")
  )

  dataset <- latent_dataset(
    source = list(lvec),
    TR = 2,
    run_length = 100
  )

  loadings <- get_spatial_loadings(dataset)
  expect_equal(dim(loadings), c(n_voxels, n_comp))
  expect_equal(loadings, loadings_mat)

  # Get subset of components
  loadings_subset <- get_spatial_loadings(dataset, components = 1:3)
  expect_equal(dim(loadings_subset), c(n_voxels, 3))
})

test_that("reconstruct_voxels works", {
  skip_if_not_installed("fmristore")

  # Define mock class if needed
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  # Create simple data for easy verification
  n_time <- 10
  n_comp <- 2
  n_voxels <- 5

  # Simple patterns
  basis <- matrix(c(1:10, 11:20), n_time, n_comp)
  loadings <- matrix(c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0), n_voxels, n_comp)

  lvec <- methods::new("mock_LatentNeuroVec",
    basis = basis,
    loadings = loadings,
    offset = numeric(0),
    mask = array(TRUE, c(5, 1, 1)),
    space = structure(c(5, 1, 1, n_time), class = "mock_space")
  )

  dataset <- latent_dataset(
    source = list(lvec),
    TR = 2,
    run_length = n_time
  )

  # Reconstruct all
  recon <- reconstruct_voxels(dataset)
  expect_equal(dim(recon), c(n_time, n_voxels))

  # Check reconstruction: data = basis %*% t(loadings)
  expected <- basis %*% t(loadings)
  expect_equal(recon, expected)
})

test_that("fmri_latent_dataset deprecation works", {
  skip_if_not_installed("fmristore")

  # Define mock class if needed
  if (!isClass("mock_LatentNeuroVec")) {
    setClass("mock_LatentNeuroVec",
      slots = c(
        basis = "matrix", loadings = "matrix", offset = "numeric",
        mask = "array", space = "ANY"
      )
    )
  }

  # Create mock object
  lvec <- methods::new("mock_LatentNeuroVec",
    basis = matrix(rnorm(100 * 5), 100, 5),
    loadings = matrix(rnorm(1000 * 5), 1000, 5),
    offset = numeric(0),
    mask = array(TRUE, c(10, 10, 10)),
    space = structure(c(10, 10, 10, 100), class = "mock_space")
  )

  expect_warning(
    result <- fmri_latent_dataset(
      latent_files = list(lvec),
      TR = 2,
      run_length = 100
    ),
    class = "lifecycle_warning_deprecated"
  )
})
