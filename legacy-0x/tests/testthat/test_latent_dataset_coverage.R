# Tests for R/latent_dataset.R methods - coverage improvement

# Define mock S4 class for testing
methods::setClass(
  "MockLatentNeuroVecLD",
  slots = c(
    basis = "matrix",
    loadings = "matrix",
    space = "numeric",
    offset = "numeric",
    mask = "logical"
  )
)

# Helper to create a mock latent dataset
make_mock_latent_dataset <- function(n_time = 100, n_comp = 10, n_vox = 500) {
  mock_lvec <- methods::new(
    "MockLatentNeuroVecLD",
    basis = matrix(rnorm(n_time * n_comp), n_time, n_comp),
    loadings = matrix(rnorm(n_vox * n_comp), n_vox, n_comp),
    space = c(10, 10, 5, n_time),
    offset = rnorm(n_vox),
    mask = rep(TRUE, n_vox)
  )

  latent_dataset(
    source = list(mock_lvec),
    TR = 2,
    run_length = n_time
  )
}

test_that("latent_dataset constructs valid object", {
  ds <- make_mock_latent_dataset()
  expect_s3_class(ds, "latent_dataset")
  expect_s3_class(ds, "fmri_dataset")
})

test_that("get_latent_scores.latent_dataset works", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  scores <- get_latent_scores(ds)
  expect_equal(dim(scores), c(50, 5))
})

test_that("get_latent_scores.latent_dataset with row/col subsetting", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)

  scores <- get_latent_scores(ds, rows = 1:10, cols = 1:3)
  expect_equal(dim(scores), c(10, 3))
})

test_that("get_spatial_loadings.latent_dataset works", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5, n_vox = 200)
  loadings <- get_spatial_loadings(ds)
  expect_equal(dim(loadings), c(200, 5))
})

test_that("get_spatial_loadings.latent_dataset with component subset", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5, n_vox = 200)
  loadings <- get_spatial_loadings(ds, components = 1:3)
  expect_equal(dim(loadings), c(200, 3))
})

test_that("get_component_info.latent_dataset returns metadata", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  info <- get_component_info(ds)

  expect_type(info, "list")
  expect_equal(info$storage_format, "latent")
  expect_equal(info$n_components, 5)
})

test_that("reconstruct_voxels.latent_dataset works", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5, n_vox = 100)
  reconstructed <- reconstruct_voxels(ds)
  expect_equal(dim(reconstructed), c(50, 100))
})

test_that("reconstruct_voxels.latent_dataset with subset", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5, n_vox = 100)
  reconstructed <- reconstruct_voxels(ds, rows = 1:10, voxels = 1:20)
  expect_equal(dim(reconstructed), c(10, 20))
})

test_that("print.latent_dataset shows info", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5, n_vox = 200)
  out <- capture.output(print(ds))
  expect_true(any(grepl("Latent Dataset", out)))
  expect_true(any(grepl("Components", out)))
  expect_true(any(grepl("timepoints", out)))
})

test_that("get_data.latent_dataset warns", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  expect_warning(get_data(ds), "latent scores")
})

test_that("get_data_matrix.latent_dataset returns scores", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  result <- get_data_matrix(ds)
  expect_equal(dim(result), c(50, 5))
})

test_that("get_mask.latent_dataset returns component mask", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  mask <- get_mask(ds)
  expect_type(mask, "logical")
  expect_length(mask, 5)
  expect_true(all(mask))
})

test_that("blocklens.latent_dataset works", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  bl <- blocklens(ds)
  expect_equal(bl, 50)
})

test_that("get_TR.latent_dataset returns TR", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  tr <- get_TR(ds)
  expect_equal(tr, 2)
})

test_that("n_runs.latent_dataset returns number of runs", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  nr <- n_runs(ds)
  expect_equal(nr, 1)
})

test_that("n_timepoints.latent_dataset returns total timepoints", {
  ds <- make_mock_latent_dataset(n_time = 50, n_comp = 5)
  nt <- n_timepoints(ds)
  expect_equal(nt, 50)
})

test_that("latent_dataset with multiple runs", {
  mock1 <- methods::new(
    "MockLatentNeuroVecLD",
    basis = matrix(rnorm(100 * 5), 100, 5),
    loadings = matrix(rnorm(200 * 5), 200, 5),
    space = c(10, 10, 2, 100),
    offset = rnorm(200),
    mask = rep(TRUE, 200)
  )

  mock2 <- methods::new(
    "MockLatentNeuroVecLD",
    basis = matrix(rnorm(80 * 5), 80, 5),
    loadings = matrix(rnorm(200 * 5), 200, 5),
    space = c(10, 10, 2, 80),
    offset = rnorm(200),
    mask = rep(TRUE, 200)
  )

  ds <- latent_dataset(
    source = list(mock1, mock2),
    TR = 2,
    run_length = c(100, 80)
  )

  expect_s3_class(ds, "latent_dataset")
  expect_equal(n_runs(ds), 2)
  expect_equal(n_timepoints(ds), 180)

  # Get data spanning both runs
  scores <- get_latent_scores(ds, rows = c(50, 150))
  expect_equal(dim(scores), c(2, 5))
})

test_that("latent_dataset source path processing", {
  # Test with absolute path (non-existent, will fail at backend creation)
  expect_error(
    latent_dataset(source = "/absolute/path.lv.h5", TR = 2, run_length = 10),
    class = "fmridataset_error"
  )
})
