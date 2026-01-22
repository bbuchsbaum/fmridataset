test_that("fmri_zarr_dataset creates valid dataset", {
  skip_if_not_installed("zarr")

  # Create test data
  arr <- array(rnorm(2 * 2 * 2 * 10), dim = c(2, 2, 2, 10))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

  # Create dataset using constructor
  dataset <- fmri_zarr_dataset(
    tmp_dir,
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

  # Test mask (should be all TRUE)
  mask <- get_mask(dataset)
  expect_equal(length(mask), 8)
  expect_true(all(mask))
})

test_that("fmri_zarr_dataset works without mask", {
  skip_if_not_installed("zarr")

  # Create test data
  arr <- array(rnorm(2 * 1 * 1 * 4), dim = c(2, 1, 1, 4))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

  dataset <- fmri_zarr_dataset(
    tmp_dir,
    TR = 2,
    run_length = 4
  )

  # Should work with default mask (all TRUE)
  mask <- get_mask(dataset)
  expect_true(all(mask))
  expect_equal(length(mask), 2)
})

test_that("fmri_zarr_dataset validates run_length", {
  skip_if_not_installed("zarr")

  # Create test data with only 2 timepoints
  arr <- array(rnorm(2 * 2 * 2 * 2), dim = c(2, 2, 2, 2))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

  # Run length doesn't match time dimension
  expect_error(
    fmri_zarr_dataset(
      tmp_dir,
      TR = 2,
      run_length = 10 # But data only has 2 timepoints
    ),
    "Sum of run_length"
  )
})

test_that("fmri_zarr_dataset handles preload option", {
  skip_if_not_installed("zarr")

  # Create test data
  arr <- array(rnorm(2 * 2 * 1 * 5), dim = c(2, 2, 1, 5))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

  # Create with preload
  expect_message(
    dataset <- fmri_zarr_dataset(
      tmp_dir,
      TR = 2,
      run_length = 5,
      preload = TRUE
    ),
    "Preloading Zarr data"
  )

  expect_s3_class(dataset, "fmri_dataset")
  expect_false(is.null(dataset$backend$data_cache))
})

test_that("fmri_zarr_dataset works with event_table and censor", {
  skip_if_not_installed("zarr")

  # Create test data
  arr <- array(rnorm(2 * 2 * 1 * 6), dim = c(2, 2, 1, 6))

  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  z <- zarr::as_zarr(arr)
  z$save(tmp_dir)

  # Create event table
  events <- data.frame(
    onset = c(0, 4),
    condition = c("A", "B")
  )

  # Create censor vector
  censor <- c(rep(1, 5), 0)

  dataset <- fmri_zarr_dataset(
    tmp_dir,
    TR = 2,
    run_length = 6,
    event_table = events,
    censor = censor
  )

  expect_s3_class(dataset, "fmri_dataset")
  expect_equal(nrow(dataset$event_table), 2)
})
