library(testthat)
library(fmridataset)

# Helper to create test dataset
create_test_dataset <- function(spatial_dims = c(4, 5, 2), n_time = 10) {
  n_voxels <- prod(spatial_dims)
  mat <- matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels)
  backend <- matrix_backend(mat, spatial_dims = spatial_dims)
  fmri_dataset(backend, TR = 2, run_length = n_time)
}

# Helper to create partial mask dataset
create_masked_dataset <- function() {
  spatial_dims <- c(4, 4, 2)
  n_voxels <- prod(spatial_dims)
  mat <- matrix(rnorm(10 * 20), nrow = 10, ncol = 20)  # Only 20 voxels active
  
  # Create mask with only some voxels active
  mask <- rep(FALSE, n_voxels)
  mask[c(1:5, 10:15, 20:25, 30:32)] <- TRUE  # 20 active voxels
  
  backend <- matrix_backend(mat, mask = mask, spatial_dims = spatial_dims)
  fmri_dataset(backend, TR = 2, run_length = 10)
}

test_that("resolve_selector handles NULL selector (all voxels)", {
  dset <- create_test_dataset()
  indices <- fmridataset:::resolve_selector(dset, NULL)
  
  n_voxels <- sum(backend_get_mask(dset$backend))
  expect_equal(indices, seq_len(n_voxels))
  expect_equal(length(indices), n_voxels)
})

test_that("resolve_selector handles numeric indices", {
  dset <- create_test_dataset()
  
  # Single index
  indices <- fmridataset:::resolve_selector(dset, 5)
  expect_equal(indices, 5L)
  
  # Multiple indices
  indices <- fmridataset:::resolve_selector(dset, c(1, 3, 5, 10))
  expect_equal(indices, c(1L, 3L, 5L, 10L))
  
  # Vector of consecutive indices
  indices <- fmridataset:::resolve_selector(dset, 1:10)
  expect_equal(indices, 1:10)
})

test_that("resolve_selector handles 3-column coordinate matrices", {
  dset <- create_test_dataset(spatial_dims = c(4, 5, 2))
  
  # Single coordinate (as vector)
  coords <- c(2, 3, 1)  # x=2, y=3, z=1
  indices <- fmridataset:::resolve_selector(dset, coords)
  expect_length(indices, 1)
  expect_true(is.integer(indices))
  
  # Multiple coordinates (as matrix)
  coords <- cbind(
    x = c(1, 2, 3),
    y = c(1, 2, 3), 
    z = c(1, 1, 2)
  )
  indices <- fmridataset:::resolve_selector(dset, coords)
  expect_length(indices, 3)
  expect_true(is.integer(indices))
  expect_true(all(indices > 0))
})

test_that("resolve_selector handles coordinate mapping with partial mask", {
  dset <- create_masked_dataset()
  
  # Test coordinate that should be in mask
  coords <- c(1, 1, 1)  # Should correspond to linear index 1
  indices <- fmridataset:::resolve_selector(dset, coords)
  expect_length(indices, 1)
  expect_true(indices > 0)
  
  # Test coordinates that map to masked-out voxels
  coords <- c(4, 4, 2)  # Should be outside active mask
  indices <- fmridataset:::resolve_selector(dset, coords)
  # Should return empty or NA depending on implementation
  expect_true(length(indices) == 0 || is.na(indices))
})

test_that("resolve_selector handles logical arrays", {
  dset <- create_test_dataset()
  
  # Create logical array matching spatial dimensions
  dims <- backend_get_dims(dset$backend)$spatial
  logical_array <- array(FALSE, dims)
  logical_array[1:2, 1:2, 1] <- TRUE  # Select some voxels
  
  indices <- fmridataset:::resolve_selector(dset, logical_array)
  expect_true(length(indices) > 0)
  expect_true(is.integer(indices))
  expect_true(all(indices > 0))
})

test_that("resolve_selector handles series_selector objects", {
  dset <- create_test_dataset()
  
  # Test index_selector
  sel <- index_selector(c(1, 5, 10))
  indices <- fmridataset:::resolve_selector(dset, sel)
  expect_equal(indices, c(1L, 5L, 10L))
  
  # Test all_selector
  sel <- all_selector()
  indices <- fmridataset:::resolve_selector(dset, sel)
  n_voxels <- sum(backend_get_mask(dset$backend))
  expect_equal(indices, seq_len(n_voxels))
})

test_that("resolve_selector throws errors for unsupported types", {
  dset <- create_test_dataset()
  
  # Test unsupported object type
  expect_error(
    fmridataset:::resolve_selector(dset, list(a = 1, b = 2)),
    "Unsupported selector type"
  )
  
  # Test character vector (unsupported)
  expect_error(
    fmridataset:::resolve_selector(dset, c("voxel1", "voxel2")),
    "Unsupported selector type"
  )
})

test_that("resolve_timepoints handles NULL (all timepoints)", {
  dset <- create_test_dataset(n_time = 15)
  timepoints <- fmridataset:::resolve_timepoints(dset, NULL)
  
  expect_equal(timepoints, 1:15)
  expect_equal(length(timepoints), 15)
})

test_that("resolve_timepoints handles numeric vectors", {
  dset <- create_test_dataset()
  
  # Specific timepoints
  timepoints <- fmridataset:::resolve_timepoints(dset, c(2, 5, 8))
  expect_equal(timepoints, c(2L, 5L, 8L))
  
  # Range of timepoints
  timepoints <- fmridataset:::resolve_timepoints(dset, 3:7)
  expect_equal(timepoints, 3:7)
  
  # Single timepoint
  timepoints <- fmridataset:::resolve_timepoints(dset, 5)
  expect_equal(timepoints, 5L)
})

test_that("resolve_timepoints handles logical vectors", {
  dset <- create_test_dataset(n_time = 8)
  
  # Logical vector selecting some timepoints
  logical_tp <- c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE)
  timepoints <- fmridataset:::resolve_timepoints(dset, logical_tp)
  expect_equal(timepoints, c(1L, 3L, 5L, 8L))
  
  # All FALSE should return empty
  logical_tp <- rep(FALSE, 8)
  timepoints <- fmridataset:::resolve_timepoints(dset, logical_tp)
  expect_equal(timepoints, integer(0))
  
  # All TRUE should return all indices
  logical_tp <- rep(TRUE, 8)
  timepoints <- fmridataset:::resolve_timepoints(dset, logical_tp)
  expect_equal(timepoints, 1:8)
})

test_that("resolve_timepoints validates logical vector length", {
  dset <- create_test_dataset(n_time = 10)
  
  # Wrong length logical vector
  logical_tp <- c(TRUE, FALSE, TRUE)  # Only 3 elements for 10 timepoints
  expect_error(
    fmridataset:::resolve_timepoints(dset, logical_tp),
    "Logical timepoints length must equal number of timepoints"
  )
})

test_that("resolve_timepoints throws errors for unsupported types", {
  dset <- create_test_dataset()
  
  # Character vector
  expect_error(
    fmridataset:::resolve_timepoints(dset, c("time1", "time2")),
    "Unsupported timepoints type"
  )
  
  # List
  expect_error(
    fmridataset:::resolve_timepoints(dset, list(1, 2, 3)),
    "Unsupported timepoints type"
  )
})

test_that("all_timepoints returns correct sequence", {
  # Test with different dataset sizes
  dset1 <- create_test_dataset(n_time = 10)
  timepoints1 <- fmridataset:::all_timepoints(dset1)
  expect_equal(timepoints1, 1:10)
  
  dset2 <- create_test_dataset(n_time = 25)
  timepoints2 <- fmridataset:::all_timepoints(dset2)
  expect_equal(timepoints2, 1:25)
  
  # Single timepoint dataset
  dset3 <- create_test_dataset(n_time = 1)
  timepoints3 <- fmridataset:::all_timepoints(dset3)
  expect_equal(timepoints3, 1L)
})

test_that("resolvers work together in fmri_series integration", {
  dset <- create_test_dataset()
  
  # Test coordinate selection with timepoint selection
  coords <- cbind(x = c(1, 2), y = c(1, 2), z = c(1, 1))
  timepoints <- c(2, 4, 6)
  
  # This should work without errors
  fs <- fmri_series(dset, selector = coords, timepoints = timepoints)
  expect_s4_class(fs, "FmriSeries")
  expect_equal(nrow(fs), 3)  # 3 timepoints
  expect_equal(ncol(fs), 2)  # 2 voxels
})

test_that("resolvers handle edge cases gracefully", {
  dset <- create_test_dataset()
  
  # Empty numeric selector
  indices <- fmridataset:::resolve_selector(dset, numeric(0))
  expect_equal(indices, integer(0))
  
  # Empty timepoints
  timepoints <- fmridataset:::resolve_timepoints(dset, integer(0))
  expect_equal(timepoints, integer(0))
  
  # Single element vectors
  indices <- fmridataset:::resolve_selector(dset, 1)
  expect_equal(indices, 1L)
  
  timepoints <- fmridataset:::resolve_timepoints(dset, 5)
  expect_equal(timepoints, 5L)
})