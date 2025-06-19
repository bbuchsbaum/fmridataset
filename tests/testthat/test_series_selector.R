library(testthat)

# Helper to create test dataset - simplified version
create_test_dataset <- function(n_voxels = 100, n_time = 20) {
  mat <- matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels)
  backend <- matrix_backend(mat, mask = rep(TRUE, n_voxels), spatial_dims = c(10, 10, 1))
  fmri_dataset(backend, TR = 2, run_length = n_time)
}

test_that("index_selector works correctly", {
  dset <- create_test_dataset()
  
  # Single index
  sel <- index_selector(1)
  expect_s3_class(sel, "index_selector")
  expect_s3_class(sel, "series_selector")
  
  indices <- resolve_indices(sel, dset)
  expect_equal(indices, 1L)
  
  # Multiple indices
  sel <- index_selector(c(1, 5, 10))
  indices <- resolve_indices(sel, dset)
  expect_equal(indices, c(1L, 5L, 10L))
  
  # Out of bounds should error
  n_voxels <- sum(backend_get_mask(dset$backend))
  sel <- index_selector(n_voxels + 1)
  expect_error(resolve_indices(sel, dset), "out-of-bounds")
})

test_that("voxel_selector works correctly", {
  dset <- create_test_dataset()
  
  # Single voxel
  sel <- voxel_selector(c(5, 5, 1))
  expect_s3_class(sel, "voxel_selector")
  
  indices <- resolve_indices(sel, dset)
  expect_length(indices, 1)
  
  # Multiple voxels
  coords <- cbind(x = c(1, 5, 10), y = c(1, 5, 10), z = c(1, 1, 1))
  sel <- voxel_selector(coords)
  indices <- resolve_indices(sel, dset)
  expect_length(indices, 3)
  
  # Out of bounds should error
  sel <- voxel_selector(c(15, 15, 10))
  expect_error(resolve_indices(sel, dset), "out-of-bounds")
})

test_that("roi_selector works correctly", {
  dset <- create_test_dataset()
  
  # Create ROI array matching our test dimensions
  roi <- array(FALSE, dim = c(10, 10, 1))
  roi[3:7, 3:7, 1] <- TRUE
  
  sel <- roi_selector(roi)
  expect_s3_class(sel, "roi_selector")
  
  indices <- resolve_indices(sel, dset)
  expect_true(length(indices) > 0)
  expect_true(length(indices) <= sum(roi))
  
  # Non-overlapping ROI should error
  roi_empty <- array(FALSE, dim = c(10, 10, 1))
  sel <- roi_selector(roi_empty)
  expect_error(resolve_indices(sel, dset), "does not overlap")
})

test_that("sphere_selector works correctly", {
  dset <- create_test_dataset()
  
  # Sphere in center of our 10x10x1 volume
  sel <- sphere_selector(center = c(5, 5, 1), radius = 3)
  expect_s3_class(sel, "sphere_selector")
  
  indices <- resolve_indices(sel, dset)
  expect_true(length(indices) > 0)
  
  # Very small radius
  sel <- sphere_selector(center = c(5, 5, 1), radius = 0.5)
  indices <- resolve_indices(sel, dset)
  expect_equal(length(indices), 1)  # Only center voxel
  
  # Sphere outside volume should error with overlap
  sel <- sphere_selector(center = c(50, 50, 50), radius = 1)
  expect_error(resolve_indices(sel, dset), "does not overlap")
})

test_that("all_selector works correctly", {
  dset <- create_test_dataset()
  
  sel <- all_selector()
  expect_s3_class(sel, "all_selector")
  
  indices <- resolve_indices(sel, dset)
  n_voxels <- sum(backend_get_mask(dset$backend))
  expect_equal(indices, seq_len(n_voxels))
})

test_that("mask_selector works correctly", {
  dset <- create_test_dataset()
  
  # Logical vector in masked space
  n_masked <- sum(backend_get_mask(dset$backend))
  mask_vec <- rep(FALSE, n_masked)
  mask_vec[1:10] <- TRUE
  
  sel <- mask_selector(mask_vec)
  expect_s3_class(sel, "mask_selector")
  
  indices <- resolve_indices(sel, dset)
  expect_equal(indices, 1:10)
  
  # 3D logical array matching our dimensions
  mask_3d <- array(FALSE, dim = c(10, 10, 1))
  mask_3d[1:5, 1:5, 1] <- TRUE
  
  sel <- mask_selector(mask_3d)
  indices <- resolve_indices(sel, dset)
  expect_true(length(indices) > 0)
  
  # Wrong size should error
  bad_mask <- rep(FALSE, 50)  # Wrong size
  sel <- mask_selector(bad_mask)
  expect_error(resolve_indices(sel, dset), "does not match")
})

test_that("selectors integrate with fmri_series", {
  dset <- create_test_dataset()
  
  # Test each selector type
  fs1 <- fmri_series(dset, selector = index_selector(1:5))
  expect_s4_class(fs1, "FmriSeries")
  expect_equal(ncol(fs1), 5)
  
  fs2 <- fmri_series(dset, selector = voxel_selector(cbind(5, 5, 1)))
  expect_equal(ncol(fs2), 1)
  
  fs3 <- fmri_series(dset, selector = all_selector())
  expect_equal(ncol(fs3), sum(backend_get_mask(dset$backend)))
  
  # Compare with legacy selector
  legacy_fs <- fmri_series(dset, selector = 1:5)
  new_fs <- fmri_series(dset, selector = index_selector(1:5))
  expect_equal(as.matrix(legacy_fs), as.matrix(new_fs))
})

test_that("print methods work", {
  # index_selector - short list
  sel1a <- index_selector(1:5)
  output1a <- capture.output(print(sel1a))
  expect_match(output1a[1], "<index_selector>")
  expect_match(output1a[2], "indices: 1, 2, 3, 4, 5")
  
  # index_selector - long list should truncate
  sel1b <- index_selector(1:20)
  output1b <- capture.output(print(sel1b))
  expect_match(output1b[1], "<index_selector>")
  expect_match(output1b[2], "\\.\\.\\.")  # Should truncate long lists
  expect_match(output1b[2], "20 total")
  
  # voxel_selector
  sel2 <- voxel_selector(cbind(1:10, 1:10, rep(5, 10)))
  output2 <- capture.output(print(sel2))
  expect_match(output2[1], "<voxel_selector>")
  expect_match(output2[2], "10 voxel")
  
  # sphere_selector
  sel3 <- sphere_selector(center = c(10, 10, 5), radius = 3.5)
  output3 <- capture.output(print(sel3))
  expect_match(output3[1], "<sphere_selector>")
  expect_match(output3[3], "3.5")
  
  # all_selector
  sel4 <- all_selector()
  output4 <- capture.output(print(sel4))
  expect_match(output4[1], "<all_selector>")
  
  # roi_selector
  roi <- array(FALSE, dim = c(10, 10, 1))
  roi[1:5, 1:5, 1] <- TRUE
  sel5 <- roi_selector(roi)
  output5 <- capture.output(print(sel5))
  expect_match(output5[1], "<roi_selector>")
  expect_true(any(grepl(as.character(sum(roi)), output5)))
  
  # mask_selector
  sel6 <- mask_selector(rep(c(TRUE, FALSE), 50))
  output6 <- capture.output(print(sel6))
  expect_match(output6[1], "<mask_selector>")
  expect_true(any(grepl("50", output6)))
})

test_that("error messages are informative", {
  dset <- create_test_dataset()
  
  # Index out of bounds
  sel <- index_selector(1000)
  err <- tryCatch(resolve_indices(sel, dset), error = function(e) e)
  expect_match(err$message, "Dataset has")
  expect_match(err$message, "voxels")
  
  # Coordinates out of bounds
  sel <- voxel_selector(c(20, 20, 20))
  err <- tryCatch(resolve_indices(sel, dset), error = function(e) e)
  expect_match(err$message, "Volume dimensions are")
  
  # Invalid selector type in legacy function
  err <- tryCatch(resolve_selector(dset, list(a = 1)), error = function(e) e)
  expect_match(err$message, "Unsupported selector type")
})