# Test dummy mode functionality for non-existent files

test_that("dummy mode allows non-existent files", {
  # These files don't exist but should work with dummy_mode = TRUE
  dummy_files <- c("dummy1.nii", "dummy2.nii", "dummy3.nii")
  dummy_mask <- "dummy_mask.nii"
  
  # Should work with dummy_mode = TRUE
  dset <- fmri_dataset(
    scans = dummy_files,
    mask = dummy_mask,
    TR = 2,
    run_length = c(100, 100, 100),
    dummy_mode = TRUE
  )
  
  expect_s3_class(dset, "fmri_dataset")
  expect_s3_class(dset$backend, "nifti_backend")
  expect_true(dset$backend$dummy_mode)
})

test_that("dummy mode returns correct dimensions", {
  dummy_files <- c("scan1.nii", "scan2.nii")
  dummy_mask <- "mask.nii"
  
  dset <- fmri_dataset(
    scans = dummy_files,
    mask = dummy_mask,
    TR = 2,
    run_length = c(100, 100),
    dummy_mode = TRUE
  )
  
  # Check dimensions
  dims <- backend_get_dims(dset$backend)
  expect_equal(dims$spatial, c(64L, 64L, 30L))
  expect_equal(dims$time, 200L)  # 2 files * 100 timepoints each
  
  # Check TR and run info
  expect_equal(get_TR(dset), 2)
  expect_equal(n_runs(dset), 2)
  expect_equal(n_timepoints(dset), 200)
})

test_that("dummy mode returns placeholder data", {
  dummy_files <- "single_scan.nii"
  dummy_mask <- "mask.nii"
  
  dset <- fmri_dataset(
    scans = dummy_files,
    mask = dummy_mask,
    TR = 1.5,
    run_length = 100,  # Changed to match default 100 timepoints per file
    dummy_mode = TRUE
  )
  
  # Get data matrix
  data_mat <- get_data_matrix(dset)
  
  # Check dimensions
  expect_equal(nrow(data_mat), 100)  # timepoints
  expect_true(ncol(data_mat) > 0)    # voxels
  
  # Data should be all zeros
  expect_true(all(data_mat == 0))
})

test_that("dummy mode returns correct mask", {
  dummy_files <- "scan.nii"
  dummy_mask <- "mask.nii"
  
  backend <- nifti_backend(
    source = dummy_files,
    mask_source = dummy_mask,
    dummy_mode = TRUE
  )
  backend <- backend_open(backend)
  
  mask <- backend_get_mask(backend)
  
  # Should return all TRUE by default
  expect_type(mask, "logical")
  expect_equal(length(mask), 64 * 64 * 30)  # prod(spatial_dims)
  expect_true(all(mask))
})

test_that("dummy mode returns placeholder metadata", {
  dummy_files <- "scan.nii"
  dummy_mask <- "mask.nii"
  
  backend <- nifti_backend(
    source = dummy_files,
    mask_source = dummy_mask,
    dummy_mode = TRUE
  )
  backend <- backend_open(backend)
  
  metadata <- backend_get_metadata(backend)
  
  expect_type(metadata, "list")
  expect_true("affine" %in% names(metadata))
  expect_true("voxel_dims" %in% names(metadata))
  expect_true("space" %in% names(metadata))
  expect_true("origin" %in% names(metadata))
  expect_true("dims" %in% names(metadata))
  
  # Check voxel dimensions (should be 3mm isotropic)
  expect_equal(metadata$voxel_dims, c(3, 3, 3))
  expect_equal(metadata$origin, c(0, 0, 0))
  expect_equal(metadata$dims, c(64, 64, 30, 100))
})

test_that("dummy mode works with subsetting", {
  dummy_files <- c("scan1.nii", "scan2.nii")
  dummy_mask <- "mask.nii"
  
  backend <- nifti_backend(
    source = dummy_files,
    mask_source = dummy_mask,
    dummy_mode = TRUE
  )
  backend <- backend_open(backend)
  
  # Test row subsetting
  data_subset <- backend_get_data(backend, rows = 1:10)
  expect_equal(nrow(data_subset), 10)
  
  # Test column subsetting
  data_subset <- backend_get_data(backend, cols = 1:50)
  expect_equal(ncol(data_subset), 50)
  
  # Test both
  data_subset <- backend_get_data(backend, rows = 1:10, cols = 1:50)
  expect_equal(dim(data_subset), c(10, 50))
})

test_that("dummy mode fails when disabled for non-existent files", {
  # These files don't exist
  dummy_files <- c("nonexistent1.nii", "nonexistent2.nii")
  dummy_mask <- "nonexistent_mask.nii"
  
  # Should fail with dummy_mode = FALSE (default)
  expect_error(
    fmri_dataset(
      scans = dummy_files,
      mask = dummy_mask,
      TR = 2,
      run_length = c(100, 100),
      dummy_mode = FALSE
    ),
    "Source files not found"
  )
})

test_that("dummy mode integrates with existing dataset functions", {
  dummy_files <- c("run1.nii", "run2.nii", "run3.nii")
  dummy_mask <- "brain_mask.nii"
  
  dset <- fmri_dataset(
    scans = dummy_files,
    mask = dummy_mask,
    TR = 2.5,
    run_length = c(100, 100, 100),  # Changed to match default 100 timepoints per file
    event_table = data.frame(
      onset = c(10, 50, 100, 150, 200, 250),
      duration = rep(5, 6),
      run = c(1, 1, 2, 2, 3, 3)
    ),
    dummy_mode = TRUE
  )
  
  # Test various dataset functions
  expect_equal(get_TR(dset), 2.5)
  expect_equal(n_runs(dset), 3)
  expect_equal(n_timepoints(dset), 300)  # Changed to 300
  expect_equal(get_run_lengths(dset), c(100, 100, 100))  # Changed to 100s
  
  # Event table should be preserved
  expect_equal(nrow(dset$event_table), 6)
  
  # Data access should work
  all_data <- get_data_matrix(dset)
  expect_equal(dim(all_data)[1], 300)  # Changed to 300
  
  # Get mask should work
  mask <- get_mask(dset)
  expect_type(mask, "logical")
})

test_that("dummy mode works with study datasets", {
  # Create multiple dummy datasets
  datasets <- list()
  
  for (i in 1:3) {
    datasets[[i]] <- fmri_dataset(
      scans = paste0("subject", i, ".nii"),
      mask = paste0("mask", i, ".nii"),
      TR = 2,
      run_length = 100,
      dummy_mode = TRUE
    )
  }
  
  # Create study dataset
  study_dset <- fmri_study_dataset(
    datasets = datasets,
    subject_ids = paste0("sub-", sprintf("%02d", 1:3))
  )
  
  expect_s3_class(study_dset, "fmri_study_dataset")
  expect_equal(length(study_dset$subject_ids), 3)
})

test_that("dummy mode preserves file paths", {
  dummy_files <- c("path/to/scan1.nii", "path/to/scan2.nii")
  dummy_mask <- "path/to/mask.nii"
  
  backend <- nifti_backend(
    source = dummy_files,
    mask_source = dummy_mask,
    dummy_mode = TRUE
  )
  
  # File paths should be preserved
  expect_equal(backend$source, dummy_files)
  expect_equal(backend$mask_source, dummy_mask)
})