# Legacy Dataset Tests from fmrireg
# These tests ensure backward compatibility with the original fmrireg dataset creation interface

test_that("can construct an fmri_dataset from file paths", {
  # Test file-based dataset construction (equivalent to old fmri_dataset function)
  # Using non-existent files for structure testing only
  
  # This would be the old way:
  # dset <- fmri_dataset(
  #   scans=c("scan1.nii", "scan2.nii", "scan3.nii"),
  #   mask="mask.nii",
  #   run_length=c(100,100,100),
  #   TR=2
  # )
  
  # Skip this test if files don't exist - we're testing structure, not file loading
  skip("Legacy file-based test requires actual NIfTI files")
  
  # If we had the files, this would be the new equivalent:
  # dset <- fmri_dataset_create(
  #   images = c("scan1.nii", "scan2.nii", "scan3.nii"),
  #   mask = "mask.nii", 
  #   run_lengths = c(100, 100, 100),
  #   TR = 2
  # )
  # expect_true(!is.null(dset))
  # expect_true(is.fmri_dataset(dset))
})

test_that("can construct an fmri_mem_dataset equivalent", {
  # Test memory-based dataset construction (equivalent to old fmri_mem_dataset function)
  
  # Create synthetic data to simulate what fmrireg would have done
  set.seed(123)
  
  # Create event table similar to fmrireg's face design
  event_table <- data.frame(
    onset = seq(10, 100, 20),
    duration = rep(2, 5),
    trial_type = rep(c("face", "house"), length.out = 5),
    run = rep(1:2, length.out = 5),
    rep_num = 1:5
  )
  
  # For our package, we use matrices instead of NeuroVec objects
  # Simulate 3 runs with different lengths that match the event structure
  test_matrix1 <- matrix(rnorm(244 * 100), nrow = 244, ncol = 100)  # Run 1: 244 timepoints
  test_matrix2 <- matrix(rnorm(244 * 100), nrow = 244, ncol = 100)  # Run 2: 244 timepoints  
  test_matrix3 <- matrix(rnorm(244 * 100), nrow = 244, ncol = 100)  # Run 3: 244 timepoints
  
  # Combine into single matrix for our interface
  combined_matrix <- rbind(test_matrix1, test_matrix2, test_matrix3)
  
  # Create mask (equivalent to LogicalNeuroVol)
  mask_vector <- runif(100) > 0.3  # ~70% of voxels included
  
  # This would be the old way:
  # dset <- fmri_mem_dataset(scans=scans, 
  #                          mask=mask, 
  #                          TR=1.5, 
  #                          event_table=tibble::as_tibble(facedes))
  
  # New way with our interface:
  dset <- fmri_dataset_create(
    images = combined_matrix,
    mask = mask_vector,
    TR = 1.5,
    run_lengths = c(244, 244, 244),
    event_table = event_table
  )
  
  expect_true(!is.null(dset))
  expect_true(is.fmri_dataset(dset))
  expect_equal(get_dataset_type(dset), "matrix")
  expect_equal(n_runs(dset$sampling_frame), 3)
  expect_equal(get_TR(dset$sampling_frame)[1], 1.5)
  expect_equal(nrow(get_event_table(dset)), 5)
})

test_that("legacy dataset interface compatibility", {
  # Test that our new interface can handle fmrireg-style parameters
  
  set.seed(456)
  test_matrix <- matrix(rnorm(1000), nrow = 200, ncol = 5)
  
  # Test with fmrireg-style parameter names and structure
  dset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(100, 100),  # equivalent to old run_length parameter
    base_path = getwd()
  )
  
  expect_true(is.fmri_dataset(dset))
  expect_equal(n_timepoints(dset$sampling_frame), 200)
  expect_equal(n_runs(dset$sampling_frame), 2)
  expect_equal(get_run_lengths(dset$sampling_frame), c(100, 100))
})

test_that("memory dataset with multiple runs works like fmrireg", {
  # Simulate the workflow that fmrireg users would have used
  
  set.seed(789)
  
  # Create separate "scans" for each run (as NeuroVec objects would have been)
  run1_data <- matrix(rnorm(150 * 50), nrow = 150, ncol = 50)
  run2_data <- matrix(rnorm(180 * 50), nrow = 180, ncol = 50) 
  run3_data <- matrix(rnorm(120 * 50), nrow = 120, ncol = 50)
  
  # Combine runs (our package handles this internally)
  all_data <- rbind(run1_data, run2_data, run3_data)
  
  # Create dataset with variable run lengths (common in fMRI)
  dset <- fmri_dataset_create(
    images = all_data,
    TR = 2.5,
    run_lengths = c(150, 180, 120)
  )
  
  expect_true(is.fmri_dataset(dset))
  expect_equal(n_runs(dset$sampling_frame), 3)
  expect_equal(get_run_lengths(dset$sampling_frame), c(150, 180, 120))
  expect_equal(n_timepoints(dset$sampling_frame), 450)
  expect_equal(get_total_duration(dset$sampling_frame), 450 * 2.5)
})

test_that("event table integration works like fmrireg", {
  # Test event table integration similar to fmrireg's approach
  
  set.seed(101112)
  test_matrix <- matrix(rnorm(2000), nrow = 400, ncol = 5)
  
  # Create event table with fmrireg-style structure
  events <- data.frame(
    onset = c(10, 30, 60, 90, 120, 150, 180, 210),
    duration = c(2, 2, 3, 2, 2, 3, 2, 2),
    trial_type = rep(c("stim_A", "stim_B"), 4),
    run = c(rep(1, 4), rep(2, 4)),
    repnum = factor(1:8)  # fmrireg often used repnum
  )
  
  dset <- fmri_dataset_create(
    images = test_matrix,
    TR = 2.0,
    run_lengths = c(200, 200),
    event_table = events
  )
  
  expect_true(is.fmri_dataset(dset))
  
  # Check event table is properly stored
  stored_events <- get_event_table(dset)
  expect_equal(nrow(stored_events), 8)
  expect_true("trial_type" %in% names(stored_events))
  expect_true("repnum" %in% names(stored_events))
  
  # Check that validation passes (events should be within time bounds)
  expect_true(validate_fmri_dataset(dset))
})

test_that("mask handling works like fmrireg", {
  # Test mask handling similar to fmrireg's NeuroVol approach
  
  set.seed(131415)
  test_matrix <- matrix(rnorm(500), nrow = 100, ncol = 5)
  
  # Test with logical mask (equivalent to LogicalNeuroVol)
  logical_mask <- c(TRUE, FALSE, TRUE, TRUE, FALSE)
  
  dset_logical <- fmri_dataset_create(
    images = test_matrix,
    mask = logical_mask,
    TR = 1.5,
    run_lengths = 100
  )
  
  expect_true(is.fmri_dataset(dset_logical))
  expect_equal(get_num_voxels(dset_logical), 3)  # 3 TRUE values
  
  # Test without mask (equivalent to no mask in fmrireg)
  dset_no_mask <- fmri_dataset_create(
    images = test_matrix,
    TR = 1.5,
    run_lengths = 100
  )
  
  expect_true(is.fmri_dataset(dset_no_mask))
  expect_equal(get_num_voxels(dset_no_mask), 5)  # All voxels
}) 