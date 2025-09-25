# Golden tests for fmri_dataset objects

test_that("matrix_dataset produces consistent output", {
  # Load reference data
  ref_data <- load_golden_data("reference_data")

  # Create dataset
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )

  # Load expected dataset
  expected <- load_golden_data("matrix_dataset")

  # Compare structure
  expect_s3_class(dset, "fmri_dataset")
  expect_equal(class(dset), class(expected))

  # Compare dimensions via get_data_matrix
  expect_equal(dim(get_data_matrix(dset)), dim(get_data_matrix(expected)))

  # Compare data content
  actual_data <- get_data(dset)
  expected_data <- get_data(expected)
  compare_golden(actual_data, expected_data)

  # Compare metadata
  expect_equal(dset$TR, expected$TR)
  expect_equal(dset$run_length, expected$run_length)
})

test_that("multi-run dataset produces consistent output", {
  ref_data <- load_golden_data("reference_data")

  # Concatenate multi-run data into single matrix
  combined_data <- do.call(rbind, ref_data$multirun_data)
  run_lengths <- sapply(ref_data$multirun_data, nrow)

  # Create multi-run dataset
  dset <- matrix_dataset(
    combined_data,
    TR = ref_data$metadata$TR,
    run_length = run_lengths
  )

  # Test basic properties
  expect_equal(n_runs(dset), 2)
  expect_equal(get_TR(dset), ref_data$metadata$TR)
  expect_equal(sum(get_run_lengths(dset)), nrow(ref_data$matrix_data))
})

test_that("masked dataset produces consistent output", {
  ref_data <- load_golden_data("reference_data")

  # Create dataset (matrix_dataset doesn't support custom masks)
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )

  # Test default mask (all TRUE)
  mask <- get_mask(dset)
  expect_equal(length(mask), ncol(ref_data$matrix_data))
  expect_true(all(mask))

  # Test data extraction
  data_mat <- get_data_matrix(dset)
  expect_equal(nrow(data_mat), nrow(ref_data$matrix_data))
  expect_equal(ncol(data_mat), ncol(ref_data$matrix_data))
})

test_that("dataset print output matches snapshot", {
  skip_if(testthat::edition_get() < 3, "Snapshot tests require testthat edition 3")

  ref_data <- load_golden_data("reference_data")

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )

  expect_snapshot({
    print(dset)
  })
})

test_that("dataset chunking produces consistent results", {
  ref_data <- load_golden_data("reference_data")

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )

  # Test voxel-wise chunking
  chunks_voxel <- data_chunks(dset, nchunks = 10)
  chunk_sizes <- numeric()
  chunk_count <- 0

  # Use proper iterator interface
  while (TRUE) {
    chunk <- tryCatch(
      {
        iterators::nextElem(chunks_voxel)
      },
      error = function(e) {
        if (grepl("StopIteration", e$message)) {
          NULL
        } else {
          stop(e)
        }
      }
    )

    if (is.null(chunk)) break

    chunk_count <- chunk_count + 1
    expect_s3_class(chunk, "data_chunk")
    chunk_sizes <- c(chunk_sizes, ncol(chunk$data))
  }

  # Should have consistent chunk sizes except possibly the last
  if (length(chunk_sizes) > 1) {
    expect_true(all(chunk_sizes[-length(chunk_sizes)] == 10))
  }
  expect_equal(sum(chunk_sizes), ncol(ref_data$matrix_data))
  expect_true(chunk_count > 0, "Should have at least one chunk")

  # Test run-wise chunking
  if (n_runs(dset) > 1) {
    chunks_run <- data_chunks(dset, runwise = TRUE)
    run_count <- 0
    for (chunk in chunks_run) {
      run_count <- run_count + 1
    }
    expect_equal(run_count, n_runs(dset))
  }
})

test_that("dataset metadata extraction is consistent", {
  ref_data <- load_golden_data("reference_data")

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )

  # Test all metadata accessors
  expect_equal(get_TR(dset), ref_data$metadata$TR)
  expect_equal(n_timepoints(dset), nrow(ref_data$matrix_data))
  expect_equal(n_runs(dset), 1)
  expect_equal(get_run_lengths(dset), nrow(ref_data$matrix_data))
  expect_equal(get_total_duration(dset), nrow(ref_data$matrix_data) * ref_data$metadata$TR)
})

test_that("empty and single-voxel datasets handle correctly", {
  # Single voxel dataset (50 timepoints, 1 voxel)
  single_voxel <- matrix(rnorm(50), nrow = 50, ncol = 1)
  dset_single <- matrix_dataset(single_voxel, TR = 2, run_length = 50)

  # Test dimensions via get_data_matrix
  dm_single <- get_data_matrix(dset_single)
  expect_equal(dim(dm_single), c(50, 1))
  expect_equal(nrow(dm_single), 50)
  expect_equal(ncol(dm_single), 1)

  # Single timepoint dataset (1 timepoint, 100 voxels)
  single_time <- matrix(rnorm(100), nrow = 1, ncol = 100)
  dset_time <- matrix_dataset(single_time, TR = 2, run_length = 1)

  # Test dimensions via get_data_matrix
  dm_time <- get_data_matrix(dset_time)
  expect_equal(dim(dm_time), c(1, 100))
  expect_equal(nrow(dm_time), 1)
  expect_equal(ncol(dm_time), 100)
})
