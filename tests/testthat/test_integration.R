test_that("complete workflow with matrix backend", {
  # 1. Create data
  n_timepoints <- 100
  n_voxels <- 50
  n_runs <- 2

  # Generate synthetic fMRI data
  set.seed(123)
  time_series <- matrix(0, nrow = n_timepoints, ncol = n_voxels)

  # Add signal to some voxels
  signal_voxels <- 1:10
  for (v in signal_voxels) {
    time_series[, v] <- sin(seq(0, 4 * pi, length.out = n_timepoints)) +
      rnorm(n_timepoints, sd = 0.5)
  }

  # Add noise to other voxels
  noise_voxels <- 11:n_voxels
  for (v in noise_voxels) {
    time_series[, v] <- rnorm(n_timepoints)
  }

  # 2. Create backend
  backend <- matrix_backend(
    data_matrix = time_series,
    spatial_dims = c(10, 5, 1),
    metadata = list(
      study = "test_study",
      subject = "sub01"
    )
  )

  # 3. Create dataset
  dataset <- fmri_dataset(
    scans = backend,
    TR = 2,
    run_length = c(50, 50),
    event_table = data.frame(
      onset = c(10, 30, 60, 80),
      duration = c(5, 5, 5, 5),
      condition = c("A", "B", "A", "B"),
      run = c(1, 1, 2, 2)
    )
  )

  # 4. Test basic accessors
  expect_equal(get_TR(dataset$sampling_frame), 2)
  expect_equal(n_runs(dataset$sampling_frame), 2)
  expect_equal(n_timepoints(dataset$sampling_frame), n_timepoints)

  # 5. Test data retrieval
  full_data <- get_data_matrix(dataset)
  expect_equal(dim(full_data), c(n_timepoints, n_voxels))
  expect_equal(full_data, time_series)

  # 6. Test mask
  mask <- get_mask(dataset)
  expect_true(is.logical(mask))
  expect_length(mask, n_voxels)

  # 7. Test chunking
  chunks <- data_chunks(dataset, nchunks = 5)
  chunk_list <- list()
  for (i in 1:5) {
    chunk_list[[i]] <- chunks$nextElem()
  }

  # Verify chunks cover all voxels
  all_voxel_inds <- unlist(lapply(chunk_list, function(x) x$voxel_ind))
  expect_equal(sort(unique(all_voxel_inds)), 1:n_voxels)

  # 8. Test runwise processing
  run_chunks <- data_chunks(dataset, runwise = TRUE)
  run1 <- run_chunks$nextElem()
  run2 <- run_chunks$nextElem()

  expect_equal(nrow(run1$data), 50)
  expect_equal(nrow(run2$data), 50)

  # 9. Test metadata preservation
  metadata <- backend_get_metadata(dataset$backend)
  expect_equal(metadata$study, "test_study")
  expect_equal(metadata$subject, "sub01")
})

test_that("complete workflow with file-based backend", {
  skip_if_not_installed("neuroim2")

  # Mock file system
  temp_dir <- tempdir()
  scan_files <- file.path(temp_dir, c("run1.nii", "run2.nii"))
  mask_file <- file.path(temp_dir, "mask.nii")

  # Create event data
  events <- data.frame(
    onset = c(5, 15, 25, 35),
    duration = rep(2, 4),
    trial_type = c("left", "right", "left", "right"),
    run = c(1, 1, 2, 2)
  )

  with_mocked_bindings(
    file.exists = function(x) TRUE,
    .package = "base",
    code = {
      with_mocked_bindings(
        read_header = function(fname) {
          # Create a proper S4 object mock that has all needed slots
          if (!methods::isClass("MockNIFTIHeader")) {
            setClass("MockNIFTIHeader", slots = c(
              dims = "integer",
              pixdims = "numeric",
              spacing = "numeric",
              origin = "numeric",
              spatial_axes = "character"
            ))
            setMethod("dim", "MockNIFTIHeader", function(x) x@dims)
          }
          new("MockNIFTIHeader",
            dims = c(10L, 10L, 10L, 50L),
            pixdims = c(-1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            spacing = c(2.0, 2.0, 2.0),
            origin = c(0.0, 0.0, 0.0),
            spatial_axes = c("x", "y", "z")
          )
        },
        read_vol = function(x) {
          # Return mock mask
          structure(
            array(c(rep(1, 500), rep(0, 500)), c(10, 10, 10)),
            class = c("NeuroVol", "array"),
            dim = c(10, 10, 10)
          )
        },
        read_vec = function(files, ...) {
          # Return mock 4D data
          n_files <- if (is.character(files)) length(files) else 1
          total_time <- n_files * 50
          structure(
            array(rnorm(10 * 10 * 10 * total_time), c(10, 10, 10, total_time)),
            class = c("NeuroVec", "array"),
            dim = c(10, 10, 10, total_time)
          )
        },
        trans = function(x) diag(4),
        spacing = function(x) c(2, 2, 2),
        space = function(x) "MNI",
        origin = function(x) c(0, 0, 0),
        NeuroSpace = function(dim, spacing, origin, axes = NULL) {
          # Create a simple mock NeuroSpace object
          structure(list(dim = dim, spacing = spacing, origin = origin),
            class = "NeuroSpace"
          )
        },
        series = function(vec, indices) {
          # Return time series for selected voxels
          n_time <- dim(vec)[4]
          matrix(rnorm(n_time * length(indices)),
            nrow = n_time,
            ncol = length(indices)
          )
        },
        .package = "neuroim2",
        {
          # Create dataset using file paths
          dataset <- fmri_dataset(
            scans = scan_files,
            mask = "mask.nii",
            TR = 2.5,
            run_length = c(50, 50),
            event_table = events,
            base_path = temp_dir,
            preload = FALSE
          )

          # Verify dataset structure
          expect_s3_class(dataset, "fmri_dataset")
          expect_s3_class(dataset$backend, "nifti_backend")

          # Test data access
          dims <- backend_get_dims(dataset$backend)
          expect_equal(dims$spatial, c(10, 10, 10))
          expect_equal(dims$time, 100)

          # Test metadata
          metadata <- backend_get_metadata(dataset$backend)
          expect_true("affine" %in% names(metadata))
          expect_equal(metadata$voxel_dims, c(2, 2, 2))

          # Test chunked processing with foreach
          if (requireNamespace("foreach", quietly = TRUE)) {
            chunks <- data_chunks(dataset, nchunks = 4)

            # Process chunks to compute mean activation
            results <- foreach::foreach(chunk = chunks, .combine = c) %do% {
              mean(chunk$data)
            }

            expect_length(results, 4)
            expect_true(all(is.numeric(results)))
          }
        }
      )
    }
  )
})

test_that("error handling in integrated workflow", {
  # Test various error conditions

  # 1. Invalid run length
  backend <- matrix_backend(matrix(1:100, 10, 10))
  expect_error(
    fmri_dataset(backend, TR = 2, run_length = 20),
    "Sum of run_length .* must equal total time points"
  )

  # 2. Backend validation failure
  backend <- matrix_backend(matrix(1:100, 10, 10))

  # Mock a failing mask (all FALSE)
  with_mocked_bindings(
    backend_get_mask = function(x) rep(FALSE, 10),
    .package = "fmridataset",
    {
      expect_error(
        fmri_dataset(backend, TR = 2, run_length = 10),
        "mask must contain at least one TRUE value"
      )
    }
  )
})

test_that("print and summary methods work in integrated workflow", {
  # Create a small dataset
  backend <- matrix_backend(
    matrix(rnorm(200), 20, 10),
    metadata = list(description = "Test dataset")
  )

  dataset <- fmri_dataset(
    backend,
    TR = 1.5,
    run_length = c(10, 10),
    event_table = data.frame(
      onset = c(5, 15),
      condition = c("A", "B")
    )
  )

  # Test print output
  output <- capture.output(print(dataset))
  expect_true(any(grepl("fMRI Dataset", output)))

  # Test sampling frame print
  frame_output <- capture.output(print(dataset$sampling_frame))
  expect_true(any(grepl("Sampling Frame", frame_output)))
  expect_true(any(grepl("TR: 1.5", frame_output)))
})

test_that("conversion between dataset types", {
  # Start with matrix dataset
  mat_data <- matrix(rnorm(300), 30, 10)
  mat_dataset <- matrix_dataset(mat_data, TR = 2, run_length = 30)

  # Convert to itself (should return identical)
  converted <- as.matrix_dataset(mat_dataset)
  expect_identical(converted$datamat, mat_dataset$datamat)

  # Create backend-based dataset
  backend <- matrix_backend(mat_data)
  backend_dataset <- fmri_dataset(backend, TR = 2, run_length = 30)

  # Both should have same data access
  expect_equal(
    get_data_matrix(mat_dataset),
    get_data_matrix(backend_dataset)
  )
})
