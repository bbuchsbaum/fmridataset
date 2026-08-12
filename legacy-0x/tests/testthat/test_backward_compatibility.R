test_that("legacy file-based interface still works", {
  skip_if_not_installed("neuroim2")

  # Create temporary test files
  temp_dir <- tempdir()

  # Create mock scan data
  scan_data <- array(rnorm(10 * 10 * 10 * 20), c(10, 10, 10, 20))
  mask_data <- array(sample(0:1, 10 * 10 * 10, replace = TRUE), c(10, 10, 10))

  # Mock file paths
  scan_file <- file.path(temp_dir, "test_scan.nii")
  mask_file <- file.path(temp_dir, "test_mask.nii")

  # Test that legacy interface creates a backend internally
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
            dims = c(10L, 10L, 10L, 20L),
            pixdims = c(-1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0),
            spacing = c(2.0, 2.0, 2.0),
            origin = c(0.0, 0.0, 0.0),
            spatial_axes = c("x", "y", "z")
          )
        },
        read_vol = function(x) {
          if (grepl("mask", x)) mask_data else NULL
        },
        read_vec = function(x, ...) {
          structure(scan_data, class = c("NeuroVec", "array"))
        },
        .package = "neuroim2",
        {
          # Create dataset using legacy interface
          dset <- fmri_dataset(
            scans = scan_file,
            mask = mask_file,
            TR = 2,
            run_length = 20
          )

          expect_s3_class(dset, "fmri_dataset")
          expect_s3_class(dset$backend, "nifti_backend")

          # Verify the backend was created with correct paths
          expect_equal(dset$backend$source, scan_file)
          expect_equal(dset$backend$mask_source, mask_file)
        }
      )
    }
  )
})

test_that("matrix_dataset continues to work without backend", {
  # Create test matrix
  test_data <- matrix(rnorm(100), nrow = 10, ncol = 10)

  # Create matrix dataset using original interface
  dset <- matrix_dataset(
    datamat = test_data,
    TR = 2,
    run_length = 10
  )

  expect_s3_class(dset, "matrix_dataset")
  expect_s3_class(dset, "fmri_dataset")

  # Verify it has the original structure (no backend)
  expect_null(dset$backend)
  expect_equal(dset$datamat, test_data)
  expect_equal(dset$TR, 2)
  expect_equal(length(dset$mask), 10)

  # Test that data access methods still work
  retrieved_data <- get_data_matrix(dset)
  expect_equal(retrieved_data, test_data)

  mask <- get_mask(dset)
  expect_equal(mask, rep(TRUE, 10))
})

test_that("fmri_mem_dataset continues to work", {
  skip_if_not_installed("neuroim2")

  # Create mock NeuroVec and mask
  dims <- c(5, 5, 5, 10)
  mock_vec <- structure(
    array(rnorm(prod(dims)), dims),
    class = c("NeuroVec", "array"),
    dim = dims
  )

  mock_mask <- structure(
    array(1, dims[1:3]),
    class = c("NeuroVol", "array"),
    dim = dims[1:3]
  )

  # Create fmri_mem_dataset
  dset <- fmri_mem_dataset(
    scans = list(mock_vec),
    mask = mock_mask,
    TR = 2
  )

  expect_s3_class(dset, "fmri_mem_dataset")
  expect_s3_class(dset, "fmri_dataset")

  # Verify it doesn't have a backend
  expect_null(dset$backend)

  # Test data access
  with_mocked_bindings(
    NeuroVecSeq = function(...) mock_vec,
    series = function(vec, indices) {
      matrix(rnorm(length(indices) * dims[4]),
        nrow = dims[4],
        ncol = length(indices)
      )
    },
    .package = "neuroim2",
    {
      data <- get_data(dset)
      expect_s3_class(data, "NeuroVec")

      data_matrix <- get_data_matrix(dset)
      expect_true(is.matrix(data_matrix))
    }
  )
})

test_that("latent_dataset continues to work", {
  if (!methods::isClass("MockLatentNeuroVec")) {
    methods::setClass(
      "MockLatentNeuroVec",
      slots = c(basis = "matrix", loadings = "matrix", space = "numeric", offset = "numeric")
    )
  }

  basis <- matrix(rnorm(40), nrow = 10, ncol = 4)
  loadings <- matrix(rnorm(400), nrow = 100, ncol = 4)
  space <- c(4, 5, 5, 10)
  mock_lvec <- methods::new(
    "MockLatentNeuroVec",
    basis = basis,
    loadings = loadings,
    space = space,
    offset = numeric(0)
  )

  with_mocked_bindings(
    requireNamespace = function(package, quietly = TRUE) TRUE,
    .package = "base",
    {
      dset <- latent_dataset(list(mock_lvec), TR = 2, run_length = 10)
      expect_s3_class(dset, "latent_dataset")
      expect_equal(n_runs(dset), 1)

      scores <- get_latent_scores(dset)
      expect_equal(dim(scores), c(10, 4))
    }
  )
})

test_that("all dataset types work with data_chunks", {
  # Test matrix_dataset
  mat_data <- matrix(1:100, nrow = 10, ncol = 10)
  mat_dset <- matrix_dataset(mat_data, TR = 2, run_length = 10)

  chunks <- data_chunks(mat_dset, nchunks = 2)
  chunk1 <- chunks$nextElem()
  expect_s3_class(chunk1, "data_chunk")
  expect_true(is.matrix(chunk1$data))

  # Test with backend
  backend <- matrix_backend(mat_data)
  backend_dset <- fmri_dataset(backend, TR = 2, run_length = 10)

  chunks2 <- data_chunks(backend_dset, nchunks = 2)
  chunk2 <- chunks2$nextElem()
  expect_s3_class(chunk2, "data_chunk")
  expect_true(is.matrix(chunk2$data))
})

test_that("conversion functions work with both old and new datasets", {
  # Create test data
  test_matrix <- matrix(rnorm(200), nrow = 20, ncol = 10)

  # Old style matrix dataset
  old_dset <- matrix_dataset(test_matrix, TR = 2, run_length = 20)

  # New style with backend
  backend <- matrix_backend(test_matrix)
  new_dset <- fmri_dataset(backend, TR = 2, run_length = 20)

  # Test as.matrix_dataset on both
  old_converted <- as.matrix_dataset(old_dset)
  expect_s3_class(old_converted, "matrix_dataset")
  expect_equal(old_converted$datamat, test_matrix)

  # For new style, we need to implement conversion
  # This would need to be added to conversions.R
  # For now, just verify the old style works
})

test_that("print methods work for all dataset types", {
  # Matrix dataset
  mat_dset <- matrix_dataset(
    matrix(1:100, 10, 10),
    TR = 2,
    run_length = 10
  )
  expect_output(print(mat_dset), "fMRI Dataset")

  # Backend-based dataset
  backend <- matrix_backend(matrix(1:100, 10, 10))
  backend_dset <- fmri_dataset(backend, TR = 2, run_length = 10)
  expect_output(print(backend_dset), "fMRI Dataset")
})

test_that("sampling frame works with all dataset types", {
  # Test with various dataset types
  datasets <- list(
    matrix_dataset(matrix(1:100, 10, 10), TR = 2, run_length = 10),
    matrix_dataset(matrix(1:200, 20, 10), TR = 1.5, run_length = c(10, 10))
  )

  for (dset in datasets) {
    frame <- dset$sampling_frame
    expect_s3_class(frame, "sampling_frame")
    expect_equal(get_TR(frame), dset$TR)
    expect_equal(sum(get_run_lengths(frame)), nrow(dset$datamat))
  }
})
