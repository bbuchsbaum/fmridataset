# Golden test helper functions for fmridataset package

# Generate reference data for golden tests
generate_reference_data <- function() {
  # Set seed for reproducibility
  set.seed(42)

  # Basic dimensions
  n_voxels <- 100
  n_time <- 50
  n_runs <- 2
  TR <- 2.0

  # Create reference datasets
  ref_data <- list(
    # Simple matrix dataset (rows = time, cols = voxels)
    matrix_data = matrix(rnorm(n_time * n_voxels), nrow = n_time, ncol = n_voxels),

    # Multi-run data
    multirun_data = lapply(1:n_runs, function(i) {
      matrix(rnorm((n_time / n_runs) * n_voxels), nrow = n_time / n_runs, ncol = n_voxels)
    }),

    # Metadata
    metadata = list(
      TR = TR,
      run_lengths = rep(n_time / n_runs, n_runs),
      dims = c(10, 10, 1, n_time),
      origin = c(0, 0, 0),
      spacing = c(2, 2, 2)
    ),

    # Mask data
    mask = rep(c(TRUE, FALSE), length.out = n_voxels),

    # Block information for event-related designs
    block_info = data.frame(
      onset = seq(0, n_time * TR - 10, by = 10),
      duration = rep(5, floor(n_time * TR / 10)),
      condition = rep(c("A", "B"), length.out = floor(n_time * TR / 10))
    )
  )

  ref_data
}

# Save golden reference data
save_golden_data <- function(data, name, dir = "tests/testthat/golden") {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }

  filename <- file.path(dir, paste0(name, ".rds"))
  saveRDS(data, filename, version = 2)
  invisible(filename)
}

# Load golden reference data
load_golden_data <- function(name, dir = NULL) {
  if (is.null(dir)) {
    # Try to find the golden directory relative to the test file
    if (file.exists("golden")) {
      dir <- "golden"
    } else if (file.exists("tests/testthat/golden")) {
      dir <- "tests/testthat/golden"
    } else {
      dir <- system.file("tests", "testthat", "golden", package = "fmridataset")
      if (!dir.exists(dir)) {
        dir <- "golden" # fallback
      }
    }
  }

  filename <- file.path(dir, paste0(name, ".rds"))
  if (!file.exists(filename)) {
    skip(paste("Golden data not found:", filename))
  }
  readRDS(filename)
}

# Compare data structures with tolerance
compare_golden <- function(actual, expected, tolerance = 1e-6, check_attributes = TRUE) {
  # Basic type checking
  expect_equal(class(actual), class(expected))

  # Numeric comparison with tolerance
  if (is.numeric(actual)) {
    expect_equal(actual, expected, tolerance = tolerance)
  } else if (is.matrix(actual) || is.array(actual)) {
    expect_equal(dim(actual), dim(expected))
    expect_equal(as.vector(actual), as.vector(expected), tolerance = tolerance)
  } else if (is.list(actual)) {
    expect_equal(names(actual), names(expected))
    for (nm in names(actual)) {
      compare_golden(actual[[nm]], expected[[nm]],
        tolerance = tolerance,
        check_attributes = check_attributes
      )
    }
  } else {
    expect_equal(actual, expected)
  }

  # Attribute checking (optional)
  if (check_attributes && !is.null(attributes(actual))) {
    # Compare non-standard attributes
    actual_attrs <- attributes(actual)
    expected_attrs <- attributes(expected)

    # Remove attributes that might differ legitimately
    exclude_attrs <- c("names", "dim", "dimnames", "class")
    actual_attrs <- actual_attrs[!names(actual_attrs) %in% exclude_attrs]
    expected_attrs <- expected_attrs[!names(expected_attrs) %in% exclude_attrs]

    if (length(actual_attrs) > 0 || length(expected_attrs) > 0) {
      expect_equal(actual_attrs, expected_attrs)
    }
  }
}

# Generate mock NeuroVec object for testing
create_mock_neurvec <- function(dims = c(10, 10, 10, 50),
                                data = NULL,
                                spacing = c(2, 2, 2),
                                origin = c(0, 0, 0)) {
  if (is.null(data)) {
    data <- array(rnorm(prod(dims)), dim = dims)
  }

  structure(
    data,
    class = c("DenseNeuroVec", "NeuroVec", "array"),
    space = structure(
      list(
        dim = dims[1:3],
        origin = origin,
        spacing = spacing
      ),
      class = "NeuroSpace"
    )
  )
}

# Generate and save all golden test data
generate_all_golden_data <- function() {
  message("Generating golden test data...")

  # Basic reference data
  ref_data <- generate_reference_data()
  save_golden_data(ref_data, "reference_data")

  # fmri_dataset objects
  set.seed(42)
  mat_dataset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = nrow(ref_data$matrix_data)
  )
  save_golden_data(mat_dataset, "matrix_dataset")

  # FmriSeries objects
  fmri_series <- as_delayed_array(mat_dataset)
  # Convert to regular array for storage
  fmri_series_data <- list(
    data = as.array(fmri_series),
    dims = dim(fmri_series),
    class = class(fmri_series)
  )
  save_golden_data(fmri_series_data, "fmri_series")

  # Sampling frame
  sframe <- fmrihrf::sampling_frame(
    TR = ref_data$metadata$TR,
    blocklens = ref_data$metadata$run_lengths
  )
  save_golden_data(sframe, "sampling_frame")

  # Mock NeuroVec for backend testing
  mock_vec <- create_mock_neurvec(
    dims = ref_data$metadata$dims,
    spacing = ref_data$metadata$spacing,
    origin = ref_data$metadata$origin
  )
  save_golden_data(mock_vec, "mock_neurvec")

  message("Golden test data generation complete!")
  invisible(TRUE)
}

# Snapshot testing wrapper for print methods
test_print_snapshot <- function(object, name) {
  # Use testthat's snapshot testing
  expect_snapshot({
    print(object)
  })
}

# Test data chunk consistency
test_chunk_golden <- function(dataset, chunk_specs, name) {
  chunks <- data_chunks(dataset, chunk_spec = chunk_specs)

  # Collect all chunks
  chunk_list <- list()
  i <- 1
  for (chunk in chunks) {
    chunk_list[[i]] <- chunk
    i <- i + 1
  }

  # Save for comparison
  save_golden_data(chunk_list, paste0(name, "_chunks"))

  # Return for immediate testing if needed
  invisible(chunk_list)
}
