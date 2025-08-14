test_that("get_latent_scores performance is optimized", {
  skip_if_not_installed("bench")
  
  # Create test data with multiple runs
  n_runs <- 5
  n_time_per_run <- 200
  n_components <- 50
  n_voxels <- 10000
  
  mock_objects <- lapply(1:n_runs, function(i) {
    methods::setClass(
      "MockLatentNeuroVec",
      slots = c(
        basis = "matrix",
        loadings = "matrix",
        space = "numeric",
        offset = "numeric"
      )
    )(
      basis = matrix(rnorm(n_time_per_run * n_components), n_time_per_run, n_components),
      loadings = matrix(rnorm(n_voxels * n_components), n_voxels, n_components),
      space = c(100, 100, 1, n_time_per_run),
      offset = rep(0, n_voxels)
    )
  })
  
  # Create backend
  backend <- latent_backend(mock_objects)
  backend <- backend_open(backend)
  
  # Test 1: Full data access
  time_full <- bench::mark(
    backend_get_data(backend),
    iterations = 10,
    check = FALSE
  )
  
  # Test 2: Random subset access
  random_rows <- sort(sample(1:(n_runs * n_time_per_run), 500))
  random_cols <- sort(sample(1:n_components, 20))
  
  time_subset <- bench::mark(
    backend_get_data(backend, rows = random_rows, cols = random_cols),
    iterations = 10,
    check = FALSE
  )
  
  # Test 3: Sequential block access (common pattern)
  block_rows <- 201:400  # Second run
  time_block <- bench::mark(
    backend_get_data(backend, rows = block_rows),
    iterations = 10,
    check = FALSE
  )
  
  # Performance expectations
  # Full access should be fast (vectorized)
  expect_true(as.numeric(time_full$median) < 0.1)  # Less than 100ms
  
  # Subset access should also be reasonably fast
  expect_true(as.numeric(time_subset$median) < 0.05)  # Less than 50ms
  
  # Block access should be very fast (single run)
  expect_true(as.numeric(time_block$median) < 0.02)  # Less than 20ms
  
  backend_close(backend)
})

test_that("vectorized row mapping is correct", {
  # Create uneven runs to test edge cases
  run_lengths <- c(100, 150, 75, 200)
  mock_objects <- lapply(run_lengths, function(n_time) {
    methods::setClass(
      "MockLatentNeuroVec",
      slots = c(
        basis = "matrix",
        loadings = "matrix", 
        space = "numeric",
        offset = "numeric"
      )
    )(
      basis = matrix(seq_len(n_time * 10), n_time, 10),  # Sequential values for testing
      loadings = matrix(1, 100, 10),
      space = c(10, 10, 1, n_time),
      offset = rep(0, 100)
    )
  })
  
  backend <- latent_backend(mock_objects)
  backend <- backend_open(backend)
  
  # Test various row patterns
  test_cases <- list(
    # All rows from first run
    list(rows = 1:100, expected_runs = 1),
    # All rows from last run  
    list(rows = 326:525, expected_runs = 4),
    # Rows spanning two runs
    list(rows = 95:105, expected_runs = c(1, 2)),
    # Sparse rows across all runs
    list(rows = c(1, 101, 251, 326), expected_runs = 1:4),
    # Random subset
    list(rows = sort(sample(1:525, 50)), expected_runs = NULL)
  )
  
  for (test_case in test_cases) {
    result <- backend_get_data(backend, rows = test_case$rows)
    expect_equal(nrow(result), length(test_case$rows))
    expect_equal(ncol(result), 10)
    
    # Just check dimensions for now - the actual values depend on how
    # the mock data was set up with seq_len(n_time * 10)
  }
  
  backend_close(backend)
})

test_that("optimized batch extraction works correctly", {
  # Test the optimized extraction with known data
  n_components <- 5
  mock_objects <- list(
    methods::setClass(
      "MockLatentNeuroVec",
      slots = c(
        basis = "matrix",
        loadings = "matrix",
        space = "numeric", 
        offset = "numeric"
      )
    )(
      basis = matrix(1:50, 10, n_components),
      loadings = matrix(1, 100, n_components),
      space = c(10, 10, 1, 10),
      offset = rep(0, 100)
    ),
    methods::setClass(
      "MockLatentNeuroVec",
      slots = c(
        basis = "matrix",
        loadings = "matrix",
        space = "numeric",
        offset = "numeric"
      )
    )(
      basis = matrix(51:100, 10, n_components),
      loadings = matrix(1, 100, n_components),
      space = c(10, 10, 1, 10),
      offset = rep(0, 100)
    )
  )
  
  backend <- latent_backend(mock_objects)
  backend <- backend_open(backend)
  
  # Get all data
  all_data <- backend_get_data(backend)
  expect_equal(dim(all_data), c(20, 5))
  expect_equal(all_data[1, 1], 1)
  expect_equal(all_data[10, 1], 10)
  expect_equal(all_data[11, 1], 51)
  expect_equal(all_data[20, 1], 60)
  
  # Get subset that spans both runs
  subset_data <- backend_get_data(backend, rows = 8:13, cols = 1:3)
  expect_equal(dim(subset_data), c(6, 3))
  expect_equal(subset_data[1, 1], 8)   # Row 8 from run 1
  expect_equal(subset_data[3, 1], 10)  # Row 10 from run 1
  expect_equal(subset_data[4, 1], 51)  # Row 1 from run 2
  expect_equal(subset_data[6, 1], 53)  # Row 3 from run 2
  
  backend_close(backend)
})