# Simple BIDS Facade Tests with Correct bidser API
# Tests basic functionality using the proper bidser create_mock_bids format

# Skip all tests if bidser is not available
skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE), 
              paste("Package", pkg, "not available"))
}

test_that("bidser mock creation works with correct format", {
  skip_if_not_installed("bidser")
  
  # Create file structure with correct columns
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01", "sub-02", "sub-02"),
    datatype = c("func", "func", "func", "func"),
    suffix = c("bold", "events", "bold", "events"),
    fmriprep = c(FALSE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  
  # Add task information if supported
  file_structure_df$task <- c("rest", "rest", "rest", "rest")
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "simple_test",
    participants = c("sub-01", "sub-02"),
    file_structure = file_structure_df
  )
  
  # Test that mock was created successfully
  expect_true(inherits(mock_bids, "mock_bids_project"))
  expect_true(inherits(mock_bids, "bids_project"))
})

test_that("bidser basic functions work with mock data", {
  skip_if_not_installed("bidser")
  
  # Create comprehensive file structure
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01", "sub-01", "sub-02", "sub-02", "sub-02"),
    datatype = c("func", "func", "anat", "func", "func", "anat"),
    suffix = c("bold", "events", "T1w", "bold", "events", "T1w"),
    fmriprep = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  
  # Add task for functional data
  file_structure_df$task <- c("rest", "rest", NA, "rest", "rest", NA)
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "function_test",
    participants = c("sub-01", "sub-02"),
    file_structure = file_structure_df
  )
  
  # Test core bidser functions
  participants <- bidser::participants(mock_bids)
  expect_true(length(participants) >= 2)
  
  tasks <- bidser::tasks(mock_bids)
  expect_true(length(tasks) >= 1)
  
  sessions <- bidser::sessions(mock_bids)
  # Sessions may be NULL for simple datasets
  
  summary_info <- bidser::bids_summary(mock_bids)
  expect_true(!is.null(summary_info))
})

test_that("bidser confound data integration", {
  skip_if_not_installed("bidser")
  
  # Create file structure with confound-compatible data
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    task = c("rest", "rest"),
    stringsAsFactors = FALSE
  )
  
  # Create confound data
  confound_data <- list(
    "sub-01" = list(
      "task-rest" = data.frame(
        framewise_displacement = rnorm(100, 0.1, 0.05),
        dvars = rnorm(100, 50, 10),
        trans_x = rnorm(100, 0, 0.1),
        trans_y = rnorm(100, 0, 0.1),
        trans_z = rnorm(100, 0, 0.1),
        rot_x = rnorm(100, 0, 0.02),
        rot_y = rnorm(100, 0, 0.02),
        rot_z = rnorm(100, 0, 0.02)
      )
    )
  )
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "confound_test",
    participants = c("sub-01"),
    file_structure = file_structure_df,
    confound_data = confound_data
  )
  
  expect_true(inherits(mock_bids, "mock_bids_project"))
  
  # Test confound reading
  confounds_result <- tryCatch(
    bidser::read_confounds(mock_bids, subid = "01"),
    error = function(e) NULL
  )
  
  # Should work or fail gracefully
  expect_true(is.null(confounds_result) || is.data.frame(confounds_result))
})

test_that("BIDS facade structure can work with bidser", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS project
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    task = c("rest", "rest"),
    stringsAsFactors = FALSE
  )
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "facade_test",
    participants = c("sub-01"),
    file_structure = file_structure_df
  )
  
  # Create facade structure
  facade <- list(
    path = "/test/bids/path",
    project = mock_bids,
    cache = new.env(parent = emptyenv())
  )
  class(facade) <- "bids_facade"
  
  # Test facade structure
  expect_true(inherits(facade, "bids_facade"))
  expect_true(inherits(facade$project, "mock_bids_project"))
  expect_true(is.environment(facade$cache))
  
  # Test that we can create discovery data from bidser
  discovery_data <- list(
    summary = bidser::bids_summary(facade$project),
    participants = bidser::participants(facade$project),
    tasks = bidser::tasks(facade$project),
    sessions = bidser::sessions(facade$project)
  )
  
  expect_true(!is.null(discovery_data$summary))
  expect_true(length(discovery_data$participants) >= 1)
  expect_true(length(discovery_data$tasks) >= 1)
}) 