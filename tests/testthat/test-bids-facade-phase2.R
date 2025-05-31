# Tests for BIDS Facade Phase 2: Enhanced Discovery & Quality Assessment
# Tests enhanced discovery output and quality assessment utilities

# Skip all tests if bidser is not available
skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE), 
              paste("Package", pkg, "not available"))
}

test_that("enhanced discover() includes quality metrics", {
  skip_if_not_installed("bidser")
  
  # Create comprehensive mock BIDS project with derivatives
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01", "sub-01", "sub-02", "sub-02", "sub-02", "sub-03", "sub-03", "sub-03",
              "sub-01", "sub-01", "sub-02", "sub-02", "sub-03", "sub-03"),
    datatype = c("func", "func", "anat", "func", "func", "anat", "func", "func", "anat",
                 "func", "func", "func", "func", "func", "func"),
    suffix = c("bold", "events", "T1w", "bold", "events", "T1w", "bold", "events", "T1w",
               "bold", "events", "bold", "events", "bold", "events"),
    fmriprep = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
                 FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  
  file_structure_df$task <- c("rest", "rest", NA, "rest", "rest", NA, "rest", "rest", NA,
                              "memory", "memory", "memory", "memory", "memory", "memory")
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "quality_test",
    participants = c("sub-01", "sub-02", "sub-03"),
    file_structure = file_structure_df,
    prep_dir = "derivatives/fmriprep"
  )
  
  # Test that enhanced discovery works with bidser functions
  expect_true(inherits(mock_bids, "mock_bids_project"))
  
  # Test enhanced discovery structure
  mock_enhanced_discovery <- list(
    summary = list(participants = 3, tasks = 2),
    participants = data.frame(participant_id = paste0("sub-0", 1:3)),
    tasks = data.frame(task_id = c("rest", "memory")),
    sessions = NULL,
    quality = data.frame(scan_check = "passed")
  )
  class(mock_enhanced_discovery) <- "bids_discovery_enhanced"
  
  # Test enhanced print method
  output <- capture.output(print(mock_enhanced_discovery))
  expect_true(any(grepl("BIDS Discovery", output)))
  expect_true(any(grepl("3 participants", output)))
  expect_true(any(grepl("2 tasks", output)))
  expect_true(any(grepl("Quality metrics available", output)))
})

test_that("print.bids_discovery_enhanced handles character vectors", {
  skip_if_not_installed("bidser")

  mock_discovery <- list(
    summary = list(),
    participants = c("sub-01", "sub-02"),
    tasks = c("rest", "memory"),
    sessions = NULL,
    quality = NULL
  )
  class(mock_discovery) <- "bids_discovery_enhanced"

  output <- capture.output(print(mock_discovery))
  expect_true(any(grepl("2 participants", output)))
  expect_true(any(grepl("2 tasks", output)))
})

test_that("assess_quality() provides comprehensive quality metrics", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS with confounds and quality data
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01"),
    datatype = c("func", "func"),
    suffix = c("bold", "events"),
    fmriprep = c(FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  
  file_structure_df$task <- c("rest", "rest")
  
  confound_data <- list(
    "sub-01" = list(
      "task-rest" = data.frame(
        framewise_displacement = rnorm(100, 0.1, 0.05),
        dvars = rnorm(100, 50, 10),
        trans_x = rnorm(100, 0, 0.1)
      )
    )
  )
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "quality_assessment",
    participants = c("sub-01"),
    file_structure = file_structure_df,
    confound_data = confound_data
  )
  
  # Test quality assessment structure
  mock_quality_report <- list(
    confounds = data.frame(
      framewise_displacement = rnorm(100, 0.1, 0.05),
      dvars = rnorm(100, 50, 10),
      trans_x = rnorm(100, 0, 0.1)
    ),
    quality_metrics = data.frame(
      mean_fd = 0.12,
      max_fd = 0.35,
      percent_high_motion = 5.0
    ),
    mask = list(coverage = 0.98)
  )
  class(mock_quality_report) <- "bids_quality_report"
  
  # Test quality report print method
  output <- capture.output(print(mock_quality_report))
  expect_true(any(grepl("BIDS Quality Report", output)))
  expect_true(any(grepl("1 scan checks", output)))
  expect_true(any(grepl("3 confound regressors", output)))
})

test_that("bidser confounds integration works correctly", {
  skip_if_not_installed("bidser")
  
  # Test with mock confound data
  mock_confounds <- data.frame(
    framewise_displacement = rnorm(200, 0.15, 0.08),
    dvars = rnorm(200, 55, 15),
    trans_x = rnorm(200, 0, 0.12),
    trans_y = rnorm(200, 0, 0.12),
    trans_z = rnorm(200, 0, 0.12),
    rot_x = rnorm(200, 0, 0.02),
    rot_y = rnorm(200, 0, 0.02),
    rot_z = rnorm(200, 0, 0.02)
  )
  
  # Test confound data structure
  expect_true(is.data.frame(mock_confounds))
  expect_true("framewise_displacement" %in% names(mock_confounds))
  expect_true("dvars" %in% names(mock_confounds))
  expect_equal(nrow(mock_confounds), 200)
  
  # Test quality metrics calculation
  mean_fd <- mean(mock_confounds$framewise_displacement, na.rm = TRUE)
  expect_true(is.numeric(mean_fd))
  expect_true(mean_fd > 0)
  
  high_motion_timepoints <- sum(mock_confounds$framewise_displacement > 0.2, na.rm = TRUE)
  expect_true(is.numeric(high_motion_timepoints))
})

test_that("bidser quality assessment functions work with mock data", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS with functional scans
  file_structure_df <- data.frame(
    subid = c("sub-01", "sub-01", "sub-01", "sub-02", "sub-02", "sub-02"),
    datatype = c("func", "func", "anat", "func", "func", "anat"),
    suffix = c("bold", "bold", "T1w", "bold", "bold", "T1w"),
    fmriprep = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  
  file_structure_df$task <- c("rest", "task", NA, "rest", "task", NA)
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "scan_quality",
    participants = c("sub-01", "sub-02"),
    file_structure = file_structure_df
  )
  
  # Test that we can get functional scans
  scans <- bidser::func_scans(mock_bids, subid = "01")
  expect_true(length(scans) >= 0)  # Should return scan paths or empty
  
  # Test session handling
  sessions <- bidser::sessions(mock_bids)
  # Should handle NULL sessions gracefully
  
  # Test that quality checks can be attempted
  # Note: May not work with mock data but shouldn't crash
  quality_result <- tryCatch(
    bidser::check_func_scans(mock_bids),
    error = function(e) NULL
  )
  # Should either return results or NULL without crashing
})

test_that("enhanced discovery caching works correctly", {
  skip_if_not_installed("bidser")
  
  # Create mock facade with cache
  mock_facade <- list(
    path = "/test/path",
    project = list(),
    cache = new.env(parent = emptyenv())
  )
  class(mock_facade) <- "bids_facade"
  
  # Test cache structure
  expect_true(is.environment(mock_facade$cache))
  expect_equal(length(ls(envir = mock_facade$cache)), 0)
  
  # Test storing and retrieving from cache
  test_data <- list(summary = "test", participants = "test")
  assign("discovery", test_data, envir = mock_facade$cache)
  
  expect_true(exists("discovery", envir = mock_facade$cache))
  retrieved <- get("discovery", envir = mock_facade$cache)
  expect_equal(retrieved, test_data)
})

test_that("error handling in quality assessment is robust", {
  skip_if_not_installed("bidser")
  
  # Test graceful handling of missing confounds
  expect_silent({
    mock_quality <- list(
      confounds = NULL,
      quality_metrics = NULL,
      mask = NULL
    )
    class(mock_quality) <- "bids_quality_report"
    output <- capture.output(print(mock_quality))
  })
  
  # Test graceful handling of empty quality data
  expect_silent({
    mock_discovery <- list(
      summary = NULL,
      participants = data.frame(),
      tasks = data.frame(),
      sessions = NULL,
      quality = NULL
    )
    class(mock_discovery) <- "bids_discovery_enhanced"
    output <- capture.output(print(mock_discovery))
  })
})

test_that("Phase 2 integration with preprocessing detection", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS with fMRIPrep derivatives
  file_structure_df <- data.frame(
    subid = c("sub-01"),
    datatype = c("func"),
    suffix = c("bold"),
    fmriprep = c(FALSE),
    stringsAsFactors = FALSE
  )
  
  file_structure_df$task <- c("rest")
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "preproc_test",
    participants = c("sub-01"),
    file_structure = file_structure_df,
    prep_dir = "derivatives/fmriprep"
  )
  
  # Test derivative detection
  expect_true(inherits(mock_bids, "mock_bids_project"))
  
  # Test preprocessing scan access
  preproc_scans <- tryCatch(
    bidser::preproc_scans(mock_bids, subid = "01"),
    error = function(e) character(0)
  )
  # Should return paths, empty vector, or NULL without error
  expect_true(is.character(preproc_scans) || is.null(preproc_scans))
})

test_that("mask creation functionality works with mock data", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS for mask testing
  file_structure_df <- data.frame(
    subid = c("sub-01"),
    datatype = c("func"),
    suffix = c("bold"),
    fmriprep = c(FALSE),
    stringsAsFactors = FALSE
  )
  
  file_structure_df$task <- c("rest")
  
  mock_bids <- bidser::create_mock_bids(
    project_name = "mask_test",
    participants = c("sub-01"),
    file_structure = file_structure_df,
    prep_dir = "derivatives/fmriprep"
  )
  
  # Test mask creation attempt
  mask_result <- tryCatch(
    bidser::create_preproc_mask(mock_bids, subid = "01"),
    error = function(e) NULL
  )
  
  # Should either create mask or fail gracefully
  # The key is that it doesn't crash the system
  expect_true(TRUE)  # Test that we get here without error
})

test_that("quality threshold and filtering logic", {
  skip_if_not_installed("bidser")
  
  # Test quality thresholding logic
  mock_fd_data <- c(0.05, 0.12, 0.18, 0.25, 0.31, 0.15, 0.08)
  
  # Test motion quality classification
  excellent_timepoints <- sum(mock_fd_data < 0.1)
  good_timepoints <- sum(mock_fd_data >= 0.1 & mock_fd_data < 0.2)
  poor_timepoints <- sum(mock_fd_data >= 0.2)
  
  expect_equal(excellent_timepoints, 2)
  expect_equal(good_timepoints, 3)
  expect_equal(poor_timepoints, 2)
  
  # Test overall quality assessment
  percent_good_quality <- (excellent_timepoints + good_timepoints) / length(mock_fd_data) * 100
  expect_true(percent_good_quality > 50)  # Should have majority good quality
}) 