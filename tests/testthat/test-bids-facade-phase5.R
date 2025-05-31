# Tests for BIDS Facade Phase 5: AI & Community Integration
# Tests workflow creation and community wisdom utilities

# Skip all tests if bidser is not available
skip_if_not_installed <- function(pkg) {
  skip_if_not(requireNamespace(pkg, quietly = TRUE), 
              paste("Package", pkg, "not available"))
}

test_that("create_workflow() creates basic workflow objects", {
  skip_if_not_installed("bidser")
  
  # Test basic workflow creation
  workflow <- create_workflow("test_workflow")
  
  expect_true(inherits(workflow, "bids_workflow"))
  expect_equal(workflow$name, "test_workflow")
  expect_null(workflow$description)
  expect_equal(length(workflow$steps), 0)
  expect_false(workflow$finished)
})

test_that("describe() adds description to workflows", {
  skip_if_not_installed("bidser")
  
  # Create workflow and add description
  workflow <- create_workflow("analysis_workflow")
  described_workflow <- describe.bids_workflow(workflow, "Complete fMRI preprocessing and analysis")
  
  expect_equal(described_workflow$description, "Complete fMRI preprocessing and analysis")
  expect_equal(described_workflow$name, "analysis_workflow")
  expect_true(inherits(described_workflow, "bids_workflow"))
})

test_that("add_step() adds processing steps to workflows", {
  skip_if_not_installed("bidser")
  
  # Create workflow and add steps
  workflow <- create_workflow("preprocessing_workflow")
  
  # Define test processing steps
  step1 <- function(data) {
    message("Motion correction")
    return(data)
  }
  
  step2 <- function(data) {
    message("Spatial smoothing")
    return(data)
  }
  
  # Add steps to workflow
  workflow_with_steps <- workflow %>%
    add_step.bids_workflow(step1) %>%
    add_step.bids_workflow(step2)
  
  expect_equal(length(workflow_with_steps$steps), 2)
  expect_true(is.function(workflow_with_steps$steps[[1]]))
  expect_true(is.function(workflow_with_steps$steps[[2]]))
})

test_that("finish_with_flourish() marks workflows as complete", {
  skip_if_not_installed("bidser")
  
  # Create and finish workflow
  workflow <- create_workflow("complete_workflow")
  finished_workflow <- finish_with_flourish.bids_workflow(workflow)
  
  expect_true(finished_workflow$finished)
  expect_equal(finished_workflow$name, "complete_workflow")
})

test_that("complex workflow creation with method chaining", {
  skip_if_not_installed("bidser")
  
  # Define processing functions
  motion_correction <- function(dataset) {
    message("Applying motion correction")
    attr(dataset, "motion_corrected") <- TRUE
    return(dataset)
  }
  
  spatial_smoothing <- function(dataset) {
    message("Applying spatial smoothing")
    attr(dataset, "smoothed") <- TRUE
    return(dataset)
  }
  
  statistical_analysis <- function(dataset) {
    message("Running statistical analysis")
    attr(dataset, "analyzed") <- TRUE
    return(dataset)
  }
  
  # Create complex workflow with chaining
  complex_workflow <- create_workflow("fmri_analysis") %>%
    describe.bids_workflow("Complete fMRI preprocessing and first-level analysis") %>%
    add_step.bids_workflow(motion_correction) %>%
    add_step.bids_workflow(spatial_smoothing) %>%
    add_step.bids_workflow(statistical_analysis) %>%
    finish_with_flourish.bids_workflow()
  
  # Verify workflow structure
  expect_equal(complex_workflow$name, "fmri_analysis")
  expect_equal(complex_workflow$description, "Complete fMRI preprocessing and first-level analysis")
  expect_equal(length(complex_workflow$steps), 3)
  expect_true(complex_workflow$finished)
  expect_true(inherits(complex_workflow, "bids_workflow"))
})

test_that("apply_workflow() executes steps on fmri_dataset", {
  skip_if_not_installed("bidser")
  
  # Create mock fmri_dataset
  mock_dataset <- list(
    data = array(rnorm(1000), dim = c(10, 10, 10)),
    mask = array(TRUE, dim = c(10, 10, 10)),
    metadata = list(TR = 2.0)
  )
  class(mock_dataset) <- "fmri_dataset"
  
  # Define test processing steps
  step1 <- function(dataset) {
    attr(dataset, "step1_applied") <- TRUE
    return(dataset)
  }
  
  step2 <- function(dataset) {
    attr(dataset, "step2_applied") <- TRUE
    return(dataset)
  }
  
  # Create workflow
  test_workflow <- create_workflow("test_processing") %>%
    add_step.bids_workflow(step1) %>%
    add_step.bids_workflow(step2)
  
  # Apply workflow
  processed_dataset <- apply_workflow.fmri_dataset(mock_dataset, test_workflow)
  
  # Verify steps were applied
  expect_true(attr(processed_dataset, "step1_applied"))
  expect_true(attr(processed_dataset, "step2_applied"))
  expect_true(inherits(processed_dataset, "fmri_dataset"))
})

test_that("apply_workflow() handles non-function steps gracefully", {
  skip_if_not_installed("bidser")
  
  # Create mock dataset
  mock_dataset <- list(data = "test")
  class(mock_dataset) <- "fmri_dataset"
  
  # Create workflow with mixed step types
  mixed_workflow <- create_workflow("mixed_steps")
  mixed_workflow$steps <- list(
    function(x) { attr(x, "processed") <- TRUE; x },  # Function step
    "not a function",                                 # Non-function step
    function(x) { attr(x, "final") <- TRUE; x }      # Another function step
  )
  class(mixed_workflow) <- "bids_workflow"
  
  # Apply workflow
  result <- apply_workflow.fmri_dataset(mock_dataset, mixed_workflow)
  
  # Verify only function steps were applied
  expect_true(attr(result, "processed"))
  expect_true(attr(result, "final"))
  expect_equal(result$data, "test")
})

test_that("discover_best_practices() provides community wisdom", {
  skip_if_not_installed("bidser")
  
  # Test motion correction advice
  output <- capture.output(advice <- discover_best_practices("motion_correction"))
  
  expect_true(any(grepl("Community Wisdom for Motion Correction", output)))
  expect_true(any(grepl("FD threshold", output)))
  expect_true(any(grepl("DVARS threshold", output)))
  expect_true(any(grepl("Scrubbing", output)))
  expect_true(any(grepl("Confound regression", output)))
  
  # Verify function returns advice invisibly
  expect_true(is.character(advice))
  expect_true(nchar(advice) > 0)
})

test_that("discover_best_practices() handles unknown topics", {
  skip_if_not_installed("bidser")
  
  # Test unknown topic
  output <- capture.output(advice <- discover_best_practices("unknown_topic"))
  
  expect_true(any(grepl("No community advice for unknown_topic", output)))
  expect_true(is.character(advice))
})

test_that("discover_best_practices() with default topic", {
  skip_if_not_installed("bidser")
  
  # Test default topic
  output <- capture.output(advice <- discover_best_practices())
  
  expect_true(any(grepl("No community advice for general", output)))
  expect_true(is.character(advice))
})

test_that("workflow validation and error handling", {
  skip_if_not_installed("bidser")
  
  # Test apply_workflow with invalid workflow
  mock_dataset <- list(data = "test")
  class(mock_dataset) <- "fmri_dataset"
  
  invalid_workflow <- list(name = "invalid")
  class(invalid_workflow) <- "not_a_workflow"
  
  # Should fail with stopifnot
  expect_error(apply_workflow.fmri_dataset(mock_dataset, invalid_workflow))
})

test_that("workflow persistence and structure integrity", {
  skip_if_not_installed("bidser")
  
  # Create workflow and verify structure persists through modifications
  original_workflow <- create_workflow("persistence_test")
  
  # Add multiple modifications
  modified_workflow <- original_workflow %>%
    describe.bids_workflow("Test workflow for persistence") %>%
    add_step.bids_workflow(function(x) x) %>%
    add_step.bids_workflow(function(x) x) %>%
    finish_with_flourish.bids_workflow()
  
  # Verify all properties are maintained
  expect_equal(modified_workflow$name, "persistence_test")
  expect_equal(modified_workflow$description, "Test workflow for persistence")
  expect_equal(length(modified_workflow$steps), 2)
  expect_true(modified_workflow$finished)
  expect_true(inherits(modified_workflow, "bids_workflow"))
  
  # Verify original workflow is unchanged
  expect_null(original_workflow$description)
  expect_equal(length(original_workflow$steps), 0)
  expect_false(original_workflow$finished)
})

test_that("community best practices covers multiple domains", {
  skip_if_not_installed("bidser")
  
  # Test that the framework can be extended for different topics
  topics <- c("motion_correction", "spatial_normalization", "statistical_modeling")
  
  for (topic in topics) {
    output <- capture.output(advice <- discover_best_practices(topic))
    expect_true(is.character(advice))
    expect_true(nchar(advice) > 0)
    
    if (topic == "motion_correction") {
      expect_true(any(grepl("FD threshold", output)))
    } else {
      expect_true(any(grepl("No community advice", output)))
    }
  }
})

test_that("workflow execution order is preserved", {
  skip_if_not_installed("bidser")
  
  # Create dataset to track execution order
  mock_dataset <- list(execution_log = character(0))
  class(mock_dataset) <- "fmri_dataset"
  
  # Define steps that log their execution
  step_a <- function(dataset) {
    dataset$execution_log <- c(dataset$execution_log, "step_a")
    return(dataset)
  }
  
  step_b <- function(dataset) {
    dataset$execution_log <- c(dataset$execution_log, "step_b")
    return(dataset)
  }
  
  step_c <- function(dataset) {
    dataset$execution_log <- c(dataset$execution_log, "step_c")
    return(dataset)
  }
  
  # Create workflow with specific order
  ordered_workflow <- create_workflow("order_test") %>%
    add_step.bids_workflow(step_a) %>%
    add_step.bids_workflow(step_b) %>%
    add_step.bids_workflow(step_c)
  
  # Execute workflow
  result <- apply_workflow.fmri_dataset(mock_dataset, ordered_workflow)
  
  # Verify execution order
  expect_equal(result$execution_log, c("step_a", "step_b", "step_c"))
})

test_that("integration between Phase 5 and earlier phases", {
  skip_if_not_installed("bidser")
  
  # Create mock BIDS facade (from earlier phases)
  mock_facade <- list(
    path = "/test/path",
    project = list(),
    cache = new.env(parent = emptyenv()),
    nl_filters = list(
      task = "rest",
      quality = "excellent",
      pipeline = "fmriprep"
    )
  )
  class(mock_facade) <- "bids_facade"
  
  # Create workflow that could work with BIDS data
  bids_workflow <- create_workflow("bids_processing") %>%
    describe.bids_workflow("Process BIDS data with quality filters") %>%
    add_step.bids_workflow(function(dataset) {
      # Mock processing step that uses BIDS metadata
      attr(dataset, "bids_processed") <- TRUE
      return(dataset)
    }) %>%
    finish_with_flourish.bids_workflow()
  
  # Verify integration potential
  expect_true(inherits(mock_facade, "bids_facade"))
  expect_true(inherits(bids_workflow, "bids_workflow"))
  expect_equal(bids_workflow$description, "Process BIDS data with quality filters")
  expect_true(!is.null(mock_facade$nl_filters))
})

test_that("error resilience in workflow execution", {
  skip_if_not_installed("bidser")
  
  # Create dataset
  mock_dataset <- list(data = "test", error_log = character(0))
  class(mock_dataset) <- "fmri_dataset"
  
  # Define steps including one that might error
  safe_step <- function(dataset) {
    dataset$error_log <- c(dataset$error_log, "safe_step_executed")
    return(dataset)
  }
  
  error_step <- function(dataset) {
    dataset$error_log <- c(dataset$error_log, "error_step_attempted")
    stop("Intentional error for testing")
  }
  
  recovery_step <- function(dataset) {
    dataset$error_log <- c(dataset$error_log, "recovery_step_executed")
    return(dataset)
  }
  
  # Create workflow with error handling
  error_workflow <- create_workflow("error_test")
  error_workflow$steps <- list(
    safe_step,
    function(dataset) {
      tryCatch(error_step(dataset), 
               error = function(e) {
                 dataset$error_log <- c(dataset$error_log, "error_caught")
                 return(dataset)
               })
    },
    recovery_step
  )
  class(error_workflow) <- "bids_workflow"
  
  # Execute workflow with error handling
  result <- apply_workflow.fmri_dataset(mock_dataset, error_workflow)
  
  # Verify error was handled and processing continued
  expect_true("safe_step_executed" %in% result$error_log)
  expect_true("error_caught" %in% result$error_log)
  expect_true("recovery_step_executed" %in% result$error_log)
}) 