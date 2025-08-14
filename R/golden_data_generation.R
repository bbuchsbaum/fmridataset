#' Generate Golden Test Data
#'
#' Generate reference data for golden tests. This function creates
#' reproducible test data that can be used to validate consistency
#' across package versions.
#'
#' @param output_dir Directory where golden data files will be saved.
#'   Defaults to "tests/testthat/golden".
#' @param seed Random seed for reproducibility. Defaults to 42.
#'
#' @return Invisibly returns TRUE on success.
#'
#' @details
#' This function generates the following golden test data:
#' \itemize{
#'   \item reference_data.rds - Basic test matrices and metadata
#'   \item matrix_dataset.rds - Example fmri_dataset object
#'   \item fmri_series.rds - Example FmriSeries data
#'   \item sampling_frame.rds - Example sampling_frame object
#'   \item mock_neurvec.rds - Mock NeuroVec object for testing
#' }
#'
#' @examples
#' \dontrun{
#' # Generate golden test data
#' generate_golden_test_data()
#' 
#' # Generate with custom seed
#' generate_golden_test_data(seed = 123)
#' }
#'
#' @export
generate_golden_test_data <- function(output_dir = "tests/testthat/golden", 
                                     seed = 42) {
  # Ensure output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Source the helper functions
  helper_file <- system.file("tests", "testthat", "helper-golden.R", 
                            package = "fmridataset")
  if (file.exists(helper_file)) {
    source(helper_file)
  } else {
    # If package not installed, try local path
    local_helper <- "tests/testthat/helper-golden.R"
    if (file.exists(local_helper)) {
      source(local_helper)
    } else {
      stop("Could not find helper-golden.R. Please ensure it exists in tests/testthat/")
    }
  }
  
  # Set seed for reproducibility
  set.seed(seed)
  
  # Generate all golden data
  tryCatch({
    generate_all_golden_data()
    message("Golden test data successfully generated in: ", output_dir)
    invisible(TRUE)
  }, error = function(e) {
    stop("Failed to generate golden test data: ", e$message)
  })
}

#' Update Golden Test Data
#'
#' Update existing golden test data with new expected outputs.
#' Use this when intentional changes to the package require updating
#' the reference data.
#'
#' @param confirm Logical. If TRUE, will prompt for confirmation before
#'   updating. Defaults to TRUE.
#' @inheritParams generate_golden_test_data
#'
#' @return Invisibly returns TRUE on success.
#'
#' @details
#' This function should be used with caution as it will overwrite
#' existing golden test data. Only use when you are certain that
#' the current outputs are correct and should become the new reference.
#'
#' @examples
#' \dontrun{
#' # Update golden data (will prompt for confirmation)
#' update_golden_test_data()
#' 
#' # Update without confirmation (use with caution!)
#' update_golden_test_data(confirm = FALSE)
#' }
#'
#' @export
update_golden_test_data <- function(output_dir = "tests/testthat/golden",
                                   seed = 42,
                                   confirm = TRUE) {
  if (confirm) {
    response <- readline(
      "Are you sure you want to update golden test data? This will overwrite existing reference data. (yes/no): "
    )
    if (!tolower(response) %in% c("yes", "y")) {
      message("Update cancelled.")
      return(invisible(FALSE))
    }
  }
  
  # Back up existing data
  if (dir.exists(output_dir)) {
    backup_dir <- paste0(output_dir, "_backup_", format(Sys.time(), "%Y%m%d_%H%M%S"))
    message("Backing up existing golden data to: ", backup_dir)
    file.copy(output_dir, backup_dir, recursive = TRUE)
  }
  
  # Generate new golden data
  generate_golden_test_data(output_dir = output_dir, seed = seed)
}

#' Validate Golden Test Data
#'
#' Check that all expected golden test data files exist and are readable.
#'
#' @inheritParams generate_golden_test_data
#'
#' @return A logical vector indicating which files exist, with names
#'   corresponding to the expected files.
#'
#' @examples
#' \dontrun{
#' # Check golden data files
#' validate_golden_test_data()
#' }
#'
#' @export
validate_golden_test_data <- function(output_dir = "tests/testthat/golden") {
  expected_files <- c(
    "reference_data.rds",
    "matrix_dataset.rds", 
    "fmri_series.rds",
    "sampling_frame.rds",
    "mock_neurvec.rds"
  )
  
  file_paths <- file.path(output_dir, expected_files)
  exists <- file.exists(file_paths)
  names(exists) <- expected_files
  
  if (all(exists)) {
    message("All golden test data files are present.")
  } else {
    missing <- expected_files[!exists]
    warning("Missing golden test data files: ", paste(missing, collapse = ", "))
  }
  
  invisible(exists)
}