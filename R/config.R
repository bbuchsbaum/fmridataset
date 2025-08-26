#' @keywords internal
#' @noRd
default_config <- function() {
  list(
    cmd_flags = "",
    jobs = 1,
    base_path = ".",
    output_dir = "stat_out"
  )
}

#' read a basic fMRI configuration file
#'
#' @description
#' Reads a fMRI configuration file in YAML or JSON format. This replaces the
#' previous implementation that used source() for security reasons.
#'
#' @param file_name name of configuration file (YAML or JSON format)
#' @param base_path the file path to be prepended to relative file names
#' @importFrom assertthat assert_that
#' @importFrom fs is_absolute_path
#' @importFrom tibble as_tibble
#' @importFrom utils read.table modifyList
#' @export
#' @return a \code{fmri_config} instance
read_fmri_config <- function(file_name, base_path = NULL) {
  # Check if yaml is available for YAML files
  if (grepl("\\.ya?ml$", file_name, ignore.case = TRUE)) {
    if (!requireNamespace("yaml", quietly = TRUE)) {
      stop("Package 'yaml' is required to read YAML configuration files")
    }
    config_data <- yaml::read_yaml(file_name)
  } else if (grepl("\\.json$", file_name, ignore.case = TRUE)) {
    if (!requireNamespace("jsonlite", quietly = TRUE)) {
      stop("Package 'jsonlite' is required to read JSON configuration files")
    }
    config_data <- jsonlite::fromJSON(file_name, simplifyVector = TRUE)
  } else {
    # For backwards compatibility, try to read as dcf format
    config_data <- read_dcf_config(file_name)
  }

  # Merge with defaults
  config <- modifyList(default_config(), config_data)

  # Handle base_path - override if provided as parameter
  if (!is.null(base_path)) {
    config$base_path <- base_path
  }

  # Validate required fields
  required_fields <- c(
    "scans", "TR", "mask", "run_length", "event_model",
    "event_table", "block_column", "baseline_model"
  )

  missing_fields <- setdiff(required_fields, names(config))
  if (length(missing_fields) > 0) {
    stop(
      "Missing required configuration fields: ",
      paste(missing_fields, collapse = ", ")
    )
  }

  # Read event table
  dname <- ifelse(
    fs::is_absolute_path(config$event_table),
    config$event_table,
    file.path(config$base_path, config$event_table)
  )

  assert_that(file.exists(dname),
    msg = paste("Event table file not found:", dname)
  )

  config$design <- suppressMessages(
    tibble::as_tibble(read.table(dname, header = TRUE),
      .name_repair = "check_unique"
    )
  )

  class(config) <- c("fmri_config", "list")
  config
}

#' Read DCF-style configuration (backwards compatibility)
#' @keywords internal
read_dcf_config <- function(file_name) {
  lines <- readLines(file_name)
  config <- list()

  for (line in lines) {
    # Skip comments and empty lines
    if (grepl("^\\s*#", line) || grepl("^\\s*$", line)) next

    # Parse key-value pairs
    if (grepl("^\\s*\\w+\\s*[:=]", line)) {
      parts <- strsplit(line, "[:=]", perl = TRUE)[[1]]
      if (length(parts) >= 2) {
        key <- trimws(parts[1])
        value <- trimws(paste(parts[-1], collapse = ":"))

        # Try to parse the value
        parsed_value <- tryCatch(
          {
            # Try numeric first
            if (grepl("^[0-9.,-]+$", value)) {
              if (grepl(",", value)) {
                as.numeric(strsplit(value, ",")[[1]])
              } else {
                as.numeric(value)
              }
            } else if (value %in% c("TRUE", "FALSE")) {
              as.logical(value)
            } else {
              # Remove quotes if present
              gsub("^['\"]|['\"]$", "", value)
            }
          },
          error = function(e) value
        )

        config[[key]] <- parsed_value
      }
    }
  }

  config
}

#' Write fMRI configuration file
#'
#' @description
#' Writes a fMRI configuration to a YAML file for easy editing and sharing.
#'
#' @param config A fmri_config object or list with configuration parameters
#' @param file_name Output file name (should end in .yaml or .yml)
#' @export
write_fmri_config <- function(config, file_name) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package 'yaml' is required to write configuration files")
  }

  # Remove computed fields
  config_to_write <- config[!names(config) %in% c("design", "class")]

  yaml::write_yaml(config_to_write, file_name)
  invisible(config_to_write)
}
