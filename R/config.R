#' @keywords internal
#' @noRd
default_config <- function() {
  env <- new.env()
  env$cmd_flags <- ""
  env$jobs <- 1
  env
}


#' read a basic fMRI configuration file
#'
#' @param file_name name of configuration file
#' @param base_path the file path to be prepended to relative file names
#' @importFrom assertthat assert_that
#' @importFrom tibble as_tibble
#' @importFrom utils read.table
#' @export
#' @return a \code{fmri_config} instance
read_fmri_config <- function(file_name, base_path = NULL) {
  # print(file_name)
  env <- default_config()

  source(file_name, env)

  env$base_path <- if (is.null(env$base_path) && is.null(base_path)) {
    "."
  } else if (!is.null(base_path) && is.null(env$base_path)) {
    base_path
  }

  if (is.null(env$output_dir)) {
    env$output_dir <- "stat_out"
  }


  assert_that(!is.null(env$scans))
  assert_that(!is.null(env$TR))
  assert_that(!is.null(env$mask))
  assert_that(!is.null(env$run_length))
  assert_that(!is.null(env$event_model))
  assert_that(!is.null(env$event_table))
  assert_that(!is.null(env$block_column))
  assert_that(!is.null(env$baseline_model))

  if (!is.null(env$censor_file)) {
    env$censor_file <- NULL
  }

  if (!is.null(env$contrasts)) {
    env$contrasts <- NULL
  }

  if (!is.null(env$nuisance)) {
    env$nuisance <- NULL
  }

  dname <- ifelse(
    is_absolute_path(env$event_table),
    env$event_table,
    file.path(env$base_path, env$event_table)
  )

  assert_that(file.exists(dname))
  env$design <- suppressMessages(tibble::as_tibble(read.table(dname, header = TRUE), .name_repair = "check_unique"))

  out <- as.list(env)
  class(out) <- c("fmri_config", "list")
  out
}
