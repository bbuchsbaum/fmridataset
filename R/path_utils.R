#' Check if a file path is absolute
#'
#' Utility function to determine whether a path is absolute on
#' either Unix or Windows platforms.
#' @param paths Character vector of file paths.
#' @return Logical vector indicating which paths are absolute.
#' @keywords internal
#' @noRd
is_absolute_path <- function(paths) {
  grepl("^(/|[A-Za-z]:)", paths)
}
