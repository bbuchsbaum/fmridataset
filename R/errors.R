#' Custom Error Classes for fmridataset
#'
#' @description
#' A hierarchy of custom S3 error classes for the fmridataset package.
#' These provide structured error handling for storage backend operations.
#'
#' @name fmridataset-errors
#' @keywords internal
NULL

#' Create a Custom fmridataset Error
#'
#' @param message Character string describing the error
#' @param class Character vector of error classes
#' @param ... Additional data to include in the error condition
#' @return A condition object
#' @keywords internal
fmridataset_error <- function(message, class = character(), ...) {
  structure(
    list(message = message, ...),
    class = c(class, "fmridataset_error", "error", "condition")
  )
}

#' Backend I/O Error
#'
#' @description
#' Raised when a storage backend encounters read/write failures.
#'
#' @param message Character string describing the I/O error
#' @param file Path to the file that caused the error (optional)
#' @param operation The operation that failed (e.g., "read", "write")
#' @param ... Additional context
#' @return A backend I/O error condition
#' @keywords internal
fmridataset_error_backend_io <- function(message, file = NULL, operation = NULL, ...) {
  fmridataset_error(
    message = message,
    class = "fmridataset_error_backend_io",
    file = file,
    operation = operation,
    ...
  )
}

#' Configuration Error
#'
#' @description
#' Raised when invalid configuration is provided to a backend or dataset.
#'
#' @param message Character string describing the configuration error
#' @param parameter The parameter that was invalid
#' @param value The invalid value provided
#' @param ... Additional context
#' @return A configuration error condition
#' @keywords internal
fmridataset_error_config <- function(message, parameter = NULL, value = NULL, ...) {
  fmridataset_error(
    message = message,
    class = "fmridataset_error_config",
    parameter = parameter,
    value = value,
    ...
  )
}

#' Stop with a Custom Error
#'
#' @param error_fn Error constructor function
#' @param ... Arguments passed to the error constructor
#' @keywords internal
stop_fmridataset <- function(error_fn, ...) {
  err <- error_fn(...)
  stop(err)
}