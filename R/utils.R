#' Null-coalescing operator
#'
#' If x is NULL, return y; otherwise return x
#' @name grapes-or-or-grapes
#' @keywords internal
#' @export
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Validate backend object
#'
#' Internal function to validate storage backend objects
#' @param backend A storage backend object
#' @keywords internal
validate_backend <- function(backend) {
  if (!inherits(backend, "storage_backend")) {
    stop("Invalid backend object: must inherit from 'storage_backend'", call. = FALSE)
  }
  
  # Check for required methods
  required_methods <- c("backend_open", "backend_close", "backend_get_dims", 
                       "backend_get_mask", "backend_get_data")
  
  for (method in required_methods) {
    if (!hasMethod(method, class(backend)[1])) {
      stop(sprintf("Backend class '%s' must implement method '%s'", 
                   class(backend)[1], method), call. = FALSE)
    }
  }
  
  # Validate dimensions
  dims <- backend_get_dims(backend)
  if (!is.list(dims) || !all(c("spatial", "time") %in% names(dims))) {
    stop("backend_get_dims must return a list with 'spatial' and 'time' elements", call. = FALSE)
  }
  
  if (length(dims$spatial) != 3 || !is.numeric(dims$spatial)) {
    stop("spatial dimensions must be a numeric vector of length 3", call. = FALSE)
  }
  
  if (!is.numeric(dims$time) || length(dims$time) != 1 || dims$time <= 0) {
    stop("time dimension must be a positive integer", call. = FALSE)
  }
  
  # Validate mask
  mask <- backend_get_mask(backend)
  if (!is.logical(mask)) {
    stop("mask must be a logical vector", call. = FALSE)
  }
  
  if (any(is.na(mask))) {
    stop("missing value where TRUE/FALSE needed", call. = FALSE)
  }
  
  if (!any(mask)) {
    stop("mask must contain at least one TRUE value", call. = FALSE)
  }
  
  expected_mask_length <- prod(dims$spatial)
  if (length(mask) != expected_mask_length) {
    stop(sprintf("mask length (%d) must equal prod(spatial_dims) (%d)", 
                 length(mask), expected_mask_length), call. = FALSE)
  }
  
  TRUE
}

#' Check if a method exists for a given class
#'
#' Internal helper to check S3 method existence
#' @param generic Generic function name
#' @param class Class name
#' @keywords internal
hasMethod <- function(generic, class) {
  method_name <- paste0(generic, ".", class)
  exists(method_name, mode = "function") || 
    !is.null(utils::getS3method(generic, class, optional = TRUE))
}