#' Backend Registry System
#'
#' @description
#' A pluggable registry system for storage backends that allows external packages
#' to register new backend types without modifying the fmridataset package.
#' This enables extensibility while maintaining backward compatibility.
#'
#' @details
#' The registry system manages backend factories that create backend instances.
#' Each backend must implement the storage backend contract defined in
#' \code{\link{storage-backend}}.
#'
#' @name backend-registry
#' @keywords internal
NULL

# Package-level environment to store the backend registry
.backend_registry <- new.env(parent = emptyenv())

#' Register a Storage Backend
#'
#' @description
#' Registers a new storage backend type in the global registry.
#' External packages can use this function to add custom backend support.
#'
#' @param name Character string, unique name for the backend type
#' @param factory Function that creates backend instances, must accept \code{...} arguments
#' @param description Optional character string describing the backend
#' @param validate_function Optional function to validate backend instances beyond the standard contract
#' @param overwrite Logical, whether to overwrite existing registration (default: FALSE)
#'
#' @details
#' The factory function should:
#' \itemize{
#'   \item Accept all necessary parameters to create a backend instance
#'   \item Return an object that inherits from "storage_backend"
#'   \item Implement all required storage backend methods
#' }
#'
#' The validate_function should:
#' \itemize{
#'   \item Accept a backend object as first argument
#'   \item Return TRUE if valid, or throw informative error if invalid
#'   \item Perform any backend-specific validation beyond the standard contract
#' }
#'
#' @return Invisibly returns TRUE on successful registration
#' @export
#'
#' @examples
#' \dontrun{
#' # Register a custom backend
#' my_backend_factory <- function(source, ...) {
#'   # Create and return backend instance
#'   backend <- list(source = source, ...)
#'   class(backend) <- c("my_backend", "storage_backend")
#'   backend
#' }
#'
#' register_backend(
#'   name = "my_backend",
#'   factory = my_backend_factory,
#'   description = "Custom backend for my data format"
#' )
#' }
register_backend <- function(name, factory, description = NULL,
                             validate_function = NULL, overwrite = FALSE) {
  # Validate inputs
  if (!is.character(name) || length(name) != 1 || nchar(name) == 0) {
    stop_fmridataset(
      fmridataset_error_config,
      "name must be a non-empty character string"
    )
  }

  if (!is.function(factory)) {
    stop_fmridataset(
      fmridataset_error_config,
      "factory must be a function"
    )
  }

  if (!is.null(description) && (!is.character(description) || length(description) != 1)) {
    stop_fmridataset(
      fmridataset_error_config,
      "description must be a character string or NULL"
    )
  }

  if (!is.null(validate_function) && !is.function(validate_function)) {
    stop_fmridataset(
      fmridataset_error_config,
      "validate_function must be a function or NULL"
    )
  }

  # Check for existing registration
  if (exists(name, envir = .backend_registry) && !overwrite) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Backend '%s' is already registered. Use overwrite = TRUE to replace.", name)
    )
  }

  # Create registration entry
  registration <- list(
    name = name,
    factory = factory,
    description = description %||% paste("Backend:", name),
    validate_function = validate_function,
    registered_at = Sys.time()
  )

  # Store in registry
  assign(name, registration, envir = .backend_registry)

  invisible(TRUE)
}

#' Get Registered Backend Information
#'
#' @description
#' Retrieves information about a registered backend or lists all registered backends.
#'
#' @param name Character string, name of backend to query. If NULL, returns all registrations.
#'
#' @return For a specific backend: a list with registration details.
#'   For all backends: a named list where each element contains registration details.
#' @export
#'
#' @examples
#' # List all registered backends
#' get_backend_registry()
#'
#' # Get specific backend info
#' get_backend_registry("nifti")
get_backend_registry <- function(name = NULL) {
  if (is.null(name)) {
    # Return all registrations
    all_names <- ls(envir = .backend_registry)
    if (length(all_names) == 0) {
      return(list())
    }
    result <- vector("list", length(all_names))
    names(result) <- all_names
    for (nm in all_names) {
      result[[nm]] <- get(nm, envir = .backend_registry)
    }
    return(result)
  }

  # Return specific registration
  if (!exists(name, envir = .backend_registry)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf("Backend '%s' is not registered", name)
    )
  }

  get(name, envir = .backend_registry)
}

#' Check if Backend is Registered
#'
#' @description
#' Tests whether a backend type is registered in the system.
#'
#' @param name Character string, name of backend to check
#'
#' @return Logical, TRUE if backend is registered
#' @export
#'
#' @examples
#' is_backend_registered("nifti") # TRUE (built-in)
#' is_backend_registered("custom") # FALSE (unless registered)
is_backend_registered <- function(name) {
  if (!is.character(name) || length(name) != 1 || nchar(name) == 0) {
    return(FALSE)
  }
  exists(name, envir = .backend_registry)
}

#' Create Backend Instance
#'
#' @description
#' Creates a backend instance using the registered factory function.
#' This is the main interface for creating backends by name.
#'
#' @param name Character string, name of registered backend type
#' @param ... Arguments passed to the backend factory function
#' @param validate Logical, whether to validate the created backend (default: TRUE)
#'
#' @return A storage backend object
#' @export
#'
#' @examples
#' \dontrun{
#' # Create a NIfTI backend (assuming it's registered)
#' backend <- create_backend("nifti",
#'   source = "data.nii",
#'   mask_source = "mask.nii"
#' )
#'
#' # Create with validation disabled (faster, but riskier)
#' backend <- create_backend("nifti",
#'   source = "data.nii",
#'   mask_source = "mask.nii",
#'   validate = FALSE
#' )
#' }
create_backend <- function(name, ..., validate = TRUE) {
  if (!is_backend_registered(name)) {
    stop_fmridataset(
      fmridataset_error_config,
      sprintf(
        "Backend '%s' is not registered. Available backends: %s",
        name, paste(list_backend_names(), collapse = ", ")
      )
    )
  }

  registration <- get_backend_registry(name)

  # Create backend instance
  backend <- tryCatch(
    {
      registration$factory(...)
    },
    error = function(e) {
      stop_fmridataset(
        fmridataset_error_config,
        sprintf("Failed to create backend '%s': %s", name, e$message),
        call = sys.call(-1)
      )
    }
  )

  # Validate backend if requested
  if (validate) {
    validate_registered_backend(backend, registration)
  }

  backend
}

#' List Registered Backend Names
#'
#' @description
#' Returns a character vector of all registered backend names.
#'
#' @return Character vector of backend names
#' @export
#'
#' @examples
#' list_backend_names()
list_backend_names <- function() {
  sort(ls(envir = .backend_registry))
}

#' Unregister a Backend
#'
#' @description
#' Removes a backend from the registry. Use with caution as this may break
#' code that depends on the backend.
#'
#' @param name Character string, name of backend to remove
#'
#' @return Invisibly returns TRUE if backend was removed, FALSE if it wasn't registered
#' @export
#'
#' @examples
#' \dontrun{
#' # Remove a custom backend
#' unregister_backend("my_custom_backend")
#' }
unregister_backend <- function(name) {
  if (!is.character(name) || length(name) != 1) {
    stop_fmridataset(
      fmridataset_error_config,
      "name must be a character string"
    )
  }

  if (exists(name, envir = .backend_registry)) {
    rm(list = name, envir = .backend_registry)
    invisible(TRUE)
  } else {
    invisible(FALSE)
  }
}

#' Validate Registered Backend
#'
#' @description
#' Validates a backend instance using both the standard contract and any
#' backend-specific validation function.
#'
#' @param backend A storage backend object
#' @param registration Backend registration information (optional, will be looked up if not provided)
#'
#' @return TRUE if valid, otherwise throws an error
#' @keywords internal
validate_registered_backend <- function(backend, registration = NULL) {
  # Standard validation first
  validate_backend(backend)

  # Backend-specific validation if available
  if (is.null(registration)) {
    # Try to determine backend type from class
    backend_classes <- class(backend)
    storage_idx <- which(backend_classes == "storage_backend")
    if (length(storage_idx) > 0 && storage_idx > 1) {
      backend_type <- backend_classes[storage_idx - 1]
      if (is_backend_registered(backend_type)) {
        registration <- get_backend_registry(backend_type)
      }
    }
  }

  if (!is.null(registration) && !is.null(registration$validate_function)) {
    tryCatch(
      {
        registration$validate_function(backend)
      },
      error = function(e) {
        stop_fmridataset(
          fmridataset_error_config,
          sprintf("Backend-specific validation failed: %s", e$message)
        )
      }
    )
  }

  TRUE
}

#' Register Built-in Backends
#'
#' @description
#' Registers all built-in backend types. This is called automatically when
#' the package is loaded, but can be called manually if needed.
#'
#' @return Invisibly returns TRUE
#' @keywords internal
register_builtin_backends <- function() {
  # Register NIfTI backend
  register_backend(
    name = "nifti",
    factory = nifti_backend,
    description = "NIfTI format backend using neuroim2",
    overwrite = TRUE
  )

  # Register H5 backend
  register_backend(
    name = "h5",
    factory = h5_backend,
    description = "HDF5 format backend using fmristore",
    overwrite = TRUE
  )

  # Register Matrix backend
  register_backend(
    name = "matrix",
    factory = matrix_backend,
    description = "In-memory matrix backend",
    overwrite = TRUE
  )

  # Register Latent backend
  register_backend(
    name = "latent",
    factory = latent_backend,
    description = "Latent space backend for dimension-reduced data",
    overwrite = TRUE
  )

  # Register Study backend
  register_backend(
    name = "study",
    factory = study_backend,
    description = "Multi-subject study backend",
    overwrite = TRUE
  )

  # Register Zarr backend (if available)
  register_backend(
    name = "zarr",
    factory = zarr_backend,
    description = "Zarr format backend",
    overwrite = TRUE
  )

  invisible(TRUE)
}

#' Print Backend Registry
#'
#' @description
#' Prints a formatted summary of registered backends.
#'
#' @param x Result from \code{get_backend_registry()}
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns the input
#' @export
print.backend_registry <- function(x, ...) {
  if (length(x) == 0) {
    cat("No backends registered\n")
    return(invisible(x))
  }

  cat("Registered Storage Backends:\n")
  cat("===========================\n\n")

  for (name in names(x)) {
    reg <- x[[name]]
    cat(sprintf("Backend: %s\n", name))
    cat(sprintf("  Description: %s\n", reg$description))
    cat(sprintf("  Registered: %s\n", format(reg$registered_at, "%Y-%m-%d %H:%M:%S")))
    if (!is.null(reg$validate_function)) {
      cat("  Custom validation: Yes\n")
    }
    cat("\n")
  }

  invisible(x)
}

# Helper function for null coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x
