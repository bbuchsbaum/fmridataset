# Modular Data Transformation System for fmridataset
# Provides a pluggable, extensible interface for data transformations

# ============================================================================
# Base Transformation Interface
# ============================================================================

#' Create a data transformation
#' 
#' Base constructor for creating transformation objects that can be applied
#' to fMRI data in a modular, composable way.
#' 
#' @param name Character name of the transformation
#' @param fn Function that takes a matrix and returns a transformed matrix
#' @param params List of parameters for the transformation
#' @param description Optional description of what the transformation does
#' @param ... Additional metadata
#' 
#' @return A transformation object
#' @export
transformation <- function(name, fn, params = list(), description = NULL, ...) {
  structure(
    list(
      name = name,
      fn = fn,
      params = params,
      description = description,
      metadata = list(...)
    ),
    class = "fmri_transformation"
  )
}

#' Check if object is a transformation
#' @param x Object to check
#' @export
is.transformation <- function(x) {
  inherits(x, "fmri_transformation")
}

#' Apply a transformation to data
#' 
#' @param transform A transformation object
#' @param data Matrix to transform (timepoints x voxels)
#' @param ... Additional arguments passed to transformation function
#' @export
apply_transformation <- function(transform, data, ...) {
  if (!is.transformation(transform)) {
    stop("Object is not a valid transformation")
  }
  
  if (!is.matrix(data)) {
    stop("Data must be a matrix")
  }
  
  # Apply the transformation function with its parameters
  do.call(transform$fn, c(list(data = data), transform$params, list(...)))
}

#' Print method for transformations
#' @param x A transformation object
#' @param ... Additional arguments (ignored)
#' @export
print.fmri_transformation <- function(x, ...) {
  cat("fMRI Transformation:", x$name, "\n")
  if (!is.null(x$description)) {
    cat("Description:", x$description, "\n")
  }
  if (length(x$params) > 0) {
    cat("Parameters:\n")
    for (name in names(x$params)) {
      cat("  ", name, ":", x$params[[name]], "\n")
    }
  }
  invisible(x)
}

# ============================================================================
# Transformation Pipeline
# ============================================================================

#' Create a transformation pipeline
#' 
#' Compose multiple transformations into a pipeline that can be applied
#' sequentially to fMRI data.
#' 
#' @param ... Transformation objects or a list of transformations
#' 
#' @return A transformation pipeline object
#' @export
transformation_pipeline <- function(...) {
  transforms <- list(...)
  
  # Handle case where a single list is passed
  if (length(transforms) == 1 && is.list(transforms[[1]]) && 
      !is.transformation(transforms[[1]])) {
    transforms <- transforms[[1]]
  }
  
  # Validate all are transformations
  for (i in seq_along(transforms)) {
    if (!is.transformation(transforms[[i]])) {
      stop("All elements must be transformation objects")
    }
  }
  
  structure(
    list(
      transformations = transforms,
      n_transforms = length(transforms)
    ),
    class = "fmri_transformation_pipeline"
  )
}

#' Check if object is a transformation pipeline
#' @param x Object to check
#' @export
is.transformation_pipeline <- function(x) {
  inherits(x, "fmri_transformation_pipeline")
}

#' Apply a transformation pipeline to data
#' 
#' @param pipeline A transformation pipeline object
#' @param data Matrix to transform (timepoints x voxels)
#' @param verbose Logical indicating whether to print progress
#' @param ... Additional arguments passed to transformations
#' @export
apply_pipeline <- function(pipeline, data, verbose = FALSE, ...) {
  if (!is.transformation_pipeline(pipeline)) {
    stop("Object is not a valid transformation pipeline")
  }
  
  if (pipeline$n_transforms == 0) {
    return(data)
  }
  
  current_data <- data
  
  for (i in seq_along(pipeline$transformations)) {
    transform <- pipeline$transformations[[i]]
    
    if (verbose) {
      cat("Applying transformation", i, "of", pipeline$n_transforms, ":", 
          transform$name, "\n")
    }
    
    current_data <- apply_transformation(transform, current_data, ...)
  }
  
  current_data
}

#' Print method for transformation pipelines
#' @param x A transformation pipeline object
#' @param ... Additional arguments (ignored)
#' @export
print.fmri_transformation_pipeline <- function(x, ...) {
  cat("fMRI Transformation Pipeline with", x$n_transforms, "transformations:\n")
  for (i in seq_along(x$transformations)) {
    cat("  ", i, ". ", x$transformations[[i]]$name, "\n", sep = "")
  }
  invisible(x)
}

# ============================================================================
# Built-in Transformations
# ============================================================================

#' Temporal z-score normalization
#' 
#' Standardizes each voxel's time series to have mean 0 and SD 1
#' 
#' @param remove_mean Logical indicating whether to center the data
#' @param remove_trend Logical indicating whether to remove linear trend first
#' @export
transform_temporal_zscore <- function(remove_mean = TRUE, remove_trend = FALSE) {
  transformation(
    name = "temporal_zscore",
    description = "Temporal z-score normalization (mean=0, sd=1 per voxel)",
    params = list(remove_mean = remove_mean, remove_trend = remove_trend),
    fn = function(data, remove_mean, remove_trend) {
      if (remove_trend) {
        # Remove linear trend first
        time_vec <- seq_len(nrow(data))
        for (j in seq_len(ncol(data))) {
          lm_fit <- lm(data[, j] ~ time_vec)
          data[, j] <- residuals(lm_fit)
        }
      }
      
      # Z-score normalization
      if (remove_mean) {
        scale(data, center = TRUE, scale = TRUE)
      } else {
        # Just scale by SD, don't center
        sds <- apply(data, 2, sd, na.rm = TRUE)
        sds[sds == 0] <- 1  # Avoid division by zero
        sweep(data, 2, sds, "/")
      }
    }
  )
}

#' Voxelwise linear detrending
#' 
#' Removes linear trend from each voxel's time series
#' 
#' @param method Character indicating detrending method ("linear", "quadratic")
#' @export
transform_detrend <- function(method = "linear") {
  transformation(
    name = "voxelwise_detrend", 
    description = paste("Voxelwise", method, "detrending"),
    params = list(method = method),
    fn = function(data, method) {
      time_vec <- seq_len(nrow(data))
      
      detrended <- data
      for (j in seq_len(ncol(data))) {
        if (method == "linear") {
          lm_fit <- lm(data[, j] ~ time_vec)
        } else if (method == "quadratic") {
          lm_fit <- lm(data[, j] ~ time_vec + I(time_vec^2))
        } else {
          stop("Method must be 'linear' or 'quadratic'")
        }
        detrended[, j] <- residuals(lm_fit)
      }
      detrended
    }
  )
}

#' Temporal smoothing
#' 
#' Apply temporal smoothing to reduce noise
#' 
#' @param window_size Integer size of smoothing window
#' @param method Character smoothing method ("gaussian", "box", "median")
#' @export
transform_temporal_smooth <- function(window_size = 3, method = "gaussian") {
  transformation(
    name = "temporal_smooth",
    description = paste("Temporal", method, "smoothing, window size:", window_size),
    params = list(window_size = window_size, method = method),
    fn = function(data, window_size, method) {
      smoothed <- data
      
      if (method == "box") {
        # Simple box car smoothing
        for (j in seq_len(ncol(data))) {
          smoothed[, j] <- stats::filter(data[, j], rep(1/window_size, window_size), 
                                       sides = 2, circular = FALSE)
        }
      } else if (method == "gaussian") {
        # Gaussian smoothing
        sigma <- window_size / 3
        kernel <- dnorm(seq(-window_size, window_size), sd = sigma)
        kernel <- kernel / sum(kernel)
        
        for (j in seq_len(ncol(data))) {
          smoothed[, j] <- stats::filter(data[, j], kernel, sides = 2, circular = FALSE)
        }
      } else if (method == "median") {
        # Median smoothing
        for (j in seq_len(ncol(data))) {
          smoothed[, j] <- stats::runmed(data[, j], k = window_size)
        }
      } else {
        stop("Method must be 'gaussian', 'box', or 'median'")
      }
      
      # Handle NAs from filtering
      smoothed[is.na(smoothed)] <- data[is.na(smoothed)]
      smoothed
    }
  )
}

#' High-pass filtering
#' 
#' Remove low-frequency components
#' 
#' @param cutoff_freq Numeric cutoff frequency in Hz
#' @param TR Numeric repetition time in seconds
#' @param order Integer filter order
#' @export
transform_highpass <- function(cutoff_freq, TR, order = 4) {
  transformation(
    name = "highpass_filter",
    description = paste("High-pass filter at", cutoff_freq, "Hz"),
    params = list(cutoff_freq = cutoff_freq, TR = TR, order = order),
    fn = function(data, cutoff_freq, TR, order) {
      # Simple high-pass filtering using signal processing
      # This is a basic implementation - more sophisticated filtering 
      # could be added as additional transformation options
      
      nyquist <- 0.5 / TR
      normalized_cutoff <- cutoff_freq / nyquist
      
      if (requireNamespace("signal", quietly = TRUE)) {
        # Use signal package if available
        bf <- signal::butter(order, normalized_cutoff, type = "high")
        filtered <- data
        for (j in seq_len(ncol(data))) {
          filtered[, j] <- signal::filtfilt(bf, data[, j])
        }
        filtered
      } else {
        # Fallback: simple detrending
        warning("signal package not available, using simple detrending instead")
        time_vec <- seq_len(nrow(data))
        detrended <- data
        for (j in seq_len(ncol(data))) {
          lm_fit <- lm(data[, j] ~ time_vec + I(time_vec^2))
          detrended[, j] <- residuals(lm_fit)
        }
        detrended
      }
    }
  )
}

#' Outlier removal/replacement
#' 
#' Detect and handle outliers in the time series
#' 
#' @param method Character method for outlier detection ("zscore", "iqr", "mad")
#' @param threshold Numeric threshold for outlier detection
#' @param replace_method Character method for replacing outliers ("interpolate", "median", "mean")
#' @export
transform_outlier_removal <- function(method = "zscore", threshold = 3, 
                                     replace_method = "interpolate") {
  transformation(
    name = "outlier_removal",
    description = paste("Outlier removal using", method, "method, threshold:", threshold),
    params = list(method = method, threshold = threshold, replace_method = replace_method),
    fn = function(data, method, threshold, replace_method) {
      cleaned <- data
      
      for (j in seq_len(ncol(data))) {
        ts <- data[, j]
        
        # Detect outliers
        if (method == "zscore") {
          z_scores <- abs(scale(ts))
          outliers <- z_scores > threshold
        } else if (method == "iqr") {
          q75 <- quantile(ts, 0.75, na.rm = TRUE)
          q25 <- quantile(ts, 0.25, na.rm = TRUE)
          iqr <- q75 - q25
          outliers <- ts < (q25 - threshold * iqr) | ts > (q75 + threshold * iqr)
        } else if (method == "mad") {
          med <- median(ts, na.rm = TRUE)
          mad_val <- mad(ts, na.rm = TRUE)
          outliers <- abs(ts - med) > threshold * mad_val
        }
        
        # Replace outliers
        if (any(outliers, na.rm = TRUE)) {
          if (replace_method == "interpolate") {
            # Linear interpolation
            outlier_indices <- which(outliers)
            good_indices <- which(!outliers)
            if (length(good_indices) > 1) {
              cleaned[outlier_indices, j] <- approx(good_indices, 
                                                  ts[good_indices], 
                                                  outlier_indices, 
                                                  rule = 2)$y
            }
          } else if (replace_method == "median") {
            cleaned[outliers, j] <- median(ts[!outliers], na.rm = TRUE)
          } else if (replace_method == "mean") {
            cleaned[outliers, j] <- mean(ts[!outliers], na.rm = TRUE)
          }
        }
      }
      
      cleaned
    }
  )
}

# ============================================================================
# Integration with fmri_dataset
# ============================================================================

#' Create transformation pipeline from legacy options
#' 
#' This function provides backwards compatibility by converting the old
#' hardcoded preprocessing options to the new transformation system.
#' 
#' @param temporal_zscore Logical for temporal z-scoring
#' @param voxelwise_detrend Logical for voxelwise detrending
#' @param ... Additional legacy options
#' @keywords internal
create_legacy_pipeline <- function(temporal_zscore = FALSE, 
                                 voxelwise_detrend = FALSE, ...) {
  transforms <- list()
  
  if (voxelwise_detrend) {
    transforms <- append(transforms, list(transform_detrend()))
  }
  
  if (temporal_zscore) {
    transforms <- append(transforms, list(transform_temporal_zscore()))
  }
  
  if (length(transforms) == 0) {
    return(NULL)
  }
  
  transformation_pipeline(transforms)
}

#' Get transformation pipeline from dataset
#' 
#' @param dataset An fmri_dataset object
#' @export
get_transformation_pipeline <- function(dataset) {
  if (!is.fmri_dataset(dataset)) {
    stop("Object is not an fmri_dataset")
  }
  
  # Check for new-style pipeline first
  if (!is.null(dataset$transformation_pipeline)) {
    return(dataset$transformation_pipeline)
  }
  
  # Fall back to legacy options for backwards compatibility
  matrix_opts <- dataset$metadata$matrix_options
  if (!is.null(matrix_opts)) {
    return(create_legacy_pipeline(
      temporal_zscore = isTRUE(matrix_opts$temporal_zscore),
      voxelwise_detrend = isTRUE(matrix_opts$voxelwise_detrend)
    ))
  }
  
  NULL
}

#' Set transformation pipeline for dataset
#' 
#' @param dataset An fmri_dataset object
#' @param pipeline A transformation pipeline or NULL
#' @export
set_transformation_pipeline <- function(dataset, pipeline) {
  if (!is.fmri_dataset(dataset)) {
    stop("Object is not an fmri_dataset")
  }
  
  if (!is.null(pipeline) && !is.transformation_pipeline(pipeline)) {
    stop("Pipeline must be a transformation_pipeline object or NULL")
  }
  
  dataset$transformation_pipeline <- pipeline
  dataset
} 