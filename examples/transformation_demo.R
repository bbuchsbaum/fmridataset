#!/usr/bin/env Rscript

# Demonstration of the Modular Transformation System
# Shows how to create and use custom transformation pipelines

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("fmridataset Transformation System Demo\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Load the package (assuming we're in development)
if (file.exists("R/aaa_generics.R")) {
  source("R/aaa_generics.R")
  source("R/utils.R") 
  source("R/sampling_frame.R")
  source("R/transformations.R")
  source("R/fmri_dataset_class.R")
  source("R/fmri_dataset_create.R")
  source("R/fmri_dataset_accessors.R")
}

suppressMessages(library(tibble))
suppressMessages(library(methods))

cat("\n1. Creating Individual Transformations\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Create individual transformations
zscore_transform <- transform_temporal_zscore(remove_trend = TRUE)
detrend_transform <- transform_detrend(method = "linear")
smooth_transform <- transform_temporal_smooth(window_size = 3, method = "gaussian")
outlier_transform <- transform_outlier_removal(method = "zscore", threshold = 3)

cat("Created transformations:\n")
print(zscore_transform)
print(detrend_transform)

cat("\n2. Creating Transformation Pipelines\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Basic preprocessing pipeline
basic_pipeline <- transformation_pipeline(
  detrend_transform,
  zscore_transform
)

# Advanced preprocessing pipeline
advanced_pipeline <- transformation_pipeline(
  detrend_transform,
  outlier_transform,
  smooth_transform,
  zscore_transform
)

cat("Basic pipeline:\n")
print(basic_pipeline)

cat("\nAdvanced pipeline:\n")
print(advanced_pipeline)

cat("\n3. Testing with Sample Data\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Generate sample data with trend and noise
set.seed(123)
n_timepoints <- 100
n_voxels <- 5

# Create data with linear trend and some outliers
sample_data <- matrix(nrow = n_timepoints, ncol = n_voxels)
time_vec <- seq_len(n_timepoints)

for (i in seq_len(n_voxels)) {
  # Base signal with trend
  signal <- 0.02 * time_vec + rnorm(n_timepoints, sd = 0.5)
  
  # Add some outliers
  outlier_indices <- sample(n_timepoints, 3)
  signal[outlier_indices] <- signal[outlier_indices] + rnorm(3, mean = 0, sd = 3)
  
  sample_data[, i] <- signal
}

cat("Original data summary:\n")
cat("  Dimensions:", nrow(sample_data), "×", ncol(sample_data), "\n")
cat("  Mean values:", paste(round(apply(sample_data, 2, mean), 3), collapse = ", "), "\n")
cat("  SD values:", paste(round(apply(sample_data, 2, sd), 3), collapse = ", "), "\n")

cat("\n4. Applying Basic Pipeline\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

basic_processed <- apply_pipeline(basic_pipeline, sample_data, verbose = TRUE)

cat("After basic processing:\n")
cat("  Mean values:", paste(round(apply(basic_processed, 2, mean), 3), collapse = ", "), "\n")
cat("  SD values:", paste(round(apply(basic_processed, 2, sd), 3), collapse = ", "), "\n")

cat("\n5. Applying Advanced Pipeline\n")  
cat(paste(rep("-", 40), collapse = ""), "\n")

advanced_processed <- apply_pipeline(advanced_pipeline, sample_data, verbose = TRUE)

cat("After advanced processing:\n")
cat("  Mean values:", paste(round(apply(advanced_processed, 2, mean), 3), collapse = ", "), "\n")
cat("  SD values:", paste(round(apply(advanced_processed, 2, sd), 3), collapse = ", "), "\n")

cat("\n6. Integration with fmri_dataset\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Create dataset with custom pipeline
dataset_with_pipeline <- fmri_dataset_create(
  images = sample_data,
  TR = 2.0,
  run_lengths = c(60, 40),
  transformation_pipeline = advanced_pipeline
)

cat("Created dataset with transformation pipeline\n")
print(dataset_with_pipeline)

# Get data with and without transformations
raw_data <- get_data_matrix(dataset_with_pipeline, apply_transformations = FALSE)
processed_data <- get_data_matrix(dataset_with_pipeline, apply_transformations = TRUE)

cat("\nComparison:\n")
cat("Raw data - Mean:", round(mean(raw_data), 3), "SD:", round(sd(raw_data), 3), "\n")
cat("Processed - Mean:", round(mean(processed_data), 3), "SD:", round(sd(processed_data), 3), "\n")

cat("\n7. Custom Transformation Example\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Create a custom transformation
custom_log_transform <- transformation(
  name = "log_transform",
  description = "Log transformation after shifting to positive values",
  params = list(shift_value = 10),
  fn = function(data, shift_value) {
    shifted_data <- data + shift_value
    log(shifted_data)
  }
)

print(custom_log_transform)

# Create pipeline with custom transformation
custom_pipeline <- transformation_pipeline(
  detrend_transform,
  custom_log_transform,
  zscore_transform
)

custom_processed <- apply_pipeline(custom_pipeline, abs(sample_data) + 1, verbose = TRUE)
cat("Custom pipeline applied successfully!\n")

cat("\n8. Performance Comparison\n")
cat(paste(rep("-", 40), collapse = ""), "\n")

# Compare old vs new approach
cat("Old approach (hardcoded):\n")
start_time <- Sys.time()
old_processed <- scale(sample_data, center = TRUE, scale = TRUE)
old_time <- as.numeric(Sys.time() - start_time)
cat("  Time:", round(old_time * 1000, 2), "ms\n")

cat("New approach (pipeline):\n")
start_time <- Sys.time()
new_processed <- apply_pipeline(basic_pipeline, sample_data)
new_time <- as.numeric(Sys.time() - start_time)
cat("  Time:", round(new_time * 1000, 2), "ms\n")

cat("Overhead factor:", round(new_time / old_time, 1), "x\n")

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Transformation System Demo Complete!\n")
cat("\nKey Benefits:\n")
cat("✓ Modular and extensible\n")
cat("✓ Composable pipelines\n") 
cat("✓ Custom transformations supported\n")
cat("✓ Backwards compatible\n")
cat("✓ Easy to understand and debug\n")
cat(paste(rep("=", 60), collapse = ""), "\n") 