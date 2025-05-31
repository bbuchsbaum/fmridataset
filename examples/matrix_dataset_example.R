# Matrix Dataset Example - fmrireg compatibility
# This example shows how to use the new matrix_dataset function for quick dataset creation

library(fmridataset)

# Example 1: Basic single-run dataset
cat("=== Example 1: Basic single-run dataset ===\n")
set.seed(123)
X <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)  # 100 timepoints, 50 voxels
dset1 <- matrix_dataset(X, TR = 2, run_length = 100)

cat("Dataset type:", get_dataset_type(dset1), "\n")
cat("Timepoints:", n_timepoints(dset1$sampling_frame), "\n")
cat("Voxels:", get_num_voxels(dset1), "\n")
cat("TR:", get_TR(dset1$sampling_frame), "\n")

# Example 2: Multi-run dataset
cat("\n=== Example 2: Multi-run dataset ===\n")
Y <- matrix(rnorm(200 * 30), nrow = 200, ncol = 30)  # 200 timepoints, 30 voxels
dset2 <- matrix_dataset(Y, TR = 1.5, run_length = c(100, 100))

cat("Number of runs:", n_runs(dset2$sampling_frame), "\n")
cat("Run lengths:", paste(get_run_lengths(dset2$sampling_frame), collapse = ", "), "\n")

# Example 3: With event table
cat("\n=== Example 3: With event table ===\n")
events <- data.frame(
  onset = c(10, 30, 60, 80),
  duration = c(2, 2, 3, 2),
  trial_type = c("A", "B", "A", "B")
)

Z <- matrix(rnorm(100 * 20), nrow = 100, ncol = 20)
dset3 <- matrix_dataset(Z, TR = 2, run_length = 100, event_table = events)

cat("Event table rows:", nrow(get_event_table(dset3)), "\n")
cat("Event types:", paste(unique(get_event_table(dset3)$trial_type), collapse = ", "), "\n")

# Example 4: Chunking compatibility (fmrireg style)
cat("\n=== Example 4: Data chunking (fmrireg style) ===\n")

# Voxel chunking
chunks_voxel <- data_chunks(dset2, nchunks = 4, by = "voxel")
cat("Voxel chunks:", attr(chunks_voxel, "total_chunks"), "\n")

chunk1 <- chunks_voxel$nextElem()
cat("First chunk dimensions:", nrow(chunk1$data), "x", ncol(chunk1$data), "\n")

# Run chunking (equivalent to runwise = TRUE in fmrireg)
chunks_run <- data_chunks(dset2, by = "run")
cat("Run chunks:", attr(chunks_run, "total_chunks"), "\n")

# Example 5: foreach compatibility
cat("\n=== Example 5: foreach compatibility ===\n")
library(foreach)

# Process chunks in parallel-ready loop
results <- foreach(chunk = chunks_voxel) %do% {
  list(
    chunk_num = chunk$chunk_num,
    mean_activation = mean(chunk$data),
    voxel_count = ncol(chunk$data)
  )
}

cat("Processed", length(results), "chunks\n")
cat("Chunk 1 mean activation:", round(results[[1]]$mean_activation, 3), "\n")

# Example 6: Print method demonstration
cat("\n=== Example 6: Print method ===\n")
print(dset1)

cat("\n=== Migration from fmrireg ===\n")
cat("Old fmrireg code:\n")
cat("  dset <- matrix_dataset(data_matrix, TR = 2, run_length = c(100, 100))\n")
cat("New fmridataset code:\n")  
cat("  dset <- matrix_dataset(data_matrix, TR = 2, run_length = c(100, 100))  # Same!\n")
cat("\nThe interface is identical for backward compatibility!\n") 