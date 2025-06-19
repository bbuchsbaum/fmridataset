# Example usage of series_selector classes
library(fmridataset)

# Create a simple test dataset
mat <- matrix(rnorm(100 * 50), nrow = 100, ncol = 50)
backend <- matrix_backend(mat, mask = rep(TRUE, 50), spatial_dims = c(5, 10, 1))
dset <- fmri_dataset(backend, TR = 2, run_length = 100)

# 1. Index selector - select specific voxel indices
cat("\n1. Index selector example:\n")
sel1 <- index_selector(c(1, 5, 10, 20))
print(sel1)
fs1 <- fmri_series(dset, selector = sel1)
cat("Selected", ncol(fs1), "voxels\n")

# 2. Voxel selector - select by 3D coordinates
cat("\n2. Voxel selector example:\n")
coords <- cbind(x = c(1, 3), y = c(2, 5), z = c(1, 1))
sel2 <- voxel_selector(coords)
print(sel2)
fs2 <- fmri_series(dset, selector = sel2)
cat("Selected", ncol(fs2), "voxels\n")

# 3. All selector - select all voxels
cat("\n3. All selector example:\n")
sel3 <- all_selector()
print(sel3)
fs3 <- fmri_series(dset, selector = sel3)
cat("Selected", ncol(fs3), "voxels (all)\n")

# 4. Sphere selector - select voxels within a sphere
cat("\n4. Sphere selector example:\n")
sel4 <- sphere_selector(center = c(3, 5, 1), radius = 2)
print(sel4)
fs4 <- fmri_series(dset, selector = sel4)
cat("Selected", ncol(fs4), "voxels in sphere\n")

# 5. ROI selector - select voxels in a region
cat("\n5. ROI selector example:\n")
roi <- array(FALSE, dim = c(5, 10, 1))
roi[2:4, 3:7, 1] <- TRUE
sel5 <- roi_selector(roi)
print(sel5)
fs5 <- fmri_series(dset, selector = sel5)
cat("Selected", ncol(fs5), "voxels in ROI\n")

# 6. Mask selector - select voxels by logical mask
cat("\n6. Mask selector example:\n")
mask_vec <- rep(FALSE, 50)
mask_vec[seq(1, 50, by = 5)] <- TRUE
sel6 <- mask_selector(mask_vec)
print(sel6)
fs6 <- fmri_series(dset, selector = sel6)
cat("Selected", ncol(fs6), "voxels by mask\n")

# Compare with legacy selector usage
cat("\n7. Legacy selector comparison:\n")
legacy_fs <- fmri_series(dset, selector = c(1, 5, 10, 20))
new_fs <- fmri_series(dset, selector = index_selector(c(1, 5, 10, 20)))
cat("Legacy and new selectors produce same result:", 
    identical(dim(legacy_fs), dim(new_fs)), "\n")