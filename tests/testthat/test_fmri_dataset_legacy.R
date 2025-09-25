library(testthat)
library(fmridataset)

# Test that fmri_dataset_legacy integrates with conversion utilities

test_that("fmri_dataset_legacy works and returns proper data", {
  with_mocked_bindings(
    file.exists = function(x) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        read_vol = function(x) {
          # Return a NeuroVol-like object with proper mask
          structure(
            array(c(TRUE, TRUE, TRUE), c(3, 1, 1)),
            dim = c(3, 1, 1),
            class = "NeuroVol"
          )
        },
        read_vec = function(x, mask, ...) {
          # Return a NeuroVec-like object
          structure(
            array(1:12, c(3, 1, 1, 4)),
            dim = c(3, 1, 1, 4),
            class = "NeuroVec"
          )
        },
        series = function(vec, inds) {
          # Extract time series for the given voxel indices
          # vec is 3x1x1x4, we want timepoints x voxels
          data <- as.vector(vec)
          matrix(data, nrow = 4, ncol = 3, byrow = FALSE)
        },
        read_header = function(x) {
          # Mock header info for dimensions
          header <- structure(
            list(
              dims = c(3, 1, 1, 4),
              spacing = c(1, 1, 1, 2),
              origin = c(0, 0, 0),
              spatial_axes = list(
                axis_1 = c(1, 0, 0),
                axis_2 = c(0, 1, 0),
                axis_3 = c(0, 0, 1)
              )
            ),
            class = "NIFTIMetaInfo"
          )
          # Add dim method for the mock object
          attr(header, "dim") <- function() c(3, 1, 1, 4)
          header
        },
        NeuroSpace = function(dim, spacing, origin, axes) {
          structure(list(dim = dim, spacing = spacing, origin = origin, axes = axes),
            class = "NeuroSpace"
          )
        },
        trans = function(x) diag(4),
        spacing = function(x) c(1, 1, 1, 2),
        space = function(x) x,
        origin = function(x) c(0, 0, 0),
        .package = "neuroim2",
        {
          with_mocked_bindings(
            backend_get_dims.nifti_backend = function(backend) {
              list(spatial = c(3, 1, 1), time = 4)
            },
            backend_get_mask.nifti_backend = function(backend) {
              rep(TRUE, 3)
            },
            validate_backend = function(backend) TRUE,
            backend_open.nifti_backend = function(backend) backend,
            .package = "fmridataset",
            {
              dset <- fmri_dataset_legacy(
                scans = "scan.nii",
                mask = "mask.nii",
                TR = 2,
                run_length = 4,
                preload = TRUE
              )
              expect_s3_class(dset, "fmri_dataset")
              # Test that we can get data matrix
              mat <- get_data_matrix(dset)
              expect_equal(dim(mat), c(4, 3))
              expect_equal(mat[1, 1], 1) # First timepoint, first voxel
            }
          )
        }
      )
    }
  )
})
