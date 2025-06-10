library(testthat)
library(fmridataset)

# Test that fmri_dataset_legacy integrates with conversion utilities

test_that("fmri_dataset_legacy works with as.matrix_dataset", {
  with_mocked_bindings(
    file.exists = function(x) TRUE,
    read_vol = function(x) array(TRUE, c(3,1,1)),
    read_vec = function(x, ...) matrix(1:12, nrow = 4, ncol = 3),
    series = function(vec, inds) vec[, inds, drop = FALSE],
    .package = c("base", "neuroim2"),
    {
      dset <- fmri_dataset_legacy(
        scans = "scan.nii",
        mask = "mask.nii",
        TR = 2,
        run_length = 4,
        preload = TRUE
      )
      expect_s3_class(dset, "fmri_dataset")
      mat <- as.matrix_dataset(dset)
      expect_s3_class(mat, "matrix_dataset")
      expect_equal(dim(mat$datamat), c(4, 3))
    }
  )
})

