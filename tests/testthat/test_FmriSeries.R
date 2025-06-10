library(DelayedArray)
library(S4Vectors)

# Basic instantiation

test_that("FmriSeries can be created and displayed", {
  mat <- DelayedArray(matrix(1:6, nrow = 3))
  vox_info <- DataFrame(id = 1:ncol(mat))
  tmp_info <- DataFrame(id = 1:nrow(mat))
  fs <- new("FmriSeries",
            mat,
            voxel_info = vox_info,
            temporal_info = tmp_info,
            selection_info = list(selector = NULL),
            dataset_info = list(backend_type = "matrix_backend"))
  expect_s4_class(fs, "FmriSeries")

  out <- capture.output(show(fs))
  expect_true(any(grepl("Orientation: time", out)))
})
