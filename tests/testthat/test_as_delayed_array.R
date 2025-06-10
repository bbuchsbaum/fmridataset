library(testthat)

create_matrix_backend <- function() {
  matrix_backend(matrix(1:20, nrow = 5, ncol = 4))
}

test_that("as_delayed_array works for matrix_backend", {
  b <- create_matrix_backend()
  da <- as_delayed_array(b)
  expect_s4_class(da, "DelayedArray")
  expect_equal(dim(da), c(5, 4))
  expect_equal(as.matrix(da), b$data_matrix)
  sub <- da[2:4, 2:3]
  expect_equal(as.matrix(sub), b$data_matrix[2:4, 2:3])
})

create_nifti_backend <- function() {
  skip_if_not_installed("neuroim2")
  dims <- c(2,2,1,5)
  data_array <- array(seq_len(prod(dims)), dims)
  mock_vec <- structure(
    data_array,
    class = c("DenseNeuroVec", "NeuroVec", "array"),
    space = structure(list(dim = dims[1:3], origin = c(0,0,0), spacing = c(1,1,1)),
                     class = "NeuroSpace")
  )
  mock_mask <- structure(array(TRUE, dims[1:3]),
                         class = c("LogicalNeuroVol", "NeuroVol", "array"),
                         dim = dims[1:3])
  nifti_backend(source = list(mock_vec), mask_source = mock_mask, preload = TRUE)
}

test_that("as_delayed_array works for nifti_backend", {
  b <- create_nifti_backend()
  da <- as_delayed_array(b)
  expect_s4_class(da, "DelayedArray")
  expect_equal(dim(da), c(5, 4))
  expected <- matrix(seq_len(5*4), nrow = 5, byrow = TRUE)
  expect_equal(as.matrix(da), expected)
})
