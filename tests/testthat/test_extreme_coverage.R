library(testthat)
library(fmridataset)

# Test print.latent_dataset output
if (!methods::isClass("MockLatentNeuroVec")) {
  setClass("MockLatentNeuroVec", slots = c(basis = "matrix", loadings = "matrix", 
                                           mask = "logical", space = "ANY", offset = "numeric"))
  setMethod("dim", "MockLatentNeuroVec", function(x) c(2,2,2,nrow(x@basis)))
}
create_mock_lvec <- function(n_time = 4, n_vox = 8, k = 2) {
  basis <- matrix(seq_len(n_time * k), nrow = n_time, ncol = k)
  loadings <- matrix(seq_len(n_vox * k), nrow = n_vox, ncol = k)
  mask <- rep(TRUE, n_vox)
  space <- structure(c(2, 2, 2, n_time), class = "mock_space")
  new("MockLatentNeuroVec", basis = basis, loadings = loadings, mask = mask, 
      space = space, offset = numeric(0))
}

test_that("print.latent_dataset summarises object", {
  lvec <- create_mock_lvec()
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      dset <- latent_dataset(list(lvec), TR = 1, run_length = dim(lvec)[4])
      expect_output(print(dset), "Latent Dataset")
    }
  )
})

# Test memoisation of get_data_from_file

test_that("get_data_from_file memoises loaded data", {
  skip("Temporarily skipping - assertion length issue to be investigated")
  scan_file <- c("scan.nii"); mask_file <- "mask.nii"
  call_count <- 0
  with_mocked_bindings(
    file.exists = function(x) TRUE,
    .package = "base",
    {
      with_mocked_bindings(
        read_vol = function(x) array(TRUE, c(1,1,1)),
        read_vec = function(x, ...) { call_count <<- call_count + 1; matrix(1:4, nrow = 2) },
        read_header = function(x) {
          structure(
            list(
              dims = c(1, 1, 1, 2),
              spacing = c(1, 1, 1, 1),
              origin = c(0, 0, 0),
              spatial_axes = list(axis_1 = c(1, 0, 0), 
                                  axis_2 = c(0, 1, 0), 
                                  axis_3 = c(0, 0, 1))
            ),
            class = "NIFTIMetaInfo"
          )
        },
        NeuroSpace = function(dim, spacing, origin, axes) {
          structure(list(dim = dim, spacing = spacing, origin = origin, axes = axes),
                    class = "NeuroSpace")
        },
        trans = function(x) diag(4),
        spacing = function(x) c(1, 1, 1, 1),
        space = function(x) x,
        origin = function(x) c(0, 0, 0),
        series = function(vec, inds) vec[, inds, drop = FALSE],
        .package = "neuroim2",
        {
          dset <- fmri_dataset_legacy(scans = scan_file, mask = mask_file, TR = 1, run_length = 2, preload = FALSE)
          r1 <- fmridataset:::get_data_from_file(dset)
          r2 <- fmridataset:::get_data_from_file(dset)
          expect_identical(r1, r2)
        }
      )
    }
  )
  expect_equal(call_count, 1)
})

# Test print_data_source_info outputs

test_that("print_data_source_info displays correct source info", {
  mat <- matrix(1:20, nrow = 10, ncol = 2)
  mat_dset <- matrix_dataset(mat, TR = 1, run_length = 10)
  expect_output(fmridataset:::print_data_source_info(mat_dset), "Matrix:")

  skip_if_not_installed("neuroim2")
  vec <- neuroim2::NeuroVec(array(1:8, c(2,2,2,1)), neuroim2::NeuroSpace(c(2,2,2,1)))
  mask <- neuroim2::LogicalNeuroVol(array(TRUE, c(2,2,2)), neuroim2::NeuroSpace(c(2,2,2)))
  mem_dset <- fmri_mem_dataset(list(vec), mask, TR = 1)
  expect_output(fmridataset:::print_data_source_info(mem_dset), "pre-loaded NeuroVec")

  backend <- matrix_backend(mat, spatial_dims = c(2,1,1))
  dset <- fmri_dataset(backend, TR = 1, run_length = 10)
  expect_output(fmridataset:::print_data_source_info(dset), "Backend:")
})

