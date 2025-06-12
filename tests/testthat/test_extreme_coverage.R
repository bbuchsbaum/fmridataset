library(testthat)
library(fmridataset)

# Test print.latent_dataset output
if (!methods::isClass("MockLatentNeuroVec")) {
  setClass("MockLatentNeuroVec", slots = c(basis = "matrix", loadings = "matrix", mask = "logical"))
  setMethod("dim", "MockLatentNeuroVec", function(x) c(2,2,2,nrow(x@basis)))
}
create_mock_lvec <- function(n_time = 4, n_vox = 8, k = 2) {
  basis <- matrix(seq_len(n_time * k), nrow = n_time, ncol = k)
  loadings <- matrix(seq_len(n_vox * k), nrow = n_vox, ncol = k)
  mask <- rep(TRUE, n_vox)
  new("MockLatentNeuroVec", basis = basis, loadings = loadings, mask = mask)
}

test_that("print.latent_dataset summarises object", {
  lvec <- create_mock_lvec()
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      dset <- latent_dataset(lvec, TR = 1, run_length = dim(lvec)[4])
      expect_output(print(dset), "Latent Dataset")
      expect_output(print(dset), "Latent Data Summary")
    }
  )
})

# Test memoisation of get_data_from_file

test_that("get_data_from_file memoises loaded data", {
  scan_file <- "scan.nii"; mask_file <- "mask.nii"
  call_count <- 0
  with_mocked_bindings(
    file.exists = function(x) TRUE,
    read_vol = function(x) array(TRUE, c(1,1,1)),
    read_vec = function(x, ...) { call_count <<- call_count + 1; matrix(1:4, nrow = 2) },
    .package = c("base", "neuroim2"),
    {
      dset <- fmri_dataset_legacy(scans = scan_file, mask = mask_file, TR = 1, run_length = 2, preload = FALSE)
      r1 <- fmridataset:::get_data_from_file(dset)
      r2 <- fmridataset:::get_data_from_file(dset)
      expect_identical(r1, r2)
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

