library(testthat)
library(fmridataset)

# Test print.latent_dataset output
if (!methods::isClass("MockLatentNeuroVec")) {
  setClass("MockLatentNeuroVec", slots = c(basis = "matrix", loadings = "matrix", 
                                           space = "ANY", offset = "numeric"))
  setMethod("dim", "MockLatentNeuroVec", function(x) c(2,2,2,nrow(x@basis)))
}
create_mock_lvec <- function(n_time = 4, n_vox = 8, k = 2) {
  basis <- matrix(seq_len(n_time * k), nrow = n_time, ncol = k)
  loadings <- matrix(seq_len(n_vox * k), nrow = n_vox, ncol = k)
  space <- c(2, 2, 2, n_time)
  new("MockLatentNeuroVec", basis = basis, loadings = loadings, 
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
  skip_if_not_installed("neuroim2")
  skip_if_not_installed("withr")

  ns_fmri <- asNamespace("fmridataset")
  ns_neuro <- asNamespace("neuroim2")

  masked_len <- 4
  test_mask <- rep(TRUE, masked_len)

  register_get_mask <- function(fn) {
    base::registerS3method("get_mask", "mock_cache_dataset", fn, envir = ns_fmri)
  }

  register_get_mask(function(x, ...) test_mask)
  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns_fmri)
    rm(list = "get_mask.mock_cache_dataset", envir = table)
  })

  original_read_vec <- get("read_vec", envir = ns_neuro)
  call_count <- 0

  unlockBinding("read_vec", ns_neuro)
  assign("read_vec", function(scans, mask, mode, ...) {
    call_count <<- call_count + 1
    matrix(seq_len(length(mask) * 2), nrow = 2)
  }, envir = ns_neuro)
  lockBinding("read_vec", ns_neuro)
  withr::defer({
    unlockBinding("read_vec", ns_neuro)
    assign("read_vec", original_read_vec, envir = ns_neuro)
    lockBinding("read_vec", ns_neuro)
  })

  fmri_clear_cache()

  dset <- structure(
    list(scans = "dummy_scan", mode = "normal"),
    class = "mock_cache_dataset"
  )

  result1 <- get_data_from_file(dset)
  result2 <- get_data_from_file(dset)

  expect_identical(result1, result2)
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
