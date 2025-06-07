test_that("latent_dataset errors when fmristore is absent", {
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) FALSE,
    .package = "base",
    {
      expect_error(
        latent_dataset(list(), TR = 2, run_length = 10),
        "fmristore"
      )
    }
  )
})

if (!methods::isClass("MockLatentNeuroVec")) {
  setClass(
    "MockLatentNeuroVec",
    slots = c(basis = "matrix", loadings = "matrix", mask = "logical")
  )
  setMethod(
    "dim",
    "MockLatentNeuroVec",
    function(x) c(2, 2, 2, nrow(x@basis))
  )
}

create_mock_lvec <- function(n_time = 5, n_vox = 10, k = 4) {
  basis <- matrix(seq_len(n_time * k), nrow = n_time, ncol = k)
  loadings <- matrix(seq_len(n_vox * k), nrow = n_vox, ncol = k)
  mask <- rep(TRUE, n_vox)
  new("MockLatentNeuroVec", basis = basis, loadings = loadings, mask = mask)
}

test_that("latent_dataset constructs dataset from minimal latent object", {
  lvec <- create_mock_lvec()
  rl <- dim(lvec)[4]
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      dset <- latent_dataset(lvec, TR = 1, run_length = rl)
      expect_s3_class(dset, "latent_dataset")
      expect_identical(get_data(dset), lvec@basis)
      expect_identical(get_mask(dset), lvec@mask)
      expect_equal(blocklens(dset), rl)
    }
  )
})

test_that("latent_dataset validates run_length sum", {
  lvec <- create_mock_lvec()
  with_mocked_bindings(
    requireNamespace = function(pkg, quietly = TRUE) TRUE,
    .package = "base",
    {
      expect_error(
        latent_dataset(lvec, TR = 1, run_length = dim(lvec)[4] - 1),
        "Sum of run lengths"
      )
    }
  )
})
