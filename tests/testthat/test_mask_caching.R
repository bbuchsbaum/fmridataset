test_that("nifti_backend caches mask only after validation", {
  # Minimal mock NeuroVec
  mock_vec <- structure(
    array(1:2, c(1, 1, 1, 2)),
    class = c("DenseNeuroVec", "NeuroVec", "array"),
    space = structure(list(dim = c(1, 1, 1), origin = c(0, 0, 0), spacing = c(1, 1, 1)),
                      class = "NeuroSpace")
  )

  invalid_mask <- structure(
    array(c(TRUE, NA), c(1, 1, 2)),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = c(1, 1, 2)
  )

  backend <- nifti_backend(source = list(mock_vec), mask_source = invalid_mask)

  expect_error(backend_get_mask(backend), "Mask contains NA values")
  expect_null(backend$mask_vec)
  expect_null(backend$mask)

  valid_mask <- structure(
    array(TRUE, c(1, 1, 2)),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = c(1, 1, 2)
  )
  backend$mask_source <- valid_mask
  mask <- backend_get_mask(backend)
  expect_true(all(mask))
  expect_identical(mask, backend$mask_vec)
})


test_that("h5_backend caches mask only after validation", {
  invalid_mask <- structure(
    array(c(TRUE, NA), c(2, 1, 1)),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = c(2, 1, 1)
  )

  backend <- new.env(parent = emptyenv())
  backend$mask_source <- invalid_mask
  backend$mask <- NULL
  backend$mask_vec <- NULL
  backend$source <- list()
  backend$mask_dataset <- "data/elements"
  backend$data_dataset <- "data"
  class(backend) <- c("h5_backend", "storage_backend")

  expect_error(backend_get_mask(backend), "H5 mask contains NA values")
  expect_null(backend$mask_vec)
  expect_null(backend$mask)

  valid_mask <- structure(
    array(TRUE, c(1, 1, 1)),
    class = c("LogicalNeuroVol", "NeuroVol", "array"),
    dim = c(1, 1, 1)
  )
  backend$mask_source <- valid_mask
  mask <- backend_get_mask(backend)
  expect_true(all(mask))
  expect_identical(mask, backend$mask_vec)
})
