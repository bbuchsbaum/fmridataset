# Tests for R/data_access.R - coverage improvement

test_that("fmri_clear_cache works", {
  result <- fmri_clear_cache()
  expect_null(result)
})

test_that("fmri_cache_info returns valid structure", {
  # Clear cache first
  fmri_clear_cache()

  info <- fmri_cache_info()
  expect_type(info, "list")
  expect_true("max_size" %in% names(info))
  expect_true("current_size" %in% names(info))
  expect_true("n_objects" %in% names(info))
  expect_true("eviction_policy" %in% names(info))
  expect_equal(info$n_objects, 0)
})

test_that("fmri_cache_resize warns about restart", {
  expect_warning(fmri_cache_resize(1024), "resizing is not supported")
  expect_warning(fmri_cache_resize(256), "cache_max_mb")
})

test_that("fmri_cache_resize validates input", {
  expect_error(fmri_cache_resize("bad"))
  expect_error(fmri_cache_resize(-1))
  expect_error(fmri_cache_resize(c(1, 2)))
})

test_that(".get_cache_size returns default or option", {
  old <- getOption("fmridataset.cache_max_mb")
  on.exit(options(fmridataset.cache_max_mb = old))

  options(fmridataset.cache_max_mb = NULL)
  expect_equal(fmridataset:::.get_cache_size(), 512 * 1024^2)

  options(fmridataset.cache_max_mb = 256)
  expect_equal(fmridataset:::.get_cache_size(), 256 * 1024^2)
})

test_that(".get_cache_evict returns default or option", {
  old <- getOption("fmridataset.cache_evict")
  on.exit(options(fmridataset.cache_evict = old))

  options(fmridataset.cache_evict = NULL)
  expect_equal(fmridataset:::.get_cache_evict(), "lru")
})

test_that(".get_cache_logging returns default or option", {
  old <- getOption("fmridataset.cache_logging")
  on.exit(options(fmridataset.cache_logging = old))

  options(fmridataset.cache_logging = NULL)
  expect_false(fmridataset:::.get_cache_logging())
})

test_that(".create_data_cache creates valid cache", {
  cache <- fmridataset:::.create_data_cache()
  expect_true(inherits(cache, "cache_mem"))
})

test_that("get_data.matrix_dataset returns data matrix", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  result <- get_data(ds)
  expect_equal(result, mat)
})

test_that("get_data_matrix.matrix_dataset returns data matrix", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  result <- get_data_matrix(ds)
  expect_equal(result, mat)
})

test_that("get_data_matrix.matrix_dataset supports subsetting", {
  mat <- matrix(1:200, nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  # Subset rows
  result <- get_data_matrix(ds, rows = 1:5)
  expect_equal(nrow(result), 5)
  expect_equal(ncol(result), 10)

  # Subset cols
  result <- get_data_matrix(ds, cols = 1:3)
  expect_equal(nrow(result), 20)
  expect_equal(ncol(result), 3)

  # Subset both
  result <- get_data_matrix(ds, rows = 1:5, cols = 1:3)
  expect_equal(dim(result), c(5, 3))
})

test_that("get_mask.matrix_dataset returns mask", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))
  result <- get_mask(ds)
  expect_type(result, "logical")
  expect_length(result, 10)
  expect_true(all(result))
})
