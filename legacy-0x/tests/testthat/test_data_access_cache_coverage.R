# Tests for cache functions in R/data_access.R - coverage improvement

test_that(".get_cache_size returns default or option", {
  # Default
  old <- options(fmridataset.cache_max_mb = NULL)
  on.exit(options(old))
  result <- fmridataset:::.get_cache_size()
  expect_equal(result, 512 * 1024^2)

  # Custom
  options(fmridataset.cache_max_mb = 256)
  result <- fmridataset:::.get_cache_size()
  expect_equal(result, 256 * 1024^2)
})

test_that(".get_cache_evict returns default or option", {
  old <- options(fmridataset.cache_evict = NULL)
  on.exit(options(old))
  result <- fmridataset:::.get_cache_evict()
  expect_equal(result, "lru")

  options(fmridataset.cache_evict = "fifo")
  result <- fmridataset:::.get_cache_evict()
  expect_equal(result, "fifo")
})

test_that(".get_cache_logging returns default or option", {
  old <- options(fmridataset.cache_logging = NULL)
  on.exit(options(old))
  result <- fmridataset:::.get_cache_logging()
  expect_false(result)

  options(fmridataset.cache_logging = TRUE)
  result <- fmridataset:::.get_cache_logging()
  expect_true(result)
})

test_that(".create_data_cache creates a cache object", {
  cache <- fmridataset:::.create_data_cache()
  expect_true(!is.null(cache))
  expect_true("reset" %in% names(cache))
  expect_true("keys" %in% names(cache))
})

test_that("fmri_clear_cache works", {
  result <- fmri_clear_cache()
  expect_null(result)
})

test_that("fmri_cache_info returns proper structure", {
  info <- fmri_cache_info()
  expect_type(info, "list")
  expect_true("max_size" %in% names(info))
  expect_true("current_size" %in% names(info))
  expect_true("n_objects" %in% names(info))
  expect_true("eviction_policy" %in% names(info))
})

test_that("fmri_cache_resize errors on invalid input", {
  expect_error(fmri_cache_resize("abc"), "positive number")
  expect_error(fmri_cache_resize(-1), "positive number")
  expect_error(fmri_cache_resize(c(1, 2)), "positive number")
})

test_that("fmri_cache_resize warns with valid input", {
  expect_warning(fmri_cache_resize(1024), "not supported")
})

# --- get_data_matrix.matrix_dataset subsetting ---

test_that("get_data_matrix.matrix_dataset with row and col subsetting", {
  mat <- matrix(1:40, nrow = 10, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 10)

  # With rows
  result <- get_data_matrix(ds, rows = 1:5)
  expect_equal(dim(result), c(5, 4))

  # With cols
  result <- get_data_matrix(ds, cols = 1:2)
  expect_equal(dim(result), c(10, 2))

  # With both
  result <- get_data_matrix(ds, rows = 1:3, cols = 2:3)
  expect_equal(dim(result), c(3, 2))
  expect_equal(result[1, 1], mat[1, 2])
})

test_that("get_data_matrix.matrix_dataset without subsetting", {
  mat <- matrix(1:40, nrow = 10, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 10)

  result <- get_data_matrix(ds)
  expect_equal(dim(result), c(10, 4))
  expect_equal(result, mat)
})

test_that("get_data.matrix_dataset returns datamat", {
  mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 10)

  result <- get_data(ds)
  expect_equal(result, mat)
})

test_that("get_mask.matrix_dataset returns mask", {
  mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = 10)

  result <- get_mask(ds)
  expect_type(result, "logical")
  expect_length(result, 4)
  expect_true(all(result))
})

test_that("get_mask.fmri_study_dataset returns mask", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(datasets = list(ds1), subject_ids = c("s1"))
  result <- get_mask(sds)
  expect_type(result, "logical")
  expect_true(all(result))
})
