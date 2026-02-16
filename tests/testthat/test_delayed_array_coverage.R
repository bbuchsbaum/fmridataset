# Tests for R/as_delayed_array.R and R/study_backend_seed_s3.R - coverage improvement

test_that(".ensure_delayed_array works when DelayedArray available", {
  skip_if_not_installed("DelayedArray")
  expect_no_error(fmridataset:::.ensure_delayed_array())
})

test_that(".ensure_delayed_array errors when disabled", {
  skip_if_not_installed("DelayedArray")
  old <- options(fmridataset.disable_delayedarray = TRUE)
  on.exit(options(old))

  expect_error(
    fmridataset:::.ensure_delayed_array(),
    "disabled"
  )
})

test_that("register_delayed_array_support is idempotent when already registered", {
  skip_if_not_installed("DelayedArray")

  # After as_delayed_array has been called, re-calling should be a no-op
  ns <- getNamespace("fmridataset")
  env <- get(".delayed_array_support_env", envir = ns)
  env$registered <- TRUE
  expect_no_error(ns$register_delayed_array_support())
})

test_that("as_delayed_array.matrix_backend creates DelayedArray", {
  skip_if_not_installed("DelayedArray")

  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)

  da <- as_delayed_array(backend)
  expect_true(inherits(da, "DelayedMatrix") || inherits(da, "DelayedArray"))
  expect_equal(dim(da), c(20, 10))
})

test_that("as_delayed_array.study_backend creates DelayedArray", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  da <- as_delayed_array(sb)
  expect_true(inherits(da, "DelayedMatrix") || inherits(da, "DelayedArray"))
  expect_equal(dim(da), c(20, 4))
})

test_that("as_delayed_array.default errors", {
  expect_error(as_delayed_array(42), "No as_delayed_array method")
  expect_error(as_delayed_array("string"), "No as_delayed_array method")
})

# --- study_backend_seed_s3.R ---

test_that("register_study_backend_seed_methods is idempotent when already registered", {
  skip_if_not_installed("DelayedArray")

  ns <- getNamespace("fmridataset")
  env <- get(".study_backend_seed_env", envir = ns)
  env$registered <- TRUE
  expect_no_error(ns$register_study_backend_seed_methods())
})

test_that("study_backend_seed constructor works", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  seed <- fmridataset:::study_backend_seed(
    backends = list(b1, b2),
    subject_ids = c("s1", "s2")
  )

  expect_s4_class(seed, "StudyBackendSeed")
  expect_equal(seed@dims, c(20L, 4L))
  expect_equal(seed@subject_ids, c("s1", "s2"))
})

test_that("study_backend_seed errors on mismatched lengths", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  expect_error(
    fmridataset:::study_backend_seed(list(b1), c("s1", "s2")),
    "must match"
  )
})

test_that("study_backend_seed errors on non-list", {
  skip_if_not_installed("DelayedArray")

  expect_error(
    fmridataset:::study_backend_seed(42, "s1"),
    "must be a list"
  )
})

test_that("study_seed_chunk_grid works", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  seed <- fmridataset:::study_backend_seed(
    backends = list(b1, b2),
    subject_ids = c("s1", "s2")
  )

  grid <- fmridataset:::study_seed_chunk_grid(seed)
  expect_true(!is.null(grid))
})

test_that("study_seed_chunk_grid with custom chunk_dim", {
  skip_if_not_installed("DelayedArray")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  seed <- fmridataset:::study_backend_seed(
    backends = list(b1),
    subject_ids = c("s1")
  )

  grid <- fmridataset:::study_seed_chunk_grid(seed, chunk_dim = c(5L, 2L))
  expect_true(!is.null(grid))
})

test_that("study_seed_is_sparse returns FALSE", {
  expect_false(fmridataset:::study_seed_is_sparse(NULL))
  expect_false(fmridataset:::study_seed_is_sparse("anything"))
})

test_that("create_study_cache creates environment", {
  cache <- fmridataset:::create_study_cache()
  expect_true(is.environment(cache))
})

test_that("find_subjects_for_rows works correctly", {
  boundaries <- c(0L, 10L, 20L, 30L)

  result <- fmridataset:::find_subjects_for_rows(c(1L, 5L), boundaries)
  expect_equal(result, 1)

  result <- fmridataset:::find_subjects_for_rows(c(11L, 15L), boundaries)
  expect_equal(result, 2)

  result <- fmridataset:::find_subjects_for_rows(c(5L, 15L, 25L), boundaries)
  expect_equal(sort(result), c(1, 2, 3))

  # Edge case: exact boundaries
  result <- fmridataset:::find_subjects_for_rows(c(10L, 20L), boundaries)
  expect_equal(sort(result), c(1, 2))
})
