# Tests for R/dataset_constructors.R error paths - coverage improvement

test_that("fmri_study_dataset errors on empty list", {
  expect_error(fmri_study_dataset(list()), "non-empty list")
})

test_that("fmri_study_dataset errors on non-list", {
  expect_error(fmri_study_dataset("bad"), "non-empty list")
})

test_that("fmri_study_dataset errors on non-fmri_dataset elements", {
  expect_error(
    fmri_study_dataset(list(42)),
    "must inherit from 'fmri_dataset'"
  )
})

test_that("fmri_study_dataset errors on mismatched subject_ids", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  expect_error(
    fmri_study_dataset(list(ds1), subject_ids = c("a", "b")),
    "must match length"
  )
})

test_that("fmri_study_dataset errors on different TRs", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 1, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  expect_error(
    fmri_study_dataset(list(ds1, ds2)),
    "equal TR"
  )
})

test_that("fmri_study_dataset auto-generates subject_ids", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(list(ds1))
  expect_s3_class(sds, "fmri_study_dataset")
  expect_equal(sds$subject_ids, 1)
})

test_that("fmri_study_dataset with multiple subjects works", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(list(ds1, ds2), subject_ids = c("s1", "s2"))
  expect_s3_class(sds, "fmri_study_dataset")
  expect_s3_class(sds, "fmri_dataset")
  expect_equal(sds$subject_ids, c("s1", "s2"))
})

test_that("fmri_study_dataset n_runs works", {
  mat1 <- matrix(rnorm(80), nrow = 20, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = c(10, 10))

  sds <- fmri_study_dataset(list(ds1), subject_ids = c("s1"))
  expect_equal(n_runs(sds), 2)
})

# --- fmri_dataset constructor (generic) ---

test_that("fmri_dataset with matrix_backend works", {
  mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  backend <- matrix_backend(mat, mask = mask)

  ds <- fmri_dataset(backend, TR = 2, run_length = 10)
  expect_s3_class(ds, "fmri_dataset")
})

test_that("fmri_dataset with study_backend works", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)
  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  ds <- fmri_dataset(sb, TR = 2, run_length = c(10, 10))
  expect_s3_class(ds, "fmri_dataset")
})

# --- backend_registry coverage ---

test_that("validate_registered_backend errors on invalid input", {
  expect_error(
    validate_registered_backend("nonexistent_type_xyz"),
    "Invalid backend|must inherit|not registered"
  )
})
