# Tests for R/study_backend.R, R/study_backend_seed_s3.R, R/study_dataset_access.R

test_that("study_backend constructs from matrix_backends", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  expect_s3_class(sb, "study_backend")
  expect_s3_class(sb, "storage_backend")
  expect_equal(length(sb$backends), 2)
  expect_equal(sb$subject_ids, c("s1", "s2"))
})

test_that("study_backend auto-generates subject_ids", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  sb <- study_backend(list(b1))
  expect_equal(sb$subject_ids, 1)
})

test_that("study_backend errors on empty list", {
  expect_error(study_backend(list()), "non-empty list")
})

test_that("study_backend errors on mismatched subject_ids length", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  expect_error(
    study_backend(list(b1), subject_ids = c("s1", "s2")),
    "must match length"
  )
})

test_that("study_backend errors on inconsistent spatial dims", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(60), nrow = 10, ncol = 6)
  mask1 <- rep(TRUE, 4)
  mask2 <- rep(TRUE, 6)

  b1 <- matrix_backend(mat1, mask = mask1)
  b2 <- matrix_backend(mat2, mask = mask2)

  expect_error(
    study_backend(list(b1, b2)),
    "spatial dimensions must match|masks differ"
  )
})

test_that("backend_get_dims.study_backend returns combined dims", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(60), nrow = 15, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))
  dims <- backend_get_dims(sb)

  expect_equal(dims$time, 25) # 10 + 15
})

test_that("backend_get_mask.study_backend returns mask", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  sb <- study_backend(list(b1))
  result <- backend_get_mask(sb)
  expect_type(result, "logical")
  expect_true(all(result))
})

test_that("backend_get_data.study_backend retrieves specific rows and cols", {
  mat1 <- matrix(1:40, nrow = 10, ncol = 4)
  mat2 <- matrix(41:80, nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  # Get first 5 rows from first subject
  result <- backend_get_data(sb, rows = 1:5, cols = 1:2)
  expect_equal(dim(result), c(5, 2))
  expect_equal(result[1, 1], mat1[1, 1])

  # Get rows spanning both subjects
  result2 <- backend_get_data(sb, rows = c(5, 15), cols = 1:4)
  expect_equal(dim(result2), c(2, 4))
})

test_that("backend_get_metadata.study_backend includes study info", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  sb <- study_backend(list(b1), subject_ids = c("sub-01"))
  meta <- backend_get_metadata(sb)

  expect_equal(meta$storage_format, "study")
  expect_equal(meta$n_subjects, 1)
  expect_equal(meta$subject_ids, "sub-01")
})

test_that("backend_open.study_backend and backend_close.study_backend work", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  sb <- study_backend(list(b1))
  opened <- backend_open(sb)
  expect_s3_class(opened, "study_backend")

  result <- backend_close(sb)
  expect_null(result)
})

test_that(".collect_study_backend_block returns correct data", {
  mat1 <- matrix(1:20, nrow = 5, ncol = 4)
  mat2 <- matrix(21:40, nrow = 5, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)

  result <- fmridataset:::.collect_study_backend_block(
    backends = list(b1, b2),
    rows = as.integer(c(1, 6)),
    cols = 1:4,
    subject_boundaries = c(0L, 5L, 10L),
    n_time = 10L,
    n_vox = 4L
  )

  expect_equal(dim(result), c(2, 4))
  expect_equal(result[1, 1], mat1[1, 1])
  expect_equal(result[2, 1], mat2[1, 1])
})

test_that(".collect_study_backend_block handles empty rows", {
  mat1 <- matrix(1:20, nrow = 5, ncol = 4)
  mask <- rep(TRUE, 4)
  b1 <- matrix_backend(mat1, mask = mask)

  result <- fmridataset:::.collect_study_backend_block(
    backends = list(b1),
    rows = integer(0),
    cols = 1:4,
    subject_boundaries = c(0L, 5L),
    n_time = 5L,
    n_vox = 4L
  )

  expect_equal(dim(result), c(0, 4))
})

# --- study_backend_seed_s3 standalone functions ---

test_that("find_subjects_for_rows identifies correct subjects", {
  boundaries <- c(0L, 10L, 20L, 30L)

  result <- fmridataset:::find_subjects_for_rows(c(5L, 15L), boundaries)
  expect_equal(sort(result), c(1, 2))

  result2 <- fmridataset:::find_subjects_for_rows(c(25L), boundaries)
  expect_equal(result2, 3)

  result3 <- fmridataset:::find_subjects_for_rows(c(1L, 10L, 11L, 30L), boundaries)
  expect_equal(sort(result3), c(1, 2, 3))
})

test_that("study_seed_is_sparse returns FALSE", {
  expect_false(fmridataset:::study_seed_is_sparse(NULL))
})

test_that("create_study_cache creates environment", {
  cache <- fmridataset:::create_study_cache()
  expect_true(is.environment(cache))
})

# --- study_backend coercion from matrix_dataset ---

test_that("study_backend coerces matrix_dataset to backend", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sb <- study_backend(list(ds1), subject_ids = c("s1"))
  expect_s3_class(sb, "study_backend")
  dims <- backend_get_dims(sb)
  expect_equal(dims$time, 10)
})

# --- fmri_study_dataset data access ---

test_that("fmri_study_dataset constructor works", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)

  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  ds <- fmri_study_dataset(
    datasets = list(ds1, ds2),
    subject_ids = c("s1", "s2")
  )

  expect_s3_class(ds, "fmri_study_dataset")
})

test_that("get_data.fmri_study_dataset returns data", {
  mat1 <- matrix(1:40, nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  ds <- fmri_study_dataset(
    datasets = list(ds1),
    subject_ids = c("s1")
  )

  result <- get_data(ds, rows = 1:5, cols = 1:2)
  expect_equal(dim(result), c(5, 2))
})

test_that("get_data_matrix.fmri_study_dataset by subject_id", {
  mat1 <- matrix(1:40, nrow = 10, ncol = 4)
  mat2 <- matrix(41:80, nrow = 10, ncol = 4)

  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)
  ds2 <- matrix_dataset(datamat = mat2, TR = 2, run_length = 10)

  ds <- fmri_study_dataset(
    datasets = list(ds1, ds2),
    subject_ids = c("s1", "s2")
  )

  # Get specific subject by name
  result <- get_data_matrix(ds, subject_id = "s1")
  expect_equal(dim(result), c(10, 4))

  # Get specific subject by index
  result2 <- get_data_matrix(ds, subject_id = 2)
  expect_equal(dim(result2), c(10, 4))

  # Get multiple subjects
  result3 <- get_data_matrix(ds, subject_id = c("s1", "s2"))
  expect_equal(dim(result3), c(20, 4))
})

test_that("get_data_matrix.fmri_study_dataset errors on bad subject_id", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  ds <- fmri_study_dataset(
    datasets = list(ds1), subject_ids = c("s1")
  )

  expect_error(get_data_matrix(ds, subject_id = "nonexistent"), "not found")
  expect_error(get_data_matrix(ds, subject_id = TRUE), "must be character or numeric")
})

test_that("get_mask.fmri_study_dataset returns mask", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  ds <- fmri_study_dataset(
    datasets = list(ds1), subject_ids = c("s1")
  )

  result <- get_mask(ds)
  expect_type(result, "logical")
  expect_true(all(result))
})
