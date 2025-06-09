context("fmri_study_dataset")

test_that("constructor combines datasets", {
  b1 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), spatial_dims = c(2,1,1))
  b2 <- matrix_backend(matrix(21:40, nrow = 10, ncol = 2), spatial_dims = c(2,1,1))

  d1 <- fmri_dataset(b1, TR = 2, run_length = 10,
                     event_table = data.frame(onset = 1, run_id = 1))
  d2 <- fmri_dataset(b2, TR = 2, run_length = 10,
                     event_table = data.frame(onset = 2, run_id = 1))

  study <- fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))

  expect_s3_class(study, "fmri_study_dataset")
  expect_equal(backend_get_dims(study$backend)$time, 20)
  expect_true(all(c("subject_id", "run_id") %in% names(study$event_table)))
  expect_equal(nrow(study$event_table), 2)
})

test_that("with_rowData attaches attribute", {
  mat <- DelayedArray::DelayedArray(matrix(1:4, nrow = 2))
  rd <- data.frame(id = 1:2)
  out <- with_rowData(mat, rd)
  expect_equal(attr(out, "rowData"), rd)
})

