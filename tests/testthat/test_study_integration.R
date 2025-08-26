test_that("full workflow and dplyr pipeline", {
  b1 <- matrix_backend(matrix(1:20, nrow = 10, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(21:40, nrow = 10, ncol = 2), spatial_dims = c(2, 1, 1))
  d1 <- fmri_dataset(b1, TR = 2, run_length = 10)
  d2 <- fmri_dataset(b2, TR = 2, run_length = 10)
  study <- fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2"))
  tbl <- as_tibble(study, materialise = TRUE)
  expect_s3_class(tbl, "tbl_df")
  expect_equal(nrow(tbl), 20)
  expect_equal(tbl$subject_id[1], "s1")
  expect_equal(tbl$subject_id[11], "s2")

  filtered <- dplyr::filter(tbl, subject_id == "s1")
  expect_equal(nrow(filtered), 10)
})

test_that("memory usage is low for lazy as_tibble", {
  skip("Memory benchmarking can be unreliable in test environments")
  skip_if_not_installed("bench")

  n_subj <- 5
  n_time <- 50
  n_vox <- 200

  datasets <- lapply(1:n_subj, function(i) {
    mat <- matrix(rnorm(n_time * n_vox), nrow = n_time, ncol = n_vox)
    b <- matrix_backend(mat, spatial_dims = c(n_vox, 1, 1))
    fmri_dataset(b, TR = 2, run_length = n_time)
  })

  study <- fmri_study_dataset(datasets, subject_ids = sprintf("s%02d", seq_len(n_subj)))
  full_size <- object.size(matrix(0, n_time * n_subj, n_vox))

  bench_res <- bench::mark(as_tibble(study), iterations = 1, check = FALSE)
  allocated <- bench_res$mem_alloc[[1]]
  expect_true(allocated < 0.5 * full_size)
})

test_that("large dataset of 100+ subjects works", {
  n_subj <- 101
  datasets <- lapply(seq_len(n_subj), function(i) {
    mat <- matrix(i, nrow = 5, ncol = 2)
    b <- matrix_backend(mat, spatial_dims = c(2, 1, 1))
    fmri_dataset(b, TR = 1, run_length = 5)
  })
  study <- fmri_study_dataset(datasets, subject_ids = sprintf("sub-%03d", seq_len(n_subj)))
  dm <- as_tibble(study)
  expect_equal(dim(dm), c(5 * n_subj, 2))
  md <- attr(dm, "rowData")
  expect_equal(length(unique(md$subject_id)), n_subj)
  expect_equal(md$subject_id[1], "sub-001")
  expect_equal(md$subject_id[5 * n_subj], sprintf("sub-%03d", n_subj))
})
