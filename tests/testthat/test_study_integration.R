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

test_that("lazy as_tibble avoids materialising study data", {
  skip_on_cran()
  skip_if_not_installed("withr")

  ns <- asNamespace("fmridataset")
  registered <- character()
  log_env <- new.env(parent = emptyenv())
  log_env$count <- 0L

  register_method <- function(name, fn) {
    base::registerS3method(name, "tracked_matrix_backend", fn, envir = ns)
    registered <<- c(registered, paste0(name, ".tracked_matrix_backend"))
  }

  withr::defer({
    table <- get(".__S3MethodsTable__.", envir = ns)
    rm(list = registered, envir = table)
  })

  passthrough <- function(name) get(paste0(name, ".matrix_backend"), envir = ns)
  register_method("backend_open", passthrough("backend_open"))
  register_method("backend_close", passthrough("backend_close"))
  register_method("backend_get_dims", passthrough("backend_get_dims"))
  register_method("backend_get_mask", passthrough("backend_get_mask"))
  register_method("backend_get_metadata", passthrough("backend_get_metadata"))

  orig_get_data <- get("backend_get_data.matrix_backend", envir = ns)
  register_method("backend_get_data", function(backend, rows = NULL, cols = NULL, ...) {
    log_env$count <- log_env$count + 1L
    orig_get_data(backend, rows = rows, cols = cols, ...)
  })

  create_dataset <- function(id) {
    mat <- matrix(id, nrow = 20, ncol = 5)
    backend <- matrix_backend(mat, spatial_dims = c(5, 1, 1))
    class(backend) <- c("tracked_matrix_backend", class(backend))
    fmri_dataset(backend, TR = 2, run_length = 20)
  }

  datasets <- lapply(1:3, create_dataset)
  study <- fmri_study_dataset(datasets, subject_ids = sprintf("s%02d", seq_len(3)))

  log_env$count <- 0L
  lazy_tbl <- as_tibble(study)
  expect_s4_class(lazy_tbl, "DelayedMatrix")
  expect_equal(log_env$count, 0L)

  log_env$count <- 0L
  mat_tbl <- as_tibble(study, materialise = TRUE)
  expect_s3_class(mat_tbl, "tbl_df")
  expect_equal(log_env$count, length(datasets))
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
