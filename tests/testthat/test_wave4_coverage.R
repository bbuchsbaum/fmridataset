# Wave 4 coverage tests: as_delarr, group_stream, backend_registry, errors, config, storage_backend

# --- as_delarr ---

test_that("as_delarr.matrix_backend works", {
  skip_if_not_installed("delarr")

  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  mask <- rep(TRUE, 10)
  backend <- matrix_backend(mat, mask = mask)

  da <- as_delarr(backend)
  expect_true(!is.null(da))
  expect_equal(nrow(da), 20)
  expect_equal(ncol(da), 10)
})

test_that("as_delarr.study_backend works", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mat2 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)
  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  da <- as_delarr(sb)
  expect_true(!is.null(da))
  expect_equal(nrow(da), 20)
  expect_equal(ncol(da), 4)
})

test_that("as_delarr.study_backend pull function retrieves data", {
  skip_if_not_installed("delarr")

  mat1 <- matrix(1:40, nrow = 10, ncol = 4)
  mat2 <- matrix(41:80, nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)

  b1 <- matrix_backend(mat1, mask = mask)
  b2 <- matrix_backend(mat2, mask = mask)
  sb <- study_backend(list(b1, b2), subject_ids = c("s1", "s2"))

  da <- as_delarr(sb)
  # Materialize subset
  result <- as.matrix(da[1:5, 1:2])
  expect_equal(dim(result), c(5, 2))
})

test_that("as_delarr.default errors", {
  expect_error(as_delarr(42), "No as_delarr method")
})

# --- group_stream ---

test_that("stream_subjects works like iter_subjects", {
  datasets <- lapply(1:3, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2", "s3"),
    age = c(25, 30, 35),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  iter <- stream_subjects(gd)
  r1 <- iter[["next"]]()
  expect_equal(r1$subject_id, "s1")
})

test_that("stream_subjects warns on prefetch > 1", {
  datasets <- lapply(1:2, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2"),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  expect_warning(stream_subjects(gd, prefetch = 5L), "not yet implemented")
})

test_that("group_reduce accumulates values", {
  datasets <- lapply(1:3, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2", "s3"),
    age = c(25, 30, 35),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  result <- group_reduce(
    gd,
    .map = function(row) row$age,
    .reduce = function(acc, val) acc + val,
    .init = 0
  )

  expect_equal(result, 90) # 25 + 30 + 35
})

test_that("group_reduce with on_error='skip' skips errors", {
  datasets <- lapply(1:3, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2", "s3"),
    age = c(25, 30, 35),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  result <- group_reduce(
    gd,
    .map = function(row) {
      if (row$age == 30) stop("skip me")
      row$age
    },
    .reduce = function(acc, val) acc + val,
    .init = 0,
    on_error = "skip"
  )

  expect_equal(result, 60) # 25 + 35, skipped 30
})

test_that("group_reduce with on_error='warn' warns", {
  datasets <- lapply(1:3, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2", "s3"),
    age = c(25, 30, 35),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  expect_warning(
    result <- group_reduce(
      gd,
      .map = function(row) {
        if (row$age == 30) stop("test error")
        row$age
      },
      .reduce = function(acc, val) acc + val,
      .init = 0,
      on_error = "warn"
    ),
    "test error"
  )
  expect_equal(result, 60)
})

test_that("group_reduce with on_error='stop' stops", {
  datasets <- lapply(1:3, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })
  subjects_df <- data.frame(
    subject_id = c("s1", "s2", "s3"),
    age = c(25, 30, 35),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )
  gd <- fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")

  expect_error(
    group_reduce(
      gd,
      .map = function(row) {
        if (row$age == 25) stop("test error")
        row$age
      },
      .reduce = function(acc, val) acc + val,
      .init = 0,
      on_error = "stop"
    ),
    "test error"
  )
})

# --- backend_registry ---

test_that("unregister_backend removes a registered backend", {
  # Register a temp backend
  register_backend("test_temp_backend", factory = function(...) {
    structure(list(), class = c("test_temp_backend", "storage_backend"))
  }, overwrite = TRUE)

  expect_true(is_backend_registered("test_temp_backend"))
  result <- unregister_backend("test_temp_backend")
  expect_true(result)
  expect_false(is_backend_registered("test_temp_backend"))
})

test_that("unregister_backend returns FALSE for non-existent backend", {
  result <- unregister_backend("nonexistent_backend_xyz")
  expect_false(result)
})

test_that("unregister_backend errors on non-string input", {
  expect_error(unregister_backend(42), "character string")
})

test_that("validate_registered_backend works on valid backends", {
  # Use a real matrix_backend which passes validation
  mat <- matrix(rnorm(20), nrow = 5, ncol = 4)
  mask <- rep(TRUE, 4)
  backend <- matrix_backend(mat, mask = mask)

  expect_true(validate_registered_backend(backend))
})

test_that("get_backend_registry returns all backends", {
  all_backends <- get_backend_registry()
  expect_type(all_backends, "list")
  expect_true(length(all_backends) > 0)
  expect_true("matrix" %in% names(all_backends))
})

test_that("get_backend_registry errors on non-existent backend", {
  expect_error(get_backend_registry("nonexistent_xyz"), "not registered")
})

test_that("register_backend errors on invalid inputs", {
  expect_error(register_backend(42, function() {}), "non-empty character string")
  expect_error(register_backend("", function() {}), "non-empty character string")
  expect_error(register_backend("test", "not_a_function"), "must be a function")
  expect_error(register_backend("test", function() {}, description = 42), "character string or NULL")
  expect_error(register_backend("test", function() {}, validate_function = 42), "must be a function or NULL")
})

test_that("register_backend errors on duplicate without overwrite", {
  register_backend("test_dup", factory = function(...) NULL, overwrite = TRUE)
  on.exit(unregister_backend("test_dup"))
  expect_error(register_backend("test_dup", factory = function(...) NULL), "already registered")
})

test_that("create_backend errors on non-existent backend", {
  expect_error(create_backend("nonexistent_xyz"), "not registered")
})

test_that("list_backend_names returns names", {
  names <- list_backend_names()
  expect_type(names, "character")
  expect_true("matrix" %in% names)
})

# --- errors.R ---

test_that("fmridataset_error creates condition", {
  err <- fmridataset_error("test message")
  expect_s3_class(err, "fmridataset_error")
  expect_s3_class(err, "error")
  expect_s3_class(err, "condition")
  expect_equal(err$message, "test message")
})

test_that("fmridataset_error_backend_io includes file and operation", {
  err <- fmridataset_error_backend_io("IO fail", file = "/tmp/test.nii", operation = "read")
  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_equal(err$file, "/tmp/test.nii")
  expect_equal(err$operation, "read")
})

test_that("fmridataset_error_config includes parameter and value", {
  err <- fmridataset_error_config("bad param", parameter = "TR", value = -1)
  expect_s3_class(err, "fmridataset_error_config")
  expect_equal(err$parameter, "TR")
  expect_equal(err$value, -1)
})

test_that("stop_fmridataset with explicit message", {
  expect_error(
    stop_fmridataset(fmridataset_error_config, "test error msg"),
    "test error msg"
  )
})

test_that("stop_fmridataset with extra data", {
  expect_error(
    stop_fmridataset(fmridataset_error_config, "bad param", parameter = "TR"),
    "bad param"
  )
})

# --- config.R ---

test_that("read_fmri_config errors on missing required fields", {
  tmp <- tempfile(fileext = ".dcf")
  writeLines(c(
    "TR: 2",
    "run_length: 100,100"
  ), tmp)
  on.exit(unlink(tmp))

  expect_error(read_fmri_config(tmp), "Missing required")
})

test_that("read_fmri_config errors on missing file", {
  expect_error(read_fmri_config("/nonexistent/config.dcf"), "cannot open|not found|No such")
})

test_that("write_fmri_config writes DCF file", {
  tmp <- tempfile(fileext = ".dcf")
  on.exit(unlink(tmp))

  config <- list(TR = 2, run_length = "100,100")
  write_fmri_config(config, tmp)

  expect_true(file.exists(tmp))
  content <- readLines(tmp)
  expect_true(any(grepl("TR", content)))
})

# --- storage_backend validate_backend ---

test_that("validate_backend errors on non-storage_backend", {
  expect_error(validate_backend(42), "must inherit from 'storage_backend'")
  expect_error(validate_backend(list()), "must inherit from 'storage_backend'")
})

test_that("validate_backend passes for valid backends", {
  mat <- matrix(rnorm(40), nrow = 10, ncol = 4)
  mask <- rep(TRUE, 4)
  backend <- matrix_backend(mat, mask = mask)

  expect_true(validate_backend(backend))
})

# --- print_methods ---

test_that("print.fmri_dataset shows dataset info", {
  mat <- matrix(rnorm(200), nrow = 20, ncol = 10)
  ds <- matrix_dataset(datamat = mat, TR = 2, run_length = c(10, 10))

  out <- capture.output(print(ds))
  expect_true(length(out) > 0)
})

test_that("print.fmri_study_dataset shows study info", {
  mat1 <- matrix(rnorm(40), nrow = 10, ncol = 4)
  ds1 <- matrix_dataset(datamat = mat1, TR = 2, run_length = 10)

  sds <- fmri_study_dataset(list(ds1), subject_ids = c("s1"))
  out <- capture.output(print(sds))
  expect_true(length(out) > 0)
})
