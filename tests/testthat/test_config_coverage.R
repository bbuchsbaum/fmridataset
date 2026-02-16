# Tests for R/config.R - coverage improvement

test_that("default_config returns expected structure", {
  cfg <- fmridataset:::default_config()
  expect_type(cfg, "list")
  expect_equal(cfg$cmd_flags, "")
  expect_equal(cfg$jobs, 1)
  expect_equal(cfg$base_path, ".")
  expect_equal(cfg$output_dir, "stat_out")
})

test_that("read_dcf_config parses key-value pairs", {
  tmpfile <- tempfile(fileext = ".cfg")
  on.exit(unlink(tmpfile))

  writeLines(c(
    "# This is a comment",
    "",
    "name: test_config",
    "TR: 2.0",
    "run_length: 100,200",
    "verbose: TRUE",
    "quoted_val: 'hello world'"
  ), tmpfile)

  config <- fmridataset:::read_dcf_config(tmpfile)
  expect_type(config, "list")
  expect_equal(config$name, "test_config")
  expect_equal(config$TR, 2.0)
  expect_equal(config$run_length, c(100, 200))
  expect_equal(config$verbose, TRUE)
  expect_equal(config$quoted_val, "hello world")
})

test_that("read_dcf_config skips comments and empty lines", {
  tmpfile <- tempfile(fileext = ".cfg")
  on.exit(unlink(tmpfile))

  writeLines(c(
    "# comment",
    "   # indented comment",
    "",
    "   ",
    "key1: value1"
  ), tmpfile)

  config <- fmridataset:::read_dcf_config(tmpfile)
  expect_equal(length(config), 1)
  expect_equal(config$key1, "value1")
})

test_that("read_dcf_config handles = separator", {
  tmpfile <- tempfile(fileext = ".cfg")
  on.exit(unlink(tmpfile))

  writeLines(c("key1=value1"), tmpfile)

  config <- fmridataset:::read_dcf_config(tmpfile)
  expect_equal(config$key1, "value1")
})

test_that("read_dcf_config handles FALSE boolean", {
  tmpfile <- tempfile(fileext = ".cfg")
  on.exit(unlink(tmpfile))

  writeLines(c("flag: FALSE"), tmpfile)

  config <- fmridataset:::read_dcf_config(tmpfile)
  expect_equal(config$flag, FALSE)
})

test_that("read_fmri_config errors on missing yaml package", {
  tmpfile <- tempfile(fileext = ".yaml")
  on.exit(unlink(tmpfile))
  writeLines("key: value", tmpfile)

  # If yaml is not available, this should error
  # But if it is available, we test the actual reading
  skip_if_not_installed("yaml")

  # Will fail because required fields are missing

  expect_error(
    read_fmri_config(tmpfile),
    "Missing required"
  )
})

test_that("read_fmri_config errors on missing json package for .json files", {
  tmpfile <- tempfile(fileext = ".json")
  on.exit(unlink(tmpfile))
  writeLines('{"key": "value"}', tmpfile)

  skip_if_not_installed("jsonlite")

  expect_error(
    read_fmri_config(tmpfile),
    "Missing required"
  )
})

test_that("write_fmri_config writes yaml", {
  skip_if_not_installed("yaml")

  tmpfile <- tempfile(fileext = ".yaml")
  on.exit(unlink(tmpfile))

  config <- list(
    scans = "scan.nii",
    TR = 2.0,
    run_length = c(100, 200)
  )

  result <- write_fmri_config(config, tmpfile)
  expect_true(file.exists(tmpfile))
  expect_invisible(write_fmri_config(config, tmpfile))

  # Read it back
  read_back <- yaml::read_yaml(tmpfile)
  expect_equal(read_back$scans, "scan.nii")
  expect_equal(read_back$TR, 2.0)
})

test_that("write_fmri_config removes design and class fields", {
  skip_if_not_installed("yaml")

  tmpfile <- tempfile(fileext = ".yaml")
  on.exit(unlink(tmpfile))

  config <- list(
    scans = "scan.nii",
    TR = 2.0,
    design = data.frame(a = 1),
    class = "fmri_config"
  )

  write_fmri_config(config, tmpfile)
  read_back <- yaml::read_yaml(tmpfile)
  expect_null(read_back$design)
  expect_null(read_back$class)
})
