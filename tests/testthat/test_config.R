context("config")

test_that("default_config returns expected fields", {
  cfg <- fmridataset:::default_config()
  expect_true(is.environment(cfg))
  expect_equal(cfg$cmd_flags, "")
  expect_equal(cfg$jobs, 1)
})


test_that("read_fmri_config parses configuration file", {
  tmpdir <- tempdir()
  cfg_file <- file.path(tmpdir, "cfg.R")
  event_file <- file.path(tmpdir, "events.txt")

  writeLines(c("block", "1", "2"), event_file)

  conf_lines <- c(
    "scans <- c('scan1.nii', 'scan2.nii')",
    "TR <- 2",
    "mask <- 'mask.nii'",
    "run_length <- c(1, 1)",
    "event_model <- 'model'",
    "event_table <- 'events.txt'",
    "block_column <- 'block'",
    "baseline_model <- 'baseline'"
  )

  writeLines(conf_lines, cfg_file)

  cfg <- read_fmri_config(cfg_file, base_path = tmpdir)

  expect_s3_class(cfg, "fmri_config")
  expect_equal(cfg$scans, c("scan1.nii", "scan2.nii"))
  expect_equal(cfg$TR, 2)
  expect_equal(cfg$base_path, tmpdir)
  expect_equal(names(cfg$design), "block")
  expect_equal(cfg$output_dir, "stat_out")
})


test_that("latent_dataset errors when fmristore missing", {
  skip_if(requireNamespace("fmristore", quietly = TRUE))
  dummy_vec <- structure(list(), class = "LatentNeuroVec")
  expect_error(latent_dataset(dummy_vec, TR = 2, run_length = 1), "fmristore")
})
