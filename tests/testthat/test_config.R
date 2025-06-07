library(fmridataset)

test_that("default_config returns expected defaults", {
  cfg <- fmridataset:::default_config()
  expect_equal(cfg$cmd_flags, "")
  expect_equal(cfg$jobs, 1)
})


test_that("read_fmri_config parses configuration files", {
  temp_dir <- tempdir()
  event_file <- file.path(temp_dir, "events.tsv")
  write.table(data.frame(onset = c(1, 2, 3), duration = c(0.5, 0.5, 0.5)),
    event_file,
    row.names = FALSE, sep = "\t"
  )

  cfg_file <- file.path(temp_dir, "config.R")
  cat(
    "scans <- c('scan1.nii', 'scan2.nii')\n",
    "TR <- 2\n",
    "mask <- 'mask.nii'\n",
    "run_length <- c(2,2)\n",
    "event_model <- 'model'\n",
    "event_table <- 'events.tsv'\n",
    "block_column <- 'run'\n",
    "baseline_model <- 'hrf'\n",
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)
  expect_s3_class(cfg, "fmri_config")
  expect_equal(cfg$TR, 2)
  expect_equal(cfg$run_length, c(2, 2))
  expect_equal(nrow(cfg$design), 3)
  expect_equal(cfg$base_path, temp_dir)
})
