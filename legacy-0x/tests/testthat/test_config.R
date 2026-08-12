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

  cfg_file <- file.path(temp_dir, "config.dcf")
  cat(
    "scans: scan1.nii,scan2.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 2,2\n",
    "event_model: model\n",
    "event_table: events.tsv\n",
    "block_column: run\n",
    "baseline_model: hrf\n",
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)
  expect_s3_class(cfg, "fmri_config")
  expect_equal(cfg$TR, 2)
  expect_equal(cfg$run_length, c(2, 2))
  expect_equal(nrow(cfg$design), 3)
  expect_equal(cfg$base_path, temp_dir)
})

test_that("read_fmri_config handles path resolution correctly", {
  temp_dir <- tempdir()

  # Create event file
  event_file <- file.path(temp_dir, "events.tsv")
  write.table(data.frame(onset = c(1, 2), duration = c(1, 1)),
    event_file,
    row.names = FALSE, sep = "\t"
  )

  # Test with relative event_table path
  cfg_file <- file.path(temp_dir, "config.dcf")
  cat(
    "scans: scan1.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 10\n",
    "event_model: model\n",
    "event_table: events.tsv\n", # Relative path
    "block_column: run\n",
    "baseline_model: hrf\n",
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)
  expect_equal(cfg$base_path, temp_dir)
  expect_equal(nrow(cfg$design), 2)

  # Test with absolute event_table path
  cfg_file2 <- file.path(temp_dir, "config2.dcf")
  cat(
    "scans: scan1.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 10\n",
    "event_model: model\n",
    paste0("event_table: ", event_file, "\n"), # Absolute path
    "block_column: run\n",
    "baseline_model: hrf\n",
    file = cfg_file2
  )

  cfg2 <- read_fmri_config(cfg_file2, base_path = temp_dir)
  expect_equal(nrow(cfg2$design), 2)
})

test_that("read_fmri_config sets default values correctly", {
  temp_dir <- tempdir()

  # Create minimal event file
  event_file <- file.path(temp_dir, "events.tsv")
  write.table(data.frame(onset = 1, duration = 1),
    event_file,
    row.names = FALSE, sep = "\t"
  )

  # Test without base_path in config
  cfg_file <- file.path(temp_dir, "config.dcf")
  cat(
    "scans: scan1.nii\n",
    "TR: 1.5\n",
    "mask: mask.nii\n",
    "run_length: 5\n",
    "event_model: model\n",
    "event_table: events.tsv\n",
    "block_column: run\n",
    "baseline_model: hrf\n",
    # No output_dir specified
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)
  expect_equal(cfg$output_dir, "stat_out") # Default value

  # Test without base_path parameter - will fail because events.tsv doesn't exist in current dir
  expect_error(
    read_fmri_config(cfg_file),
    "Event table file not found"
  )
})

test_that("read_fmri_config validates required fields", {
  temp_dir <- tempdir()

  # Test missing scans
  cfg_file <- file.path(temp_dir, "bad_config1.dcf")
  cat("TR: 2\n", file = cfg_file)

  expect_error(read_fmri_config(cfg_file), "Missing required configuration fields")

  # Test missing TR
  cfg_file2 <- file.path(temp_dir, "bad_config2.dcf")
  cat("scans: scan1.nii\n", file = cfg_file2)

  expect_error(read_fmri_config(cfg_file2), "Missing required configuration fields")
})

test_that("read_fmri_config handles missing event table file", {
  temp_dir <- tempdir()

  cfg_file <- file.path(temp_dir, "config.R")
  cat(
    "scans: scan1.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 10\n",
    "event_model: model\n",
    "event_table: nonexistent.tsv\n", # File doesn't exist
    "block_column: run\n",
    "baseline_model: hrf\n",
    file = cfg_file
  )

  expect_error(read_fmri_config(cfg_file, base_path = temp_dir), "Event table file not found")
})

test_that("read_fmri_config handles optional fields correctly", {
  temp_dir <- tempdir()

  # Create event file
  event_file <- file.path(temp_dir, "events.tsv")
  write.table(data.frame(onset = 1, duration = 1),
    event_file,
    row.names = FALSE, sep = "\t"
  )

  # Test with optional fields
  cfg_file <- file.path(temp_dir, "config.dcf")
  cat(
    "scans: scan1.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 10\n",
    "event_model: model\n",
    "event_table: events.tsv\n",
    "block_column: run\n",
    "baseline_model: hrf\n",
    "censor_file: some_file.txt\n",
    "contrasts: list(a = 1)\n",
    "nuisance: nuisance.txt\n",
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)
  # Optional fields are preserved from the config file
  expect_equal(cfg$censor_file, "some_file.txt")
  expect_equal(cfg$contrasts, "list(a : 1)")
  expect_equal(cfg$nuisance, "nuisance.txt")
})

test_that("default_config creates proper environment", {
  cfg <- fmridataset:::default_config()

  # Check it's an environment
  expect_true(is.list(cfg))

  # Check default values
  expect_equal(cfg$cmd_flags, "")
  expect_equal(cfg$jobs, 1)

  # Check we can modify it
  cfg$new_value <- "test"
  expect_equal(cfg$new_value, "test")
})

test_that("read_fmri_config preserves tibble format for design", {
  temp_dir <- tempdir()

  # Create event file with multiple columns
  event_file <- file.path(temp_dir, "events.tsv")
  events_df <- data.frame(
    onset = c(1, 2, 3),
    duration = c(0.5, 0.5, 0.5),
    condition = c("A", "B", "A"),
    response = c(1, 0, 1)
  )
  write.table(events_df, event_file, row.names = FALSE, sep = "\t")

  cfg_file <- file.path(temp_dir, "config.dcf")
  cat(
    "scans: scan1.nii\n",
    "TR: 2\n",
    "mask: mask.nii\n",
    "run_length: 10\n",
    "event_model: model\n",
    "event_table: events.tsv\n",
    "block_column: run\n",
    "baseline_model: hrf\n",
    file = cfg_file
  )

  cfg <- read_fmri_config(cfg_file, base_path = temp_dir)

  # Check design is a tibble
  expect_s3_class(cfg$design, "tbl_df")
  expect_equal(nrow(cfg$design), 3)
  expect_equal(ncol(cfg$design), 4)
  expect_true("condition" %in% names(cfg$design))
})
