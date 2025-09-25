# Comprehensive path handling tests for fmridataset

make_dummy_dataset <- function(scans, mask, ..., base_path = ".", runs = NULL) {
  if (is.null(runs)) {
    runs <- rep(100, length(scans))
  }
  fmri_dataset(
    scans = scans,
    mask = mask,
    TR = 2,
    run_length = runs,
    base_path = base_path,
    dummy_mode = TRUE,
    ...
  )
}

test_that("relative paths are resolved against base_path", {
  skip_on_cran()
  skip_if_not_installed("withr")

  base_dir <- withr::local_tempdir()
  data_dir <- file.path(base_dir, "data")
  dir.create(file.path(data_dir, "sub-01"), recursive = TRUE)

  scans <- c("sub-01/run-01.nii", "sub-01/run-02.nii")
  dset <- make_dummy_dataset(
    scans = scans,
    mask = "mask.nii",
    base_path = data_dir,
    runs = rep(100, length(scans))
  )

  backend <- dset$backend
  expect_equal(
    backend$source,
    file.path(data_dir, scans)
  )
  expect_equal(
    backend$mask_source,
    file.path(data_dir, "mask.nii")
  )
})

test_that("absolute scan and mask paths are preserved", {
  skip_on_cran()

  abs_scans <- normalizePath(rep(tempfile(pattern = "run"), 2), mustWork = FALSE)
  abs_mask <- normalizePath(tempfile(pattern = "mask"), mustWork = FALSE)

  dset <- make_dummy_dataset(
    scans = abs_scans,
    mask = abs_mask,
    runs = rep(100, length(abs_scans))
  )

  backend <- dset$backend
  expect_equal(backend$source, abs_scans)
  expect_equal(backend$mask_source, abs_mask)
})

test_that("paths containing spaces are retained", {
  skip_on_cran()
  skip_if_not_installed("withr")

  base_dir <- withr::local_tempdir()
  spaced_dir <- file.path(base_dir, "path with spaces", "nested dir")
  dir.create(spaced_dir, recursive = TRUE)

  dset <- make_dummy_dataset(
    scans = "scan.nii",
    mask = "mask.nii",
    base_path = spaced_dir,
    runs = 100
  )

  backend <- dset$backend
  expect_equal(backend$source, file.path(spaced_dir, "scan.nii"))
  expect_equal(backend$mask_source, file.path(spaced_dir, "mask.nii"))
})

test_that("special characters in base paths are supported", {
  skip_on_cran()
  skip_if_not_installed("withr")

  chars <- c("user's", "data(backup)", "set[2023]", "this&that", "name#1", "user@host")
  if (.Platform$OS.type != "windows") {
    chars <- c(chars, "time:12:00", "asterisk*")
  }

  temp_root <- withr::local_tempdir()
  for (dir_name in chars) {
    special_dir <- file.path(temp_root, dir_name)
    dir.create(special_dir, recursive = TRUE, showWarnings = FALSE)

    dset <- make_dummy_dataset(
      scans = "scan.nii",
      mask = "mask.nii",
      base_path = special_dir,
      runs = 100
    )

    backend <- dset$backend
    expect_equal(backend$source, file.path(special_dir, "scan.nii"))
  }
})

test_that("path normalization yields consistent locations", {
  skip_on_cran()
  skip_if_not_installed("withr")

  tmp <- withr::local_tempdir()
  variants <- list(
    c("./data", "data"),
    c("data/../data", "data"),
    c("./data/./sub", "data/sub"),
    c("data//sub", "data/sub")
  )

  for (pair in variants) {
    paths <- file.path(tmp, pair)
    file_paths <- file.path(paths, "scan.nii")
    unique_dirs <- unique(dirname(file_paths))
    for (dir in unique_dirs) {
      dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    }
    for (fp in file_paths) {
      file.create(fp, showWarnings = FALSE)
    }
    withr::defer(unlink(unique(dirname(file_paths)), recursive = TRUE))

    d1 <- make_dummy_dataset(
      scans = "scan.nii",
      mask = "mask.nii",
      base_path = paths[1],
      runs = 100
    )
    d2 <- make_dummy_dataset(
      scans = "scan.nii",
      mask = "mask.nii",
      base_path = paths[2],
      runs = 100
    )

    expect_equal(
      normalizePath(dirname(d1$backend$source), mustWork = TRUE),
      normalizePath(dirname(d2$backend$source), mustWork = TRUE)
    )
    expect_equal(
      basename(d1$backend$source),
      basename(d2$backend$source)
    )
  }
})

test_that("symlink base paths are honoured", {
  skip_on_cran()
  skip_if(.Platform$OS.type == "windows", "Symlink support is limited on Windows")
  skip_if_not_installed("withr")

  base_dir <- withr::local_tempdir()
  target <- file.path(base_dir, "real_data")
  link <- file.path(base_dir, "linked_data")

  dir.create(target, recursive = TRUE)
  file.symlink(target, link)
  withr::defer(unlink(link))

  dset <- make_dummy_dataset(
    scans = "scan.nii",
    mask = "mask.nii",
    base_path = link,
    runs = 100
  )

  expect_equal(dset$backend$source, file.path(link, "scan.nii"))
})
