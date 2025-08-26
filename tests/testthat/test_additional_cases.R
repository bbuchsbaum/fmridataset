library(fmridataset)

test_that("fmri_dataset prepends base_path", {
  temp_dir <- tempdir()
  # create placeholder files so existence checks pass
  scan_file <- file.path(temp_dir, "scan.nii")
  mask_file <- file.path(temp_dir, "mask.nii")
  file.create(scan_file)
  file.create(mask_file)

  with_mocked_bindings(
    nifti_backend = function(source, mask_source, preload = FALSE, mode = "normal", ...) {
      structure(list(source = source, mask_source = mask_source),
        class = c("nifti_backend", "storage_backend")
      )
    },
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(1, 1, 1), time = 10),
    .package = "fmridataset",
    {
      dset <- fmri_dataset(
        scans = "scan.nii",
        mask = "mask.nii",
        TR = 1,
        run_length = 10,
        base_path = temp_dir
      )
      expect_equal(dset$backend$source, scan_file)
      expect_equal(dset$backend$mask_source, mask_file)
    }
  )

  unlink(c(scan_file, mask_file))
})

test_that("fmri_dataset leaves absolute paths unchanged", {
  temp_dir <- tempdir()
  scan_file <- file.path(temp_dir, "abs_scan.nii")
  mask_file <- file.path(temp_dir, "abs_mask.nii")
  file.create(scan_file)
  file.create(mask_file)

  with_mocked_bindings(
    nifti_backend = function(source, mask_source, preload = FALSE, mode = "normal", ...) {
      structure(list(source = source, mask_source = mask_source),
        class = c("nifti_backend", "storage_backend")
      )
    },
    validate_backend = function(backend) TRUE,
    backend_open = function(backend) backend,
    backend_get_dims = function(backend) list(spatial = c(1, 1, 1), time = 10),
    .package = "fmridataset",
    {
      dset <- fmri_dataset(
        scans = scan_file,
        mask = mask_file,
        TR = 1,
        run_length = 10,
        base_path = temp_dir
      )
      expect_equal(dset$backend$source, scan_file)
      expect_equal(dset$backend$mask_source, mask_file)
    }
  )

  unlink(c(scan_file, mask_file))
})


test_that("study_backend rejects unknown strict setting", {
  b <- matrix_backend(matrix(1:10, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  expect_error(
    study_backend(list(b), strict = "foo"),
    "unknown strict setting"
  )
})


test_that("fmri_study_dataset requires equal TR across datasets", {
  b1 <- matrix_backend(matrix(1:10, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  b2 <- matrix_backend(matrix(11:20, nrow = 5, ncol = 2), spatial_dims = c(2, 1, 1))
  d1 <- fmri_dataset(b1, TR = 2, run_length = 5)
  d2 <- fmri_dataset(b2, TR = 1, run_length = 5)
  expect_error(
    fmri_study_dataset(list(d1, d2), subject_ids = c("s1", "s2")),
    "All datasets must have equal TR"
  )
})
