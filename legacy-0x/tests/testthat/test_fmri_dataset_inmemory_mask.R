test_that("fmri_dataset() accepts an in-memory mask without touching disk", {
  skip_if_not_installed("neuroim2")

  dims <- c(6, 5, 4, 8)
  vspace <- neuroim2::NeuroSpace(dim = dims, spacing = c(2, 2, 3), origin = c(10, 20, 30))
  mspace <- neuroim2::NeuroSpace(dim = dims[1:3], spacing = c(2, 2, 3), origin = c(10, 20, 30))

  # Real on-disk scan file; in-memory mask (the round-trip we want to avoid).
  scan_dir <- withr::local_tempdir()
  scan_file <- file.path(scan_dir, "bold.nii")
  vec <- neuroim2::NeuroVec(array(rnorm(prod(dims)), dims), vspace)
  neuroim2::write_vec(vec, scan_file)

  mask_arr <- array(TRUE, dims[1:3])
  mask_arr[1, 1, 1] <- FALSE
  mask_vol <- neuroim2::LogicalNeuroVol(mask_arr, mspace)

  files_before <- list.files(scan_dir)

  dset <- fmri_dataset(
    scans = scan_file,
    mask = mask_vol,
    TR = 2,
    run_length = dims[4]
  )

  # The in-memory mask must not have been written anywhere on disk.
  expect_identical(list.files(scan_dir), files_before)

  expect_s3_class(dset, "fmri_file_dataset")

  # The mask actually used is the one we passed (values + spatial metadata),
  # not a default all-TRUE mask.
  m <- get_mask(dset)
  expect_true(inherits(m, "LogicalNeuroVol"))
  expect_equal(dim(m), dims[1:3])
  expect_equal(as.logical(as.vector(m)), as.logical(as.vector(mask_arr)))
  expect_equal(
    neuroim2::spacing(neuroim2::space(m)),
    neuroim2::spacing(mspace)
  )
})

test_that("fmri_dataset() still accepts a mask file path (back-compat)", {
  skip_if_not_installed("neuroim2")

  dims <- c(6, 5, 4, 8)
  vspace <- neuroim2::NeuroSpace(dim = dims, spacing = c(2, 2, 3))
  mspace <- neuroim2::NeuroSpace(dim = dims[1:3], spacing = c(2, 2, 3))

  scan_dir <- withr::local_tempdir()
  scan_file <- file.path(scan_dir, "bold.nii")
  vec <- neuroim2::NeuroVec(array(rnorm(prod(dims)), dims), vspace)
  neuroim2::write_vec(vec, scan_file)

  mask_file <- file.path(scan_dir, "mask.nii")
  mask_vol <- neuroim2::LogicalNeuroVol(array(TRUE, dims[1:3]), mspace)
  neuroim2::write_vol(mask_vol, mask_file)

  dset <- fmri_dataset(
    scans = scan_file,
    mask = mask_file,
    TR = 2,
    run_length = dims[4]
  )

  expect_s3_class(dset, "fmri_file_dataset")
  expect_equal(sum(get_mask(dset)), prod(dims[1:3]))
})

test_that("fmri_dataset() rejects a mask that is neither a path nor a NeuroVol", {
  expect_error(
    fmri_dataset(
      scans = c("a.nii", "b.nii"),
      mask = 1:10,
      TR = 2,
      run_length = c(5, 5)
    ),
    "file path .* or a neuroim2 NeuroVol"
  )
})
