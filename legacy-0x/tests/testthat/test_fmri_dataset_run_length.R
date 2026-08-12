# Tests for issue #81: optional/header-derived run_length + per-run validation.

# Write a 4D NeuroVec of the given spatial dims and number of volumes to `path`.
write_scan <- function(path, spatial = c(4, 3, 2), n_vols, spacing = c(2, 2, 3)) {
  dims <- c(spatial, n_vols)
  sp <- neuroim2::NeuroSpace(dim = dims, spacing = spacing)
  vec <- neuroim2::NeuroVec(array(rnorm(prod(dims)), dims), sp)
  neuroim2::write_vec(vec, path)
  invisible(path)
}

make_mask <- function(spatial = c(4, 3, 2), spacing = c(2, 2, 3)) {
  neuroim2::LogicalNeuroVol(
    array(TRUE, spatial),
    neuroim2::NeuroSpace(dim = spatial, spacing = spacing)
  )
}

test_that("run_length is derived from headers when omitted (one run per file)", {
  skip_if_not_installed("neuroim2")

  dir <- withr::local_tempdir()
  f1 <- write_scan(file.path(dir, "run-01.nii"), n_vols = 8)
  f2 <- write_scan(file.path(dir, "run-02.nii"), n_vols = 6)

  dset <- fmri_dataset(scans = c(f1, f2), mask = make_mask(), TR = 2)

  expect_equal(dset$nruns, 2)
  expect_equal(fmrihrf::blocklens(dset$sampling_frame), c(8, 6))
})

test_that("supplied run_length mismatching the headers errors early naming the run", {
  skip_if_not_installed("neuroim2")

  dir <- withr::local_tempdir()
  f1 <- write_scan(file.path(dir, "run-01.nii"), n_vols = 8)
  f2 <- write_scan(file.path(dir, "sub-1049_run-03.nii"), n_vols = 6)

  expect_error(
    fmri_dataset(
      scans = c(f1, f2),
      mask = make_mask(),
      TR = 2,
      run_length = c(8, 7) # second run is wrong (header says 6)
    ),
    "run 2 \\(sub-1049_run-03\\.nii\\).*run_length = 7 but BOLD has 6 volumes"
  )
})

test_that("a correct explicit run_length still constructs (back-compat)", {
  skip_if_not_installed("neuroim2")

  dir <- withr::local_tempdir()
  f1 <- write_scan(file.path(dir, "run-01.nii"), n_vols = 8)
  f2 <- write_scan(file.path(dir, "run-02.nii"), n_vols = 6)

  dset <- fmri_dataset(scans = c(f1, f2), mask = make_mask(), TR = 2, run_length = c(8, 6))
  expect_equal(fmrihrf::blocklens(dset$sampling_frame), c(8, 6))
})

test_that("multi-run run_length on a single 4D file falls back to the total check", {
  skip_if_not_installed("neuroim2")

  dir <- withr::local_tempdir()
  f1 <- write_scan(file.path(dir, "concat.nii"), n_vols = 12)

  # Correct split sums to the total -> constructs.
  dset <- fmri_dataset(scans = f1, mask = make_mask(), TR = 2, run_length = c(6, 6))
  expect_equal(fmrihrf::blocklens(dset$sampling_frame), c(6, 6))

  # Wrong split -> the total-volume check fires.
  expect_error(
    fmri_dataset(scans = f1, mask = make_mask(), TR = 2, run_length = c(6, 5)),
    "Sum of run_length \\(11\\) must equal total time points \\(12\\)"
  )
})

test_that("run_length is required in dummy_mode (no headers to derive from)", {
  skip_if_not_installed("neuroim2")

  expect_error(
    fmri_dataset(
      scans = c("a.nii", "b.nii"),
      mask = make_mask(),
      TR = 2,
      dummy_mode = TRUE
    ),
    "run_length.*must be supplied when dummy_mode"
  )
})
