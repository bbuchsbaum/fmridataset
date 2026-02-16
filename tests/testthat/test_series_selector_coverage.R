# Tests for R/series_selector.R - coverage improvement (print methods and constructors)

test_that("print.index_selector shows indices", {
  sel <- index_selector(1:3)
  out <- capture.output(print(sel))
  expect_true(any(grepl("index_selector", out)))
  expect_true(any(grepl("1, 2, 3", out)))
})

test_that("print.index_selector truncates long indices", {
  sel <- index_selector(1:20)
  out <- capture.output(print(sel))
  expect_true(any(grepl("20 total", out)))
})

test_that("print.voxel_selector shows coordinates", {
  sel <- voxel_selector(c(10, 20, 15))
  out <- capture.output(print(sel))
  expect_true(any(grepl("voxel_selector", out)))
  expect_true(any(grepl("1 voxel", out)))
})

test_that("print.voxel_selector with multiple coordinates", {
  coords <- cbind(x = c(10, 20), y = c(20, 30), z = c(15, 15))
  sel <- voxel_selector(coords)
  out <- capture.output(print(sel))
  expect_true(any(grepl("2 voxel", out)))
})

test_that("print.voxel_selector truncates many coordinates", {
  coords <- cbind(x = 1:10, y = 1:10, z = 1:10)
  sel <- voxel_selector(coords)
  out <- capture.output(print(sel))
  expect_true(any(grepl("more", out)))
})

test_that("print.sphere_selector shows center and radius", {
  sel <- sphere_selector(c(30, 30, 20), radius = 5)
  out <- capture.output(print(sel))
  expect_true(any(grepl("sphere_selector", out)))
  expect_true(any(grepl("Center", out)))
  expect_true(any(grepl("Radius", out)))
})

test_that("print.roi_selector with array", {
  roi <- array(FALSE, dim = c(4, 4, 4))
  roi[2:3, 2:3, 2:3] <- TRUE
  sel <- roi_selector(roi)
  out <- capture.output(print(sel))
  expect_true(any(grepl("roi_selector", out)))
  expect_true(any(grepl("dimensions", out)))
  expect_true(any(grepl("active voxels", out)))
})

test_that("print.mask_selector works", {
  mask <- c(TRUE, FALSE, TRUE, FALSE, TRUE)
  sel <- mask_selector(mask)
  out <- capture.output(print(sel))
  expect_true(any(grepl("mask_selector", out)))
  expect_true(any(grepl("length: 5", out)))
  expect_true(any(grepl("TRUE values: 3", out)))
})

test_that("print.series_selector generic works", {
  sel <- all_selector()
  out <- capture.output(print(sel))
  expect_true(any(grepl("all_selector", out)))
})

test_that("roi_selector errors on invalid input", {
  expect_error(roi_selector("not_an_array"), "3D array")
  expect_error(roi_selector(42), "3D array")
})

test_that("mask_selector coerces numeric to logical", {
  sel <- mask_selector(c(1, 0, 1))
  expect_type(sel$mask, "logical")
  expect_equal(sel$mask, c(TRUE, FALSE, TRUE))
})
