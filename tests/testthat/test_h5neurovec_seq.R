library(testthat)


test_that("H5NeuroVecSeq is consistent with H5NeuroVec", {
  skip_if_not_installed("neuroim2")

  dims <- c(4, 4, 4, 6)
  arr <- array(rnorm(prod(dims)), dims)
  space <- neuroim2::NeuroSpace(dim = dims)
  vec <- neuroim2::NeuroVec(arr, space)

  single_file <- tempfile(fileext = ".h5")
  neuroim2::writeH5NeuroVec(vec, single_file)
  h5_single <- neuroim2::H5NeuroVec(single_file)

  seq_dir <- tempfile()
  dir.create(seq_dir)

  files <- character(dims[4])
  for (i in seq_len(dims[4])) {
    vol <- neuroim2::NeuroVec(arr[,,,i, drop = FALSE],
                              neuroim2::NeuroSpace(c(dims[1:3], 1)))
    files[i] <- file.path(seq_dir, sprintf("vol%02d.h5", i))
    neuroim2::writeH5NeuroVec(vol, files[i])
  }

  h5_seq <- neuroim2::H5NeuroVecSeq(files)

  expect_equal(
    neuroim2::series(h5_single, 1:prod(dims[1:3])),
    neuroim2::series(h5_seq, 1:prod(dims[1:3]))
  )
  expect_equal(neuroim2::dim(h5_seq), neuroim2::dim(h5_single))
})

