# A study backend hands out column positions in the COMBINED mask, but each
# child backend resolves columns against its OWN mask. When the masks differ
# those numberings disagree, and before column routing existed every subject
# was silently offset by its own mask surplus -- different anatomy under the
# same column label, with no error.
#
# The fixtures below encode each voxel's identity in its value, so a
# misalignment is directly readable in the returned matrix.

# Matrix backend whose column k holds the value of full-volume voxel k.
identity_matrix_backend <- function(mask, n_time, offset = 0) {
  n_vox <- length(mask)
  datamat <- matrix(
    rep(seq_len(n_vox) + offset, each = n_time),
    nrow = n_time, ncol = n_vox
  )
  matrix_backend(datamat, mask = mask, spatial_dims = c(n_vox, 1, 1))
}

test_that("intersect mode returns the same voxel for every subject", {
  n_vox <- 64
  n_time <- 3

  mask1 <- rep(TRUE, n_vox)
  mask2 <- rep(TRUE, n_vox)
  mask2[1] <- FALSE # subject 2 is missing the first voxel

  b1 <- identity_matrix_backend(mask1, n_time, offset = 0)
  b2 <- identity_matrix_backend(mask2, n_time, offset = 1000)

  sb <- study_backend(
    list(b1, b2),
    subject_ids = c("s1", "s2"),
    strict = "intersect"
  )

  combined_voxels <- which(backend_get_mask(sb))
  expect_equal(combined_voxels, 2:n_vox)

  dat <- backend_get_data(sb, rows = seq_len(2 * n_time), cols = 1:4)

  # Columns 1:4 of the combined mask are full-volume voxels 2,3,4,5.
  expect_equal(unname(dat[1, ]), c(2, 3, 4, 5))
  expect_equal(unname(dat[n_time + 1, ]), c(1002, 1003, 1004, 1005))

  # Stated generally: subject 2's values are subject 1's plus the offset,
  # column for column. That equality is exactly what misrouting breaks.
  expect_equal(
    unname(dat[seq_len(n_time), ]) + 1000,
    unname(dat[n_time + seq_len(n_time), ])
  )
})

test_that("intersect routing holds for every column of the combined mask", {
  n_vox <- 64
  n_time <- 2

  mask1 <- rep(TRUE, n_vox)
  mask1[3] <- FALSE
  mask2 <- rep(TRUE, n_vox)
  mask2[7] <- FALSE

  sb <- study_backend(
    list(
      identity_matrix_backend(mask1, n_time, offset = 0),
      identity_matrix_backend(mask2, n_time, offset = 1000)
    ),
    subject_ids = c("s1", "s2"),
    strict = "intersect"
  )

  combined_voxels <- which(backend_get_mask(sb))
  expect_equal(combined_voxels, setdiff(seq_len(n_vox), c(3, 7)))

  dat <- backend_get_data(sb, rows = seq_len(2 * n_time))

  # Every column must carry the full-volume voxel the combined mask names.
  expect_equal(unname(dat[1, ]), as.numeric(combined_voxels))
  expect_equal(unname(dat[n_time + 1, ]), as.numeric(combined_voxels) + 1000)
})

test_that("column routing survives reordered and repeated column requests", {
  n_vox <- 64
  n_time <- 2

  mask1 <- rep(TRUE, n_vox)
  mask2 <- rep(TRUE, n_vox)
  mask2[1] <- FALSE

  sb <- study_backend(
    list(
      identity_matrix_backend(mask1, n_time, offset = 0),
      identity_matrix_backend(mask2, n_time, offset = 1000)
    ),
    subject_ids = c("s1", "s2"),
    strict = "intersect"
  )

  combined_voxels <- which(backend_get_mask(sb))
  requested <- c(5L, 1L, 3L, 1L)

  dat <- backend_get_data(sb, rows = seq_len(2 * n_time), cols = requested)

  expect_equal(unname(dat[1, ]), as.numeric(combined_voxels[requested]))
  expect_equal(unname(dat[n_time + 1, ]), as.numeric(combined_voxels[requested]) + 1000)
})

test_that("identical mode routing is the identity", {
  n_vox <- 64
  n_time <- 2
  mask <- rep(TRUE, n_vox)

  sb <- study_backend(
    list(
      identity_matrix_backend(mask, n_time, offset = 0),
      identity_matrix_backend(mask, n_time, offset = 1000)
    ),
    subject_ids = c("s1", "s2"),
    strict = "identical"
  )

  expect_equal(sb$`_col_maps`[[1]], seq_len(n_vox))
  expect_equal(sb$`_col_maps`[[2]], seq_len(n_vox))

  dat <- backend_get_data(sb, rows = seq_len(2 * n_time))
  expect_equal(unname(dat[1, ]), as.numeric(seq_len(n_vox)))
  expect_equal(unname(dat[n_time + 1, ]), as.numeric(seq_len(n_vox)) + 1000)
})

test_that("a backend without stored routing still routes correctly", {
  n_vox <- 64
  n_time <- 2

  mask1 <- rep(TRUE, n_vox)
  mask2 <- rep(TRUE, n_vox)
  mask2[1] <- FALSE

  sb <- study_backend(
    list(
      identity_matrix_backend(mask1, n_time, offset = 0),
      identity_matrix_backend(mask2, n_time, offset = 1000)
    ),
    subject_ids = c("s1", "s2"),
    strict = "intersect"
  )

  stored <- sb$`_col_maps`
  sb$`_col_maps` <- NULL # as if built by an older version

  dat <- backend_get_data(sb, rows = seq_len(2 * n_time), cols = 1:3)
  combined_voxels <- which(backend_get_mask(sb))

  expect_equal(unname(dat[1, ]), as.numeric(combined_voxels[1:3]))
  expect_equal(unname(dat[n_time + 1, ]), as.numeric(combined_voxels[1:3]) + 1000)
  expect_false(is.null(stored))
})

test_that("masks over different voxel grids are refused", {
  expect_error(
    study_backend(
      list(
        identity_matrix_backend(rep(TRUE, 64), 2),
        identity_matrix_backend(rep(TRUE, 80), 2)
      ),
      subject_ids = c("s1", "s2"),
      strict = "intersect"
    ),
    class = "fmridataset_error"
  )
})
