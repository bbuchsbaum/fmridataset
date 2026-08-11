test_that("index_space has namespaced stable feature IDs", {
  x <- index_space(3)
  y <- index_space(3)

  expect_equal(n_features(x), 3)
  expect_length(feature_ids(x), 3)
  expect_false(identical(feature_ids(x), feature_ids(y)))
  expect_false(identical(space_digest(x), space_digest(y)))
})

test_that("volume_space uses packed full-volume indices", {
  x <- volume_space(c(2, 2, 2), affine = diag(4), support = c(1, 3, 8))

  expect_equal(n_features(x), 3)
  expect_identical(feature_ids(x), c("voxel-1", "voxel-3", "voxel-8"))
  expect_identical(native_shape(x), c(2L, 2L, 2L))
  expect_equal(feature_data(x)$.linear_index, c(1L, 3L, 8L))
})

test_that("space compatibility uses digest and feature IDs", {
  x <- volume_space(c(2, 2, 2), affine = diag(4), support = 1:4)
  y <- volume_space(c(2, 2, 2), affine = diag(4), support = 1:4)
  z <- volume_space(c(2, 2, 2), affine = diag(4), support = 2:5)

  expect_true(compatible_space(x, y)$compatible)
  expect_false(compatible_space(x, z)$compatible)
  expect_error(assert_compatible_space(x, z), class = "fmridataset_error_space_mismatch")
})

test_that("volume vectorization and reconstruction round trip", {
  x <- volume_space(c(2, 2, 2), affine = diag(4), support = c(1, 3, 8))
  a <- array(seq_len(8), dim = c(2, 2, 2))

  v <- vectorize_space(x, a)
  out <- reconstruct_space(x, v)

  expect_equal(v, c(1, 3, 8))
  expect_equal(as.numeric(out)[c(1, 3, 8)], v)
  expect_true(all(is.na(as.numeric(out)[-c(1, 3, 8)])))
})

test_that("restricted spaces preserve selected feature identity", {
  x <- volume_space(c(2, 2, 2), affine = diag(4), support = 1:6)
  y <- restrict_space(x, c(6, 2))

  expect_identical(feature_ids(y), c("voxel-6", "voxel-2"))
  expect_equal(n_features(y), 2)
})
