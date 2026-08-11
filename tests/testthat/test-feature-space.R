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

.surface_fixture <- function() {
  surface_space(
    vertex_ids = c("L-1", "L-2", "L-3", "R-1", "R-2", "R-3"),
    hemisphere = rep(c("left", "right"), each = 3L),
    topology = rbind(c(1, 2, 3), c(4, 5, 6)),
    geometry = cbind(seq_len(6L), 0, rep(c(0, 1), each = 3L)),
    medial_wall = c(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE),
    template = "fsLR-32k"
  )
}

test_that("surface_space owns stable active vertex identity", {
  x <- .surface_fixture()

  expect_s3_class(x, "surface_space")
  expect_identical(feature_ids(x), c("L-1", "L-3", "R-1", "R-2", "R-3"))
  expect_identical(n_features(x), 5L)
  expect_identical(native_shape(x), c(vertex = 6L))
  expect_identical(feature_data(x)$hemisphere, c("left", "left", "right", "right", "right"))
  expect_identical(feature_data(x)$.vertex_index, c(1L, 3L, 4L, 5L, 6L))
  expect_match(x$topology$digest, "^[0-9a-f]{64}$")
  expect_match(x$geometry$digest, "^[0-9a-f]{64}$")
})

test_that("surface vectorization reconstruction and restriction round trip", {
  x <- .surface_fixture()
  full <- stats::setNames(seq_len(6L) * 10, rev(x$vertex_ids))
  packed <- vectorize_space(x, full)
  map <- reconstruct_space(x, packed)

  expect_s3_class(map, "surface_map")
  expect_identical(packed, c(60, 40, 30, 20, 10))
  expect_true(is.na(map$values[[2L]]))
  expect_identical(vectorize_space(x, map), packed)
  restricted <- restrict_space(x, c(5L, 1L, 3L))
  expect_identical(feature_ids(restricted), c("R-3", "L-1", "R-1"))
  expect_identical(space_digest(restricted), space_digest(restrict_space(x, c(5L, 1L, 3L))))
})

test_that("surface adjacency is induced on active support", {
  x <- .surface_fixture()
  graph <- adjacency(x)

  expect_s4_class(graph, "sparseMatrix")
  expect_identical(dim(graph), c(5L, 5L))
  expect_true(all(graph == Matrix::t(graph)))
  expect_true(graph[1L, 2L])
  right <- as.matrix(graph[3:5, 3:5])
  expect_true(all(right[row(right) != col(right)]))
  expect_false(any(diag(as.matrix(graph))))
})

test_that("surface identity includes topology geometry and support", {
  x <- .surface_fixture()
  y <- x
  y$geometry$data[1L, 1L] <- 999
  y$geometry$digest <- digest::digest(y$geometry$data, algo = "sha256", serialize = TRUE)

  expect_true(compatible_space(x, .surface_fixture())$compatible)
  expect_false(compatible_space(x, y)$compatible)
  expect_false(compatible_space(x, restrict_space(x, 1:3))$compatible)
})

test_that("surface contracts reject ambiguous meshes", {
  expect_error(
    surface_space(c("a", "b"), c("left", "middle")),
    "hemisphere"
  )
  expect_error(
    surface_space(c("a", "b", "c"), rep("left", 3), topology = matrix(c(1, 2, 4), nrow = 1)),
    "topology"
  )
  expect_error(
    surface_space(c("a", "b"), rep("left", 2), support = 1:2, medial_wall = c(TRUE, FALSE)),
    "medial"
  )
})

test_that("surface spaces survive frame and FDS manifest round trips", {
  x <- .surface_fixture()
  frame <- fmri_frame(
    assays = list(beta = matrix(seq_len(10L), nrow = 2L)),
    observations = data.frame(.obs_id = c("o1", "o2")),
    features = feature_axis(feature_data(x), space = x)
  )
  manifest <- fds_frame_manifest(frame)
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list("assays/beta" = memory_source(matrix(seq_len(10L), nrow = 2L)))
  )

  expect_s3_class(space(restored), "surface_space")
  expect_identical(space_digest(space(restored)), space_digest(x))
  expect_identical(feature_ids(restored), feature_ids(x))
})
