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
    template = "fsLR-32k",
    surf_to_world = diag(c(1, 1, 1, 1))
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

test_that("surface identity respects neurosurf world transforms", {
  x <- .surface_fixture()
  y <- .surface_fixture()
  y$surf_to_world[1L, 4L] <- 10

  expect_false(identical(space_digest(x), space_digest(y)))
  expect_false(compatible_space(x, y)$compatible)
  expect_identical(restrict_space(x, c(1L, 3L))$surf_to_world,
                   x$surf_to_world)
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

test_that("neurosurf geometry adapts and reconstructs without duplicate mesh semantics", {
  withr::local_envvar(RGL_USE_NULL = "TRUE")
  skip_if_not_installed("neurosurf")

  vertices <- matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0),
                     ncol = 3L, byrow = TRUE)
  geom <- neurosurf::SurfaceGeometry(
    vertices,
    matrix(c(0, 1, 2), nrow = 1L),
    hemi = "lh",
    label = "pial",
    surf_to_world = diag(c(2, 2, 2, 1))
  )
  x <- surface_space_from_neurosurf(geom, template = "toy-surface")
  out <- reconstruct_space(x, c(4, 5, 6), format = "neurosurf")

  expect_identical(feature_ids(x), c("L-1", "L-2", "L-3"))
  expect_identical(x$topology$data, matrix(c(1L, 2L, 3L), nrow = 1L))
  expect_identical(x$surf_to_world, diag(c(2, 2, 2, 1)))
  expect_true(methods::is(out, "NeuroSurface"))
  expect_identical(neuroim2::indices(out), 1:3)
  expect_identical(neuroim2::values(out), c(4, 5, 6))
})

.parcel_fixture <- function(aggregation = "mean") {
  parent <- volume_space(c(3, 2, 1), support = 1:6,
                         template = "MNI152NLin6Asym")
  membership <- Matrix::sparseMatrix(
    i = 1:6,
    j = c(1L, 1L, 1L, 2L, 2L, 2L),
    x = 1,
    dims = c(6L, 2L)
  )
  parcel_space(
    parent = parent,
    parcel_ids = c(10L, 20L),
    membership = membership,
    data = data.frame(
      id = c(10L, 20L),
      label = c("Anterior", "Posterior"),
      hemi = c("left", "right")
    ),
    atlas = list(
      id = "toy-atlas",
      name = "Toy atlas",
      version = "1",
      space = "MNI152NLin6Asym"
    ),
    aggregation = aggregation
  )
}

test_that("parcel_space follows neuroatlas parcel identity conventions", {
  x <- .parcel_fixture()

  expect_s3_class(x, "parcel_space")
  expect_s3_class(parent_space(x), "volume_space")
  expect_identical(n_features(x), 2L)
  expect_identical(feature_ids(x), c("toy-atlas:10", "toy-atlas:20"))
  expect_identical(feature_data(x)$id, c(10L, 20L))
  expect_identical(feature_data(x)$label, c("Anterior", "Posterior"))
  expect_identical(feature_data(x)$hemi, c("left", "right"))
  expect_identical(native_shape(x), c(parcel = 2L))
  expect_identical(dim(parcel_membership(x)), c(6L, 2L))
  expect_identical(dim(parcel_aggregation(x)), c(2L, 6L))
})

test_that("parcel aggregation and parent reconstruction have explicit semantics", {
  x <- .parcel_fixture()
  native <- array(1:6, dim = c(3, 2, 1))

  expect_equal(vectorize_space(x, native), c(2, 5))
  painted <- reconstruct_space(x, c(10, 20))
  expect_equal(as.numeric(painted), rep(c(10, 20), each = 3L))
  expect_equal(vectorize_space(x, painted), c(10, 20))

  summed <- .parcel_fixture("sum")
  expect_equal(vectorize_space(summed, native), c(6, 15))
})

test_that("parcel restriction preserves parent identity and atlas metadata", {
  x <- .parcel_fixture()
  y <- restrict_space(x, 2L)

  expect_identical(feature_ids(y), "toy-atlas:20")
  expect_identical(feature_data(y)$id, 20L)
  expect_identical(space_digest(parent_space(y)),
                   space_digest(parent_space(x)))
  expect_identical(y$atlas, x$atlas)
  expect_identical(dim(parcel_membership(y)), c(6L, 1L))
})

test_that("parcel adjacency is induced through the parent space", {
  x <- .parcel_fixture()
  graph <- adjacency(x)

  expect_s4_class(graph, "sparseMatrix")
  expect_identical(dim(graph), c(2L, 2L))
  expect_true(graph[1L, 2L])
  expect_true(graph[2L, 1L])
  expect_false(any(diag(as.matrix(graph))))
})

test_that("parcel identity includes parent, atlas, membership, and aggregation", {
  x <- .parcel_fixture()
  expect_true(compatible_space(x, .parcel_fixture())$compatible)
  expect_false(compatible_space(x, .parcel_fixture("sum"))$compatible)

  y <- .parcel_fixture()
  y$atlas$version <- "2"
  expect_false(identical(space_digest(x), space_digest(y)))

  z <- .parcel_fixture()
  z$membership[3L, 1L] <- 0
  z$membership[3L, 2L] <- 1
  expect_false(identical(space_digest(x), space_digest(z)))
})

test_that("parcel contracts reject ambiguous operators", {
  parent <- volume_space(c(2, 2, 1))
  data <- data.frame(id = c(1L, 2L), label = c("a", "b"),
                     hemi = c("left", "right"))

  expect_error(
    parcel_space(parent, 1:2, matrix(1, 3, 2), data,
                 atlas = list(id = "toy")),
    "parent"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, 1, 0, 0), 0), data,
                 atlas = list(id = "toy")),
    "parcel"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, -1, 0, 0), c(0, 0, 1, 1)), data,
                 atlas = list(id = "toy")),
    "non-negative"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, 1, 0, 0), c(0, 0, 1, 1)), data,
                 atlas = list(name = "missing id")),
    "atlas.*id"
  )
})

test_that("parcel spaces survive frame and FDS manifest round trips", {
  x <- .parcel_fixture()
  frame <- fmri_frame(
    assays = list(beta = matrix(seq_len(4L), nrow = 2L)),
    observations = data.frame(.obs_id = c("o1", "o2")),
    features = feature_axis(feature_data(x), space = x)
  )
  manifest <- fds_frame_manifest(frame)
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list("assays/beta" = memory_source(matrix(seq_len(4L), nrow = 2L)))
  )

  expect_s3_class(space(restored), "parcel_space")
  expect_identical(space_digest(space(restored)), space_digest(x))
  expect_identical(feature_ids(restored), feature_ids(x))
})

test_that("parcel spaces survive physical HDF5 frame round trips", {
  skip_if_not_installed("fmristore")
  x <- .parcel_fixture()
  frame <- fmri_frame(
    assays = list(beta = matrix(seq_len(4L), nrow = 2L)),
    observations = data.frame(.obs_id = c("o1", "o2")),
    features = feature_axis(feature_data(x), space = x)
  )
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path), add = TRUE)

  write_frame(frame, path)
  restored <- open_frame(path)

  expect_s3_class(space(restored), "parcel_space")
  expect_identical(space_digest(space(restored)), space_digest(x))
  expect_equal(collect_assay(restored), matrix(seq_len(4L), nrow = 2L))
})

test_that("neuroatlas surface coding is delegated through get_roi", {
  withr::local_envvar(RGL_USE_NULL = "TRUE")
  skip_if_not_installed("neurosurf")
  skip_if_not_installed("neuroatlas")

  left_vertices <- matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0),
                          ncol = 3L, byrow = TRUE)
  right_vertices <- sweep(left_vertices, 2L, c(0, 0, 1), "+")
  faces <- matrix(c(0, 1, 2), nrow = 1L)
  left_geom <- neurosurf::SurfaceGeometry(left_vertices, faces, "lh")
  right_geom <- neurosurf::SurfaceGeometry(right_vertices, faces, "rh")
  atlas <- list(
    name = "toy-surface",
    lh_atlas = neurosurf::NeuroSurface(left_geom, 1:3, c(1, 1, 0)),
    rh_atlas = neurosurf::NeuroSurface(right_geom, 1:3, c(2, 2, 0)),
    ids = 1:2,
    labels = c("A", "B"),
    orig_labels = c("A", "B"),
    hemi = c("left", "right"),
    network = NULL,
    cmap = NULL,
    surf_type = "pial",
    surface_space = "toy"
  )
  class(atlas) <- c("toy", "surfatlas", "atlas")
  parent <- surface_space(
    vertex_ids = c(paste0("L-", 1:3), paste0("R-", 1:3)),
    hemisphere = rep(c("left", "right"), each = 3L),
    topology = rbind(c(1, 2, 3), c(4, 5, 6)),
    geometry = rbind(left_vertices, right_vertices),
    template = "toy"
  )
  x <- parcel_space_from_atlas(atlas, parent)

  expect_identical(feature_ids(x), c("toy-surface:1", "toy-surface:2"))
  expect_identical(
    as.matrix(parcel_membership(x)),
    rbind(c(1, 0), c(1, 0), c(0, 0),
          c(0, 1), c(0, 1), c(0, 0))
  )
  expect_identical(feature_data(x)$hemi, c("left", "right"))
})
