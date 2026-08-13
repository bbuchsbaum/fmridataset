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
  expect_identical(
    restrict_space(x, c(1L, 3L))$surf_to_world,
    x$surf_to_world
  )
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
    ncol = 3L, byrow = TRUE
  )
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
  parent <- volume_space(c(3, 2, 1),
    support = 1:6,
    template = "MNI152NLin6Asym"
  )
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
  expect_identical(
    space_digest(parent_space(y)),
    space_digest(parent_space(x))
  )
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
  data <- data.frame(
    id = c(1L, 2L), label = c("a", "b"),
    hemi = c("left", "right")
  )

  expect_error(
    parcel_space(parent, 1:2, matrix(1, 3, 2), data,
      atlas = list(id = "toy")
    ),
    "parent"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, 1, 0, 0), 0), data,
      atlas = list(id = "toy")
    ),
    "parcel"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, -1, 0, 0), c(0, 0, 1, 1)), data,
      atlas = list(id = "toy")
    ),
    "non-negative"
  )
  expect_error(
    parcel_space(parent, 1:2, cbind(c(1, 1, 0, 0), c(0, 0, 1, 1)), data,
      atlas = list(name = "missing id")
    ),
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
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
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
    ncol = 3L, byrow = TRUE
  )
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
    rbind(
      c(1, 0), c(1, 0), c(0, 0),
      c(0, 1), c(0, 1), c(0, 0)
    )
  )
  expect_identical(feature_data(x)$hemi, c("left", "right"))
})

.basis_fixture <- function(operator_backend = c("matrix", "sparse", "source")) {
  operator_backend <- match.arg(operator_backend)
  parent <- volume_space(c(2, 2, 1),
    support = 1:4,
    template = "MNI152NLin6Asym"
  )
  decoder <- matrix(
    c(
      1, 0,
      0, 1,
      1, 1,
      2, -1
    ),
    nrow = 4L, byrow = TRUE
  )
  encoder <- solve(crossprod(decoder), t(decoder))
  if (operator_backend == "sparse") {
    decoder <- Matrix::Matrix(decoder, sparse = TRUE)
    encoder <- Matrix::Matrix(encoder, sparse = TRUE)
  } else if (operator_backend == "source") {
    decoder <- memory_source(decoder, chunks = c(2L, 1L))
    encoder <- memory_source(encoder, chunks = c(1L, 2L))
  }
  basis_space(
    parent = parent,
    component_ids = c("smooth", "contrast"),
    encoder = encoder,
    decoder = decoder,
    data = data.frame(
      component_id = c("smooth", "contrast"),
      label = c("Smooth field", "Spatial contrast")
    ),
    basis_type = "toy_dictionary",
    provenance = list(method = "analytic", version = "1")
  )
}

test_that("basis_space owns stable component and parent identity", {
  x <- .basis_fixture()

  expect_s3_class(x, "basis_space")
  expect_s3_class(parent_space(x), "volume_space")
  expect_identical(n_features(x), 2L)
  expect_identical(feature_ids(x), c("smooth", "contrast"))
  expect_identical(native_shape(x), c(component = 2L))
  expect_identical(
    feature_data(x)$component_id,
    c("smooth", "contrast")
  )
  expect_identical(
    feature_data(x)$label,
    c("Smooth field", "Spatial contrast")
  )
  expect_identical(dim(basis_analysis(x)), c(2L, 4L))
  expect_identical(dim(basis_synthesis(x)), c(4L, 2L))
  expect_true(basis_projection_info(x)$left_inverse_validated)
  expect_lt(basis_projection_info(x)$left_inverse_error, 1e-12)
})

test_that("basis projection is least squares for non-orthonormal dictionaries", {
  x <- .basis_fixture()
  decoder <- basis_synthesis(x)
  coefficients <- c(2.5, -1.25)
  parent_vector <- as.numeric(decoder %*% coefficients)
  native <- array(parent_vector, dim = c(2, 2, 1))

  expect_equal(vectorize_space(x, native), coefficients, tolerance = 1e-12)
  expect_equal(as.numeric(reconstruct_space(x, coefficients)), parent_vector,
    tolerance = 1e-12
  )
  expect_equal(
    vectorize_space(x, reconstruct_space(x, coefficients)),
    coefficients,
    tolerance = 1e-12
  )

  noisy <- parent_vector + c(0.1, -0.2, 0.05, 0.15)
  expected <- solve(crossprod(decoder), crossprod(decoder, noisy))
  expect_equal(
    vectorize_space(x, array(noisy, dim = c(2, 2, 1))),
    as.numeric(expected),
    tolerance = 1e-12
  )
})

test_that("dense sparse and lazy basis operators are equivalent", {
  dense <- .basis_fixture("matrix")
  sparse <- .basis_fixture("sparse")
  lazy <- .basis_fixture("source")
  native <- array(c(0.3, -1.2, 2.1, 0.7), dim = c(2, 2, 1))
  coefficients <- c(-0.5, 1.7)

  expect_equal(vectorize_space(sparse, native),
    vectorize_space(dense, native),
    tolerance = 1e-12
  )
  expect_equal(vectorize_space(lazy, native),
    vectorize_space(dense, native),
    tolerance = 1e-12
  )
  expect_equal(as.numeric(reconstruct_space(sparse, coefficients)),
    as.numeric(reconstruct_space(dense, coefficients)),
    tolerance = 1e-12
  )
  expect_equal(as.numeric(reconstruct_space(lazy, coefficients)),
    as.numeric(reconstruct_space(dense, coefficients)),
    tolerance = 1e-12
  )
  expect_true(compatible_space(dense, sparse)$compatible)
  expect_true(compatible_space(dense, lazy)$compatible)
})

test_that("basis restriction preserves parent and revalidates coordinates", {
  x <- .basis_fixture()
  y <- restrict_space(x, 2L)

  expect_identical(feature_ids(y), "contrast")
  expect_identical(feature_data(y)$component_id, "contrast")
  expect_identical(
    space_digest(parent_space(y)),
    space_digest(parent_space(x))
  )
  expect_identical(dim(basis_analysis(y)), c(1L, 4L))
  expect_identical(dim(basis_synthesis(y)), c(4L, 1L))
  expect_equal(
    vectorize_space(y, reconstruct_space(y, 3.2)),
    3.2,
    tolerance = 1e-12
  )
})

test_that("encode-only basis spaces fail reconstruction explicitly", {
  full <- .basis_fixture()
  x <- basis_space(
    parent = parent_space(full),
    component_ids = feature_ids(full),
    encoder = basis_analysis(full),
    decoder = NULL,
    basis_type = "encode_only",
    provenance = list(method = "external")
  )
  native <- array(1:4, dim = c(2, 2, 1))

  expect_length(vectorize_space(x, native), 2L)
  expect_null(basis_synthesis(x))
  expect_error(reconstruct_space(x, c(1, 2)), "decoder")
})

test_that("basis identity includes operators ordering type and provenance", {
  x <- .basis_fixture()
  expect_true(compatible_space(x, .basis_fixture())$compatible)

  reordered <- restrict_space(x, c(2L, 1L))
  expect_false(identical(space_digest(x), space_digest(reordered)))

  changed_type <- .basis_fixture()
  changed_type$basis_type <- "different"
  expect_false(identical(space_digest(x), space_digest(changed_type)))

  changed_provenance <- .basis_fixture()
  changed_provenance$provenance$version <- "2"
  expect_false(identical(space_digest(x), space_digest(changed_provenance)))

  changed_decoder <- .basis_fixture()
  changed_decoder$decoder[1L, 1L] <- 2
  expect_false(identical(space_digest(x), space_digest(changed_decoder)))
})

test_that("basis contracts reject ambiguous or non-identifiable maps", {
  parent <- volume_space(c(2, 2, 1))
  decoder <- cbind(c(1, 0, 1, 0), c(0, 1, 0, 1))
  encoder <- solve(crossprod(decoder), t(decoder))

  expect_error(
    basis_space(parent, character(), matrix(numeric(), 0, 4)),
    "at least one component"
  )
  expect_error(
    basis_space(parent, c("a", "b"), matrix(1, 2, 3), decoder),
    "encoder"
  )
  expect_error(
    basis_space(parent, c("a", "b"), encoder, matrix(1, 3, 2)),
    "decoder"
  )
  bad_encoder <- encoder
  bad_encoder[1L, 1L] <- Inf
  expect_error(
    basis_space(parent, c("a", "b"), bad_encoder, decoder),
    "finite"
  )
  expect_error(
    basis_space(parent, c("a", "b"), encoder * 2, decoder),
    "left inverse"
  )
  expect_error(
    basis_space_from_decoder(
      parent, c("a", "b"),
      cbind(c(1, 0, 1, 0), c(2, 0, 2, 0))
    ),
    "full column rank"
  )
})

test_that("basis spaces survive logical and physical FDS round trips", {
  x <- .basis_fixture("sparse")
  frame <- fmri_frame(
    assays = list(scores = matrix(seq_len(6L), nrow = 3L)),
    observations = data.frame(.obs_id = paste0("o", 1:3)),
    features = feature_axis(feature_data(x), space = x)
  )
  manifest <- fds_frame_manifest(frame)
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list(
      "assays/scores" = memory_source(matrix(seq_len(6L), nrow = 3L))
    )
  )
  expect_s3_class(space(restored), "basis_space")
  expect_identical(space_digest(space(restored)), space_digest(x))

  skip_if_not_installed("fmristore")
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path), add = TRUE)
  write_frame(frame, path)
  reopened <- open_frame(path)
  expect_s3_class(space(reopened), "basis_space")
  expect_identical(space_digest(space(reopened)), space_digest(x))
  expect_equal(collect_assay(reopened), matrix(seq_len(6L), nrow = 3L))
})

test_that("fmrilatent loadings adapt without stealing model ownership", {
  skip_if_not_installed("fmrilatent")
  parent <- volume_space(c(2, 2, 1),
    support = 1:4,
    template = "toy-native"
  )
  decoder <- Matrix::Matrix(
    matrix(c(1, 0, 0, 1, 1, 1, 2, -1), nrow = 4L, byrow = TRUE),
    sparse = TRUE
  )
  scores <- Matrix::Matrix(matrix(c(1, 0, 0, 1, 2, -1),
    nrow = 3L,
    byrow = TRUE
  ))
  mask_array <- array(TRUE, dim = c(2, 2, 1))
  mask <- neuroim2::LogicalNeuroVol(
    mask_array, neuroim2::NeuroSpace(c(2, 2, 1))
  )
  latent <- fmrilatent::LatentNeuroVec(
    basis = scores,
    loadings = decoder,
    space = neuroim2::NeuroSpace(c(2, 2, 1, 3)),
    mask = mask,
    offset = rep(10, 4),
    meta = list(family = "toy_pca")
  )
  x <- basis_space_from_fmrilatent(latent, parent = parent)

  expect_s3_class(x, "basis_space")
  expect_identical(x$basis_type, "toy_pca")
  expect_identical(dim(basis_synthesis(x)), c(4L, 2L))
  expect_match(feature_ids(x), "^component-[0-9a-f]{12}-")
  expect_identical(x$provenance$source_class, "LatentNeuroVec")
  expect_false("offset" %in% names(x))
  expect_equal(
    vectorize_space(x, reconstruct_space(x, c(1.5, -0.25))),
    c(1.5, -0.25),
    tolerance = 1e-10
  )
})

test_that("composite_space owns ordered part-qualified feature identity", {
  x <- make_composite_space_fixture()

  expect_s3_class(x, "composite_space")
  expect_identical(
    composite_part_names(x),
    c("left_cortex", "right_cortex", "subcortical")
  )
  expect_length(composite_parts(x), 3L)
  expect_s3_class(composite_part(x, "left_cortex"), "surface_space")
  expect_identical(n_features(x), 8L)
  expect_identical(
    feature_ids(x),
    c(
      paste0("left_cortex::L-", 1:3),
      paste0("right_cortex::R-", 1:3),
      paste0("subcortical::voxel-", 1:2)
    )
  )
  expect_identical(
    native_shape(x),
    list(
      left_cortex = c(vertex = 3L),
      right_cortex = c(vertex = 3L),
      subcortical = c(2L, 1L, 1L)
    )
  )
  fd <- feature_data(x)
  expect_identical(fd$.feature_id, feature_ids(x))
  expect_identical(
    fd$.part,
    rep(
      c("left_cortex", "right_cortex", "subcortical"),
      c(3L, 3L, 2L)
    )
  )
  expect_identical(fd$.part_index, c(1:3, 1:3, 1:2))
  expect_identical(
    fd$.part_feature_id,
    c(
      paste0("L-", 1:3), paste0("R-", 1:3),
      paste0("voxel-", 1:2)
    )
  )
})

test_that("composite vectorization and reconstruction route native parts", {
  x <- make_composite_space_fixture()
  native <- list(
    subcortical = array(c(70, 80), dim = c(2, 1, 1)),
    right_cortex = c(40, 50, 60),
    left_cortex = c(10, 20, 30)
  )

  packed <- vectorize_space(x, native)
  expect_identical(packed, seq(10, 80, by = 10))
  restored <- reconstruct_space(x, packed)
  expect_s3_class(restored, "composite_map")
  expect_identical(names(restored$parts), composite_part_names(x))
  expect_equal(vectorize_space(x, restored), packed, tolerance = 0)
  expect_s3_class(restored$parts$left_cortex, "surface_map")
  expect_true(methods::is(restored$parts$subcortical, "NeuroVol"))

  expect_error(vectorize_space(x, native[-1L]), "exactly")
  expect_error(vectorize_space(x, c(native, unexpected = list(1))), "exactly")
})

test_that("composite restriction preserves arbitrary cross-part feature order", {
  x <- make_composite_space_fixture()
  index <- c(8L, 2L, 5L, 7L)
  y <- restrict_space(x, index)

  expect_identical(feature_ids(y), feature_ids(x)[index])
  expect_identical(
    composite_part_names(y),
    c("left_cortex", "right_cortex", "subcortical")
  )
  expect_identical(n_features(composite_part(y, "left_cortex")), 1L)
  expect_identical(n_features(composite_part(y, "right_cortex")), 1L)
  expect_identical(n_features(composite_part(y, "subcortical")), 2L)
  values <- c(8, 2, 5, 7)
  expect_equal(
    vectorize_space(y, reconstruct_space(y, values)),
    values,
    tolerance = 0
  )

  empty_part <- restrict_space(x, 4:6)
  expect_identical(composite_part_names(empty_part), "right_cortex")

  empty <- restrict_space(x, integer())
  expect_s3_class(empty, "composite_space")
  expect_identical(n_features(empty), 0L)
  expect_identical(feature_ids(empty), character())
  expect_identical(nrow(feature_data(empty)), 0L)
  expect_identical(dim(adjacency(empty)), c(0L, 0L))
  expect_identical(
    vectorize_space(empty, reconstruct_space(empty, numeric())),
    numeric()
  )
})

test_that("composite adjacency is block diagonal in routed feature order", {
  x <- make_composite_space_fixture()
  graph <- adjacency(x)

  expect_s4_class(graph, "sparseMatrix")
  expect_identical(dim(graph), c(8L, 8L))
  expect_true(all(graph == Matrix::t(graph)))
  expect_false(any(graph[1:3, 4:8]))
  expect_false(any(graph[4:6, c(1:3, 7:8)]))
  expect_true(graph[7L, 8L])

  index <- c(8L, 2L, 5L, 7L)
  restricted <- adjacency(restrict_space(x, index))
  expect_equal(as.matrix(restricted), as.matrix(graph[index, index]))
})

test_that("composite identity includes ordered part names spaces and routing", {
  x <- make_composite_space_fixture()
  expect_true(compatible_space(x, make_composite_space_fixture())$compatible)

  reordered <- composite_space(composite_parts(x)[c(2L, 1L, 3L)],
    composite_type = x$composite_type
  )
  expect_false(identical(space_digest(x), space_digest(reordered)))

  renamed <- composite_parts(x)
  names(renamed)[[1L]] <- "cortex_left"
  renamed <- composite_space(renamed, composite_type = x$composite_type)
  expect_false(identical(space_digest(x), space_digest(renamed)))

  routed <- restrict_space(x, c(8L, 2L, 5L, 7L))
  rerouted <- restrict_space(x, c(2L, 8L, 5L, 7L))
  expect_false(identical(space_digest(routed), space_digest(rerouted)))
})

test_that("composite contracts reject ambiguous parts and metadata", {
  x <- make_composite_space_fixture()
  parts <- composite_parts(x)

  expect_error(composite_space(list()), "at least one")
  expect_error(composite_space(unname(parts)), "named")
  duplicated <- parts[1:2]
  names(duplicated) <- c("cortex", "cortex")
  expect_error(composite_space(duplicated), "unique")
  bad_name <- parts[1]
  names(bad_name) <- "left::cortex"
  expect_error(composite_space(bad_name), "part names")
  expect_error(composite_space(list(bad = 1:3)), "feature_space")
  expect_error(
    composite_space(parts, metadata = list(loader = function() NULL)),
    "serializable"
  )

  expect_error(
    composite_space(
      parts,
      route = data.frame(part = "left_cortex", part_index = 1L)
    ),
    "every child feature"
  )
  bad_route <- data.frame(
    part = rep(names(parts), vapply(parts, n_features, integer(1))),
    part_index = unlist(lapply(parts, function(part) seq_len(n_features(part))))
  )
  bad_route$part_index[[1L]] <- 1.5
  expect_error(composite_space(parts, route = bad_route), "integers")

  nested <- composite_space(list(
    cortex = composite_space(parts[1:2], composite_type = "bilateral"),
    subcortical = parts$subcortical
  ))
  expect_identical(n_features(nested), n_features(x))
  expect_match(feature_ids(nested)[[1L]], "^cortex::left_cortex::")
  expect_true(all(c(".part", ".child.part") %in% names(feature_data(nested))))

  triply_nested <- composite_space(list(all = nested))
  expect_true(".child.child.part" %in% names(feature_data(triply_nested)))
  expect_identical(anyDuplicated(names(feature_data(triply_nested))), 0L)
})

test_that("composite spaces survive views and FDS round trips", {
  x <- make_composite_space_fixture()
  source <- counting_source(memory_source(matrix(seq_len(16L), nrow = 2L)))
  frame <- fmri_frame(
    assays = list(signal = source),
    observations = data.frame(.obs_id = c("o1", "o2")),
    features = feature_axis(feature_data(x), space = x)
  )
  view <- frame[, c(8L, 2L, 5L, 7L)]
  expect_s3_class(space(view), "composite_space")
  expect_identical(feature_ids(space(view)), feature_ids(x)[c(8L, 2L, 5L, 7L)])
  expect_equal(source_counts(source)$reads, 0)

  manifest <- fds_frame_manifest(frame)
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list(
      "assays/signal" = memory_source(matrix(seq_len(16L), nrow = 2L))
    )
  )
  expect_s3_class(space(restored), "composite_space")
  expect_identical(space_digest(space(restored)), space_digest(x))

  skip_if_not_installed("fmristore")
  skip_if_not("write_frame_h5" %in% getNamespaceExports("fmristore"))
  path <- tempfile(fileext = ".h5")
  on.exit(unlink(path), add = TRUE)
  write_frame(frame, path)
  reopened <- open_frame(path)
  expect_s3_class(space(reopened), "composite_space")
  expect_identical(space_digest(space(reopened)), space_digest(x))
})

test_that("composite neurosurf reconstruction delegates each surface part", {
  skip_if_not_installed("neurosurf")
  x <- make_composite_space_fixture()
  restored <- reconstruct_space(x, seq_len(n_features(x)), format = "neurosurf")

  expect_true(methods::is(restored$parts$left_cortex, "NeuroSurface"))
  expect_true(methods::is(restored$parts$right_cortex, "NeuroSurface"))
  expect_true(methods::is(restored$parts$subcortical, "NeuroVol"))
  expect_equal(vectorize_space(x, restored), seq_len(n_features(x)))

  left <- restored$parts$left_cortex
  geometry <- neurosurf::geometry(left)
  changed_vertices <- neurosurf::vertices(geometry)
  changed_vertices[1L, 1L] <- changed_vertices[1L, 1L] + 1
  changed_geometry <- neurosurf::SurfaceGeometry(
    changed_vertices,
    neurosurf::faces(geometry) - 1L,
    hemi = "lh",
    surf_to_world = neurosurf::surf_to_world(geometry)
  )
  changed <- neurosurf::NeuroSurface(
    changed_geometry,
    neurosurf::indices(left),
    neurosurf::values(left)
  )
  bad_map <- restored
  bad_map$parts$left_cortex <- changed
  expect_error(
    vectorize_space(x, bad_map),
    class = "fmridataset_error_space_mismatch"
  )

  named <- stats::setNames(seq_len(n_features(x)), feature_ids(x))
  named <- rev(named)
  expect_equal(
    vectorize_space(x, reconstruct_space(x, named)),
    seq_len(n_features(x))
  )
  expect_error(
    reconstruct_space(x, stats::setNames(
      seq_len(n_features(x)),
      rep("duplicate", n_features(x))
    )),
    "exactly once"
  )
})
