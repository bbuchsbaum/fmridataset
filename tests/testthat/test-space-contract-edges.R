# Two contract edges that both let contradictory or empty input through
# silently rather than being handled explicitly.

test_that("an emptied composite space rebuilds through its own constructor", {
  composite <- make_composite_space_fixture()
  empty <- restrict_space(composite, integer(0))

  expect_equal(n_features(empty), 0L)
  expect_equal(length(feature_ids(empty)), 0L)

  # paste() and data.frame() both recycle a scalar part name against a
  # zero-length index, so an emptied composite used to produce a "part::" key
  # and fail its own "every child feature exactly once" validation, and the
  # default route died on "arguments imply differing number of rows: 1, 0".
  expect_s3_class(
    composite_space(composite_parts(empty), route = empty$route),
    "composite_space"
  )
  expect_s3_class(composite_space(composite_parts(empty)), "composite_space")
})

test_that("an emptied composite keeps a stable digest and restricts again", {
  composite <- make_composite_space_fixture()

  expect_identical(
    space_digest(restrict_space(composite, integer(0))),
    space_digest(restrict_space(composite, integer(0)))
  )
  expect_equal(n_features(restrict_space(composite, c(1, 2))), 2L)
})

test_that("fmri_frame refuses a space that contradicts its feature axis", {
  ids <- c("f1", "f2", "f3")
  index <- index_space(3, ids = ids)
  axis <- feature_axis(tibble::tibble(.feature_id = ids), space = index)
  values <- matrix(as.double(1:9), 3, 3)
  observations <- data.frame(.obs_id = c("o1", "o2", "o3"))

  expect_error(
    fmri_frame(
      list(a = values),
      observations = observations,
      features = axis,
      space = volume_space(c(3, 1, 1), affine = diag(4), support = 1:3)
    ),
    class = "fmridataset_error_space_mismatch"
  )
})

test_that("a space agreeing with the feature axis is accepted", {
  ids <- c("f1", "f2", "f3")
  index <- index_space(3, ids = ids)
  axis <- feature_axis(tibble::tibble(.feature_id = ids), space = index)
  values <- matrix(as.double(1:9), 3, 3)
  observations <- data.frame(.obs_id = c("o1", "o2", "o3"))

  both <- fmri_frame(list(a = values),
    observations = observations,
    features = axis, space = index
  )
  axis_only <- fmri_frame(list(a = values),
    observations = observations,
    features = axis
  )
  space_only <- fmri_frame(list(a = values),
    observations = observations,
    space = index
  )

  expect_identical(space_digest(space(both)), space_digest(index))
  expect_identical(space_digest(space(axis_only)), space_digest(index))
  expect_identical(space_digest(space(space_only)), space_digest(index))
})

test_that("collect_spatial_maps rejects duplicated observations as documented", {
  fixture <- make_frame_fixture()
  frame <- fixture$frame

  expect_error(
    collect_spatial_maps(frame, observations = c(1L, 1L)),
    class = "fmridataset_error_alignment"
  )
})

# ----------------------------------------------------- source native space

.toy_nifti_source <- function() {
  dims <- c(3L, 2L, 2L)
  dir <- tempfile("nifti-native-")
  dir.create(dir)
  image <- file.path(dir, "toy.nii")
  mask <- file.path(dir, "mask.nii")
  values <- array(seq_len(prod(dims) * 2L) * 1.0, c(dims, 2L))
  neuroim2::write_vec(
    neuroim2::NeuroVec(values, neuroim2::NeuroSpace(c(dims, 2L))),
    image
  )
  neuroim2::write_vol(
    neuroim2::LogicalNeuroVol(array(TRUE, dims), neuroim2::NeuroSpace(dims)),
    mask
  )
  nifti_array_source(image, mask)
}

test_that("a NIfTI source rejects a frame volume_space it does not realize", {
  source <- .toy_nifti_source()
  native <- nifti_source_space(source)
  observations <- data.frame(.obs_id = c("a", "b"))
  build <- function(spatial) {
    fmri_frame(
      assays = list(signal = source), observations = observations,
      space = spatial, active_assay = "signal"
    )
  }

  reordered <- restrict_space(native, rev(seq_len(n_features(native))))
  expect_error(build(reordered), class = "fmridataset_error_space_mismatch")
  moved <- volume_space(native$dim,
    affine = native$affine * 2,
    support = native$support
  )
  expect_error(build(moved), class = "fmridataset_error_space_mismatch")

  # A labelled copy of the native space is the same domain, and the native
  # and reconstruct paths agree on it.
  x <- build(nifti_source_space(source, template = "toy"))
  expect_identical(execution_path(x, "spatial"), "native")
  expect_equal(
    as.numeric(collect_spatial_maps(x, 2L, path = "native")[[1L]]),
    as.numeric(collect_spatial_maps(x, 2L, path = "reconstruct")[[1L]])
  )

  # A non-volumetric space makes no voxel claim: it is accepted, but a native
  # read can no longer be routed through it.
  ids <- sprintf("f%02d", seq_len(n_features(native)))
  y <- build(index_space(n_features(native), ids = ids))
  expect_identical(execution_path(y, "spatial"), "reconstruct")
  expect_error(
    execution_path(y, "spatial", path = "native"),
    class = "fmridataset_error_backend_io"
  )
})

test_that("NIfTI source affines keep the header's axis orientation", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")
  source <- nifti_array_source(path, path)
  image <- suppressWarnings(neuroim2::read_vol(path))
  expect_equal(
    source$affine, unname(neuroim2::trans(neuroim2::space(image))),
    tolerance = 1e-7
  )
  expect_length(
    vectorize_space(nifti_source_space(source), image),
    source_shape(source)[[2L]]
  )
})

# ----------------------------------------------------- volume vectorization

test_that("volume vectorization checks shape and affine, not just length", {
  x <- volume_space(c(2, 2, 1), support = 1:4)
  good <- neuroim2::NeuroVol(array(1:4, c(2, 2, 1)), neuroim2::NeuroSpace(c(2, 2, 1)))
  expect_equal(vectorize_space(x, good), 1:4)
  expect_equal(vectorize_space(x, array(1:4, c(2, 2, 1))), 1:4)
  # A dimensionless vector is the documented full-volume column-major form.
  expect_equal(vectorize_space(x, 4:1), 4:1)

  moved <- neuroim2::NeuroVol(
    array(1:4, c(2, 2, 1)),
    neuroim2::NeuroSpace(c(2, 2, 1), spacing = c(3, 3, 3), origin = c(-50, 10, 5))
  )
  wrong_dim <- neuroim2::NeuroVol(array(1:4, c(4, 1, 1)), neuroim2::NeuroSpace(c(4, 1, 1)))
  for (bad in list(moved, wrong_dim, array(1:4, c(1, 4, 1)), 1:5)) {
    expect_error(vectorize_space(x, bad), class = "fmridataset_error_space_mismatch")
  }
})

# ----------------------------------------------------- basis restriction

test_that("basis restriction subsets a caller's encoder instead of replacing it", {
  parent <- volume_space(c(3, 2, 1), support = 1:6)
  decoder <- as.matrix(Matrix::sparseMatrix(
    i = 1:6, j = c(1, 1, 1, 2, 2, 2), x = 1, dims = c(6L, 2L)
  ))
  encoder <- matrix(0, 2, 6)
  encoder[1, 1] <- 1
  encoder[2, 4] <- 1
  x <- basis_space(parent, c("c1", "c2"), encoder = encoder, decoder = decoder)

  same <- restrict_space(x, 1:2)
  expect_true(same_space(x, same)$same)
  expect_identical(basis_analysis(same), encoder)

  native <- array(c(1, 0, 0, 5, 0, 0), c(3, 2, 1))
  second <- restrict_space(x, 2L)
  expect_identical(basis_analysis(second), encoder[2, , drop = FALSE])
  expect_equal(vectorize_space(second, native), 5)
  expect_equal(vectorize_space(restrict_space(x, c(2L, 1L)), native), c(5, 1))
})

# ----------------------------------------------------- parcel restriction

test_that("parcel restriction rebuilds the default decoder from its membership", {
  parent <- index_space(3, ids = c("v1", "v2", "v3"))
  membership <- Matrix::sparseMatrix(
    i = c(1, 2, 2, 3), j = c(1, 1, 2, 2), x = 1, dims = c(3L, 2L)
  )
  atlas <- list(id = "toy", n_parcels = 2L)
  x <- parcel_space(parent, c("A", "B"), membership, atlas = atlas)
  restricted <- restrict_space(x, 1L)
  fresh <- parcel_space(parent, "A", membership[, 1, drop = FALSE], atlas = atlas)

  expect_equal(unname(reconstruct_space(restricted, 10)), c(10, 10, NA))
  expect_true(same_space(restricted, fresh)$same)
  expect_equal(vectorize_space(restricted, reconstruct_space(restricted, 10)), 10)
  expect_identical(
    space_digest(unserialize(serialize(restricted, NULL))),
    space_digest(restricted)
  )

  # A supplied decoder is the caller's operator and is subset, not rebuilt.
  custom <- Matrix::Matrix(c(1, 0.25, 0, 0, 0.75, 1), 3, 2, sparse = TRUE)
  y <- parcel_space(parent, c("A", "B"), membership, atlas = atlas, decoder = custom)
  expect_equal(unname(reconstruct_space(restrict_space(y, 1L), 8)), c(8, 2, NA))

  # A space serialized before the default flag existed is still classified.
  legacy <- x
  legacy$decoder_default <- NULL
  expect_true(same_space(restrict_space(legacy, 1L), fresh)$same)
})

# ----------------------------------------------------- named vectors

test_that("named vectors must cover the feature IDs exactly in every space", {
  x <- index_space(3, ids = c("a", "b", "c"))
  expect_equal(unname(reconstruct_space(x, c(c = 3, b = 2, a = 1))), c(1, 2, 3))
  expect_equal(vectorize_space(x, c(c = 3, b = 2, a = 1)), c(1, 2, 3))
  expect_equal(reconstruct_space(x, c(3, 2, 1)), c(3, 2, 1))

  parent <- volume_space(c(3, 2, 1), support = 1:6)
  membership <- Matrix::sparseMatrix(
    i = 1:6, j = c(1, 1, 1, 2, 2, 2), x = 1, dims = c(6L, 2L)
  )
  parcels <- parcel_space(parent, c("p1", "p2"), membership, atlas = "toy")
  basis <- basis_space(parent, c("c1", "c2"),
    encoder = t(as.matrix(membership)) / 3, decoder = as.matrix(membership)
  )
  surface <- surface_space(c("v1", "v2", "v3"), rep("left", 3))
  volume <- volume_space(c(2, 1, 1), support = 1:2)

  expect_equal(
    as.numeric(reconstruct_space(basis, c(c2 = 2, c1 = 1))),
    c(1, 1, 1, 2, 2, 2)
  )
  expect_equal(
    as.numeric(reconstruct_space(parcels, c("toy:p2" = 2, "toy:p1" = 1))),
    c(1, 1, 1, 2, 2, 2)
  )
  expect_equal(vectorize_space(surface, c(v3 = 3, v1 = 1, v2 = 2)), c(1, 2, 3))

  bad <- list(
    list(x, c(a = 1, b = 2)),
    list(x, c(a = 1, b = 2, z = 3)),
    list(x, stats::setNames(c(1, 2, 3), c("a", "a", "b"))),
    list(basis, c(c1 = 1)),
    list(parcels, c("toy:p1" = 1, other = 2)),
    list(volume, c("voxel-1" = 1, "voxel-9" = 2)),
    list(surface, c(v1 = 1, v2 = 2))
  )
  for (case in bad) {
    expect_error(
      reconstruct_space(case[[1L]], case[[2L]]),
      class = "fmridataset_error_space_mismatch"
    )
  }
  expect_error(vectorize_space(x, c(a = 1, b = 2, z = 3)),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(vectorize_space(surface, c(v1 = 1, v2 = 2)),
    class = "fmridataset_error_space_mismatch"
  )
})

# ----------------------------------------------------- restriction selectors

test_that("every built-in space restricts by one selection law", {
  parent <- volume_space(c(2, 2, 1), support = 1:4)
  membership <- Matrix::sparseMatrix(i = 1:4, j = c(1, 2, 3, 3), x = 1, dims = c(4L, 3L))
  spaces <- list(
    index = index_space(3, ids = c("a", "b", "c")),
    volume = volume_space(c(3, 1, 1), support = 1:3),
    surface = surface_space(c("v1", "v2", "v3"), rep("left", 3)),
    parcel = parcel_space(parent, c("p1", "p2", "p3"), membership, atlas = "toy"),
    basis = basis_space(index_space(3, ids = c("x", "y", "z")), c("c1", "c2", "c3"),
      encoder = diag(3), decoder = diag(3)
    ),
    composite = composite_space(list(
      l = index_space(1, ids = "l1"), r = index_space(2, ids = c("r1", "r2"))
    ))
  )
  for (name in names(spaces)) {
    x <- spaces[[name]]
    ids <- feature_ids(x)
    expect_identical(feature_ids(restrict_space(x, c(TRUE, FALSE, TRUE))), ids[c(1, 3)], info = name)
    expect_identical(feature_ids(restrict_space(x, -1)), ids[2:3], info = name)
    expect_identical(feature_ids(restrict_space(x, c(3, 1))), ids[c(3, 1)], info = name)
    expect_identical(feature_ids(restrict_space(x, ids[c(3, 1)])), ids[c(3, 1)], info = name)
    expect_identical(n_features(restrict_space(x, integer())), 0L, info = name)
    for (bad in list(5, c(1, 1), c(TRUE, FALSE), "missing", c(-1, 2), NA_integer_)) {
      expect_error(restrict_space(x, bad), class = "fmridataset_error_alignment", info = name)
    }
  }
})

# ----------------------------------------------------- basis from decoder

test_that("basis_space_from_decoder handles wide and empty dictionaries", {
  wide <- matrix(c(1, 0, 0, 1, 1, 1), 2, 3)
  expect_error(
    basis_space_from_decoder(volume_space(c(2, 1, 1)), c("a", "b", "c"), wide),
    class = "fmridataset_error_space_mismatch"
  )
  parent <- volume_space(c(3, 2, 1), support = 1:6)
  empty <- basis_space_from_decoder(parent, character(), matrix(0, 6, 0))
  expect_identical(n_features(empty), 0L)
  expect_identical(dim(basis_analysis(empty)), c(0L, 6L))
})

# ----------------------------------------------------- accessor errors

test_that("feature-space accessors raise structured errors", {
  x <- index_space(2, ids = c("a", "b"))
  for (accessor in list(
    parent_space, parcel_membership, parcel_aggregation,
    basis_analysis, basis_synthesis, basis_projection_info, composite_parts
  )) {
    expect_error(accessor(x), class = "fmridataset_error_space_mismatch")
  }
})
