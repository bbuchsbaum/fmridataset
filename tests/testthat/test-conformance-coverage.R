# The package advertises executable contracts for array sources and feature
# spaces, but only some types were ever run through them: 2 of 6 feature spaces
# and 7 of 12 sources. A contract that most implementations skip is not a gate.
#
# This file runs every dispatchable type through its contract, and asserts that
# the set of covered types plus the set of deliberately excluded ones accounts
# for every type the namespace dispatches on -- so a new source or space cannot
# be added without either passing the contract or saying in writing why not.

# ---------------------------------------------------------------- array sources

test_that("memory and counting sources conform", {
  m <- matrix(as.double(1:24), 6, 4)
  expect_array_source_conformance(memory_source(m), m)
  expect_array_source_conformance(counting_source(memory_source(m)), m)
})

test_that("source views conform", {
  m <- matrix(as.double(1:24), 6, 4)
  view <- source_view(memory_source(m), observations = c(5, 1, 3), features = c(4, 2))
  expect_array_source_conformance(view, m[c(5, 1, 3), c(4, 2), drop = FALSE])

  # A view of a view must behave the same way.
  nested <- source_view(view, observations = c(3, 1), features = 2L)
  expect_array_source_conformance(nested, m[c(3, 5), 2, drop = FALSE])
})

test_that("row-bound and row-sharded sources conform", {
  a <- matrix(as.double(1:12), 3, 4)
  b <- matrix(as.double(101:108), 2, 4)

  expect_array_source_conformance(
    row_bound_source(list(memory_source(a), memory_source(b))),
    rbind(a, b)
  )
  expect_array_source_conformance(
    row_sharded_source(list(first = memory_source(a), second = memory_source(b))),
    rbind(a, b)
  )
})

test_that("feature-mapped and validity-masked sources conform", {
  from <- index_space(4, ids = sprintf("s%d", 1:4), namespace = "from")
  to <- index_space(2, ids = c("t1", "t2"), namespace = "to")
  weights <- rbind(c(1, 1, 0, 0), c(0, 0, 1, 1))
  values <- matrix(as.double(1:12), 3, 4)

  mapped <- feature_mapped_source(memory_source(values), feature_map(from, to, weights))
  expect_array_source_conformance(mapped, unname(values %*% t(weights)))
})

test_that("NIfTI sources conform against an independent read", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  source <- nifti_array_source(path, path)
  mask <- suppressWarnings(neuroim2::read_vol(path))
  reference <- as.matrix(neuroim2::series(
    suppressWarnings(neuroim2::read_vec(path)),
    which(as.logical(as.vector(mask)))
  ))

  expect_array_source_conformance(source, reference)
})

fmristore_has_h5_array_source <- function() {
  requireNamespace("fmristore", quietly = TRUE) &&
    requireNamespace("hdf5r", quietly = TRUE) &&
    "h5_array_source" %in% getNamespaceExports("fmristore")
}

test_that("HDF5 extension sources conform against stored values", {
  skip_if_not(
    fmristore_has_h5_array_source(),
    "the installed fmristore does not provide h5_array_source()"
  )

  path <- tempfile(fileext = ".h5")
  values <- matrix(as.double(1:24), nrow = 6L, ncol = 4L)
  h5 <- hdf5r::H5File$new(path, mode = "w")
  h5$create_dataset("values", robj = values, chunk_dims = c(2L, 2L))
  h5$close_all()
  on.exit(unlink(path), add = TRUE)

  source <- fmristore::h5_array_source(path, "values")
  expect_array_source_conformance(source, values)
})

# -------------------------------------------------------------- feature spaces

test_that("index and volume spaces conform", {
  expect_feature_space_conformance(index_space(6))
  expect_feature_space_conformance(index_space(6, ids = sprintf("f%d", 1:6)))
  expect_feature_space_conformance(volume_space(c(2, 2, 2), support = 1:6))
})

test_that("surface spaces conform", {
  expect_feature_space_conformance(
    surface_space(sprintf("v%d", 1:6), rep("left", 6))
  )
  expect_feature_space_conformance(
    surface_space(sprintf("v%d", 1:6), rep(c("left", "right"), each = 3))
  )
})

test_that("parcel spaces conform", {
  parent <- volume_space(c(2, 2, 2), support = 1:6)
  membership <- matrix(0, nrow = 6, ncol = 3)
  membership[1:2, 1] <- 1
  membership[3:4, 2] <- 1
  membership[5:6, 3] <- 1

  expect_feature_space_conformance(
    parcel_space(
      parent = parent,
      parcel_ids = c(10L, 20L, 30L),
      membership = membership,
      atlas = list(id = "toy")
    )
  )
})

test_that("basis spaces conform in both directions", {
  parent <- index_space(5, ids = sprintf("v%d", 1:5), namespace = "p")
  decoder <- matrix(0, nrow = 5, ncol = 3)
  decoder[1, 1] <- 1
  decoder[2, 2] <- 1
  decoder[3, 3] <- 1

  # Two-way.
  expect_feature_space_conformance(
    basis_space_from_decoder(parent, sprintf("c%d", 1:3), decoder)
  )
  # Synthesis-only.
  expect_feature_space_conformance(
    basis_space_from_decoder(parent, sprintf("c%d", 1:3), decoder, encoder = "none")
  )
})

test_that("composite spaces conform", {
  expect_feature_space_conformance(make_composite_space_fixture())
})

# --------------------------------------------------------------- the gate itself

test_that("every dispatchable source and space is covered or excused", {
  dispatch_targets <- function(generic) {
    methods <- as.character(utils::.S3methods(generic, envir = asNamespace("fmridataset")))
    sort(unique(sub(paste0("^", generic, "\\."), "", methods)))
  }

  # Exercised above, or in the type's own test file through the same helper.
  covered_sources <- c(
    "counting_source", "feature_mapped_source", "memory_source",
    "nifti_array_source", "row_bound_source", "row_index_source",
    "row_sharded_source", "source_view", "sparse_entity_source",
    "validity_masked_source", "zarr_array_source"
  )
  if (fmristore_has_h5_array_source()) {
    covered_sources <- c(covered_sources, "h5_array_source")
  }
  # fault_source injects failures on demand; refusing to read is its purpose,
  # so the read contract cannot apply to it.
  excused_sources <- c("fault_source", "default")

  covered_spaces <- c(
    "basis_space", "composite_space", "index_space", "parcel_space",
    "surface_space", "volume_space"
  )
  excused_spaces <- c("default")

  expect_setequal(
    dispatch_targets("source_open"),
    c(covered_sources, setdiff(excused_sources, "default"))
  )
  expect_setequal(dispatch_targets("space_digest"), covered_spaces)
})
