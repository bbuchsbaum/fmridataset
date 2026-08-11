.nifti_source_fixture <- function() {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  testthat::skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")
  mask <- suppressWarnings(neuroim2::read_vol(path))
  support <- which(as.logical(as.vector(mask)))
  list(path = path, mask = mask, support = support)
}

test_that("NIfTI sources expose serializable pushdown contracts", {
  fixture <- .nifti_source_fixture()
  source <- nifti_array_source(c(fixture$path, fixture$path), fixture$path)

  expect_invisible(validate_array_source(source))
  expect_identical(source_shape(source)[[1L]], 8L)
  expect_identical(source_shape(source)[[2L]], length(fixture$support))
  expect_identical(source_dtype(source), "float32")
  expect_true("native_read" %in% source_capabilities(source))
  expect_false(contains_runtime_state(source))
  restored <- unserialize(serialize(source, NULL))
  expect_identical(source_fingerprint(restored), source_fingerprint(source))

  spatial <- nifti_source_space(source, template = "fixture")
  expect_identical(feature_ids(spatial), paste0("voxel-", fixture$support))
  expect_identical(n_features(spatial), length(fixture$support))
})

test_that("NIfTI sources push observation and packed-feature selections", {
  fixture <- .nifti_source_fixture()
  source <- nifti_array_source(c(fixture$path, fixture$path), fixture$path)
  observations <- c(8L, 1L, 5L, 1L)
  features <- c(3L, 1L, 3L)

  full <- suppressWarnings(neuroim2::read_vec(
    c(fixture$path, fixture$path),
    mask = fixture$mask,
    mode = "normal"
  ))
  reference <- neuroim2::series(
    full,
    fixture$support[features],
    drop = FALSE
  )[observations, , drop = FALSE]
  calls <- list()
  with_mocked_bindings(
    .nifti_read_vec = function(path, indices = NULL, mask = NULL, mode = "normal") {
      calls[[length(calls) + 1L]] <<- list(
        path = path,
        indices = indices,
        active_features = sum(as.logical(as.vector(mask)))
      )
      neuroim2::read_vec(path, indices = indices, mask = mask, mode = mode)
    },
    .package = "fmridataset",
    {
      expect_equal(
        source_read(source, observations, features),
        reference,
        tolerance = 0
      )
    }
  )
  expect_length(calls, 2L)
  expect_identical(lapply(calls, `[[`, "indices"), list(c(4L, 1L), c(1L, 1L)))
  expect_identical(vapply(calls, `[[`, integer(1), "active_features"), c(2L, 2L))

  plan <- as_delarr(source)
  restored <- unserialize(serialize(plan, NULL))
  expect_equal(
    delarr::collect(restored[observations, features]),
    reference,
    tolerance = 0
  )
})

test_that("NIfTI native reads preserve requested observation order", {
  fixture <- .nifti_source_fixture()
  source <- nifti_array_source(c(fixture$path, fixture$path), fixture$path)
  selected <- c(8L, 1L, 5L)
  native <- source_read_native(source, selected)

  expect_s4_class(native, "NeuroVec")
  expect_identical(as.integer(dim(native)[[4L]]), length(selected))
  native_matrix <- neuroim2::series(
    native,
    fixture$support[1:3],
    drop = FALSE
  )
  expect_equal(
    native_matrix,
    source_read(source, selected, 1:3),
    tolerance = 0
  )

  restricted <- source_view(source, features = 1:3)
  expect_false("native_read" %in% source_capabilities(restricted))
  expect_error(source_read_native(restricted, 1L), "feature-restricted")
})

test_that("NIfTI sources detect stale files before numerical reads", {
  fixture <- .nifti_source_fixture()
  parent <- tempfile("nifti-source-")
  dir.create(parent)
  on.exit(unlink(parent, recursive = TRUE), add = TRUE)
  image <- file.path(parent, "image.nii")
  mask <- file.path(parent, "mask.nii")
  expect_true(file.copy(fixture$path, image))
  expect_true(file.copy(fixture$path, mask))
  source <- nifti_array_source(image, mask)

  Sys.setFileTime(image, Sys.time() + 2)
  expect_error(
    source_read(source, 1L, 1L),
    "changed after the descriptor"
  )
})

test_that("NIfTI source validates spatial agreement and selectors", {
  fixture <- .nifti_source_fixture()
  source <- nifti_array_source(fixture$path, fixture$path)
  incompatible <- volume_space(c(2, 2, 2), support = 1:3)

  expect_error(
    nifti_array_source(fixture$path, incompatible),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(
    nifti_array_source(fixture$path, fixture$path, chunks = 1L),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    nifti_array_source(fixture$path, fixture$path, chunks = c(1, NA)),
    class = "fmridataset_error_source_contract"
  )
  expect_identical(dim(source_read(source, integer(), 1:2)), c(0L, 2L))
  expect_identical(dim(source_read(source, 1:2, integer())), c(2L, 0L))
  expect_error(source_read(source, 9L, 1L), class = "fmridataset_error_alignment")
})
