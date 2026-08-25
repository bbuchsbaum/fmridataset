# index_space() defaulted `namespace` to uuid::UUIDgenerate() unconditionally
# and folded it into space_digest(). Two spaces declared with the SAME explicit
# feature IDs therefore carried different digests and compared as incompatible,
# which blocked bind_observations() on any two independently-built frames --
# with a diagnostic that said the spaces "differ in type, digest, or IDs"
# although the type and IDs were identical.
#
# The namespace still disambiguates spaces that legitimately reuse ID strings
# (parcel "1" of two different atlases), but only when the caller asks for it.

test_that("identical explicit feature IDs give compatible spaces", {
  a <- index_space(3, ids = c("x", "y", "z"))
  b <- index_space(3, ids = c("x", "y", "z"))

  expect_identical(feature_ids(a), feature_ids(b))
  expect_identical(space_digest(a), space_digest(b))
  expect_true(compatible_space(a, b)$compatible)
  expect_invisible(assert_compatible_space(a, b))
})

test_that("an explicit namespace still separates spaces that reuse IDs", {
  atlas_a <- index_space(2, ids = c("1", "2"), namespace = "atlasA")
  atlas_b <- index_space(2, ids = c("1", "2"), namespace = "atlasB")
  atlas_a2 <- index_space(2, ids = c("1", "2"), namespace = "atlasA")

  expect_false(compatible_space(atlas_a, atlas_b)$compatible)
  expect_true(compatible_space(atlas_a, atlas_a2)$compatible)
})

test_that("a namespaced space is not compatible with an unnamespaced one", {
  expect_false(
    compatible_space(
      index_space(2, ids = c("1", "2"), namespace = "atlasA"),
      index_space(2, ids = c("1", "2"))
    )$compatible
  )
})

test_that("generated IDs remain unique across independently-built spaces", {
  a <- index_space(3)
  b <- index_space(3)

  expect_false(identical(feature_ids(a), feature_ids(b)))
  expect_false(compatible_space(a, b)$compatible)
  # The namespace is carried inside the generated IDs, so identity survives
  # without the digest needing a separate copy of it.
  expect_match(feature_ids(a)[1], "^feature-")
})

test_that("restriction preserves compatibility", {
  a <- index_space(4, ids = c("w", "x", "y", "z"))
  b <- index_space(4, ids = c("w", "x", "y", "z"))

  expect_true(
    compatible_space(restrict_space(a, c(1, 3)), restrict_space(b, c(1, 3)))$compatible
  )
  # A different restriction is a different space.
  expect_false(
    compatible_space(restrict_space(a, c(1, 3)), restrict_space(b, c(2, 4)))$compatible
  )
})

test_that("independently-built frames over the same feature IDs bind", {
  make <- function(ids) {
    fmri_frame(
      list(a = matrix(as.double(seq_len(2 * length(ids))), length(ids), 2)),
      observations = data.frame(.obs_id = ids),
      space = index_space(2, ids = c("f1", "f2"))
    )
  }

  bound <- bind_observations(make(c("o1", "o2")), make(c("o3", "o4")))
  expect_equal(observation_ids(bound), c("o1", "o2", "o3", "o4"))
  expect_equal(feature_ids(bound), c("f1", "f2"))
})

test_that("frames over differently-namespaced spaces still refuse to bind", {
  make <- function(ids, ns) {
    fmri_frame(
      list(a = matrix(as.double(seq_len(2 * length(ids))), length(ids), 2)),
      observations = data.frame(.obs_id = ids),
      space = index_space(2, ids = c("f1", "f2"), namespace = ns)
    )
  }

  expect_error(
    bind_observations(make(c("o1", "o2"), "A"), make(c("o3", "o4"), "B")),
    class = "fmridataset_error_space_mismatch"
  )
})
