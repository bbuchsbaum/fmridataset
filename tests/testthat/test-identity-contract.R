test_that("canonicalization v1 is explicit and R-only", {
  contract <- canonicalization_contract()
  expect_identical(contract$id, "org.fmridataset.r-canonical/v1")
  expect_identical(contract$version, 1L)
  expect_identical(contract$algorithm, "sha256")
  expect_identical(contract$portability, "R-only")
})

test_that("identity descriptors preserve distinct domains", {
  frame <- make_frame_fixture()$frame
  identities <- list(
    semantic = identity_descriptor(frame),
    schema = identity_descriptor(frame_schema(frame)),
    space = identity_descriptor(space(frame)),
    source = identity_descriptor(assay(frame)$source),
    content = identity_descriptor(frame, domain = "content", content_digest = "receipt-1")
  )
  expect_identical(
    unname(vapply(identities, `[[`, character(1), "domain")),
    names(identities)
  )
  expect_true(all(vapply(identities, inherits, logical(1), "fmri_identity")))
  expect_identical(identities$content$digest, "receipt-1")
  expect_error(identity_descriptor(frame, domain = "content"),
               class = "fmridataset_error_identity")
})

test_that("semantic and schema identities ignore source wrappers", {
  frame <- make_frame_fixture()$frame
  wrapped <- frame
  wrapped$assays <- aligned_assay_set(
    lapply(assays(frame), function(value) counting_source(value$source)),
    observation_axis(frame), feature_axis(frame)
  )
  expect_identical(identity_descriptor(wrapped)$digest,
                   identity_descriptor(frame)$digest)
  expect_identical(frame_schema_digest(wrapped), frame_schema_digest(frame))
})

test_that("same_space is exact and compatibility names are migration aliases", {
  x <- volume_space(c(2L, 2L, 2L), support = 1:4, template = "exact")
  y <- volume_space(c(2L, 2L, 2L), support = 1:4, template = "exact")
  same_shape <- volume_space(c(2L, 2L, 2L), support = 5:8, template = "exact")
  expect_true(same_space(x, y)$same)
  expect_identical(compatible_space(x, y), same_space(x, y))
  expect_false(same_space(x, same_shape)$same)
  expect_error(assert_same_space(x, same_shape),
               class = "fmridataset_error_space_mismatch")
})

test_that("identity inspection reads zero numerical bytes", {
  frame <- make_frame_fixture(instrument = TRUE)$frame
  identity_descriptor(frame)
  identity_descriptor(frame_schema(frame))
  identity_descriptor(space(frame))
  lapply(assays(frame), function(value) identity_descriptor(value$source))
  expect_true(all(vapply(
    assays(frame), function(value) source_counts(value$source)$bytes, numeric(1)
  ) == 0))
})
