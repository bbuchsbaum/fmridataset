# basis_space() required an encoder, so a basis could only exist if parent data
# could be PROJECTED into its components. That excluded exactly the case
# latent_dataset serves: spatial loadings are a synthesis dictionary, and ICA or
# dictionary-learning fits are routinely rank-deficient or non-orthogonal, so no
# exact left inverse exists. Such a basis can still reconstruct voxels from
# scores, which is all that path ever needed.
#
# A basis now carries an analysis operator, a synthesis operator, or both, and
# refuses only the case where it has neither.

basis_parent <- function() index_space(5, ids = sprintf("v%d", 1:5), namespace = "p")
full_rank_decoder <- function() matrix(c(1, 0, 0, 0, 0, 0, 1, 0, 0, 0), nrow = 5, ncol = 2)
rank_deficient_decoder <- function() matrix(c(1, 1, 0, 0, 0, 2, 2, 0, 0, 0), nrow = 5, ncol = 2)

test_that("a full-rank dictionary still yields a validated two-way basis", {
  basis <- basis_space_from_decoder(basis_parent(), c("c1", "c2"), full_rank_decoder())

  expect_false(is.null(basis_analysis(basis)))
  expect_false(is.null(basis_synthesis(basis)))
  expect_true(basis$projection$left_inverse_validated)
})

test_that("a rank-deficient dictionary is refused with a route forward", {
  err <- tryCatch(
    basis_space_from_decoder(basis_parent(), c("c1", "c2"), rank_deficient_decoder()),
    error = function(e) e
  )

  expect_s3_class(err, "fmridataset_error_space_mismatch")
  expect_match(conditionMessage(err), "full column rank")
  expect_match(conditionMessage(err), "encoder = \"none\"")
})

test_that("a synthesis-only basis reconstructs but does not project", {
  basis <- basis_space_from_decoder(
    basis_parent(), c("c1", "c2"), rank_deficient_decoder(),
    encoder = "none"
  )

  expect_null(basis_analysis(basis))
  expect_false(is.null(basis_synthesis(basis)))
  expect_false(basis$projection$left_inverse_validated)
  expect_equal(n_features(basis), 2L)
  expect_equal(feature_ids(basis), c("c1", "c2"))

  # Synthesis is exact: reconstruct is decoder %*% scores.
  expect_equal(
    as.numeric(reconstruct_space(basis, c(1, 2))),
    as.numeric(rank_deficient_decoder() %*% c(1, 2))
  )

  err <- tryCatch(vectorize_space(basis, rnorm(5)), error = function(e) e)
  expect_s3_class(err, "fmridataset_error_space_mismatch")
  expect_match(conditionMessage(err), "no encoder")
  expect_match(conditionMessage(err), "reconstruct_space")
})

test_that("an analysis-only basis projects but does not reconstruct", {
  basis <- basis_space(basis_parent(), c("c1", "c2"), encoder = t(full_rank_decoder()))

  expect_false(is.null(basis_analysis(basis)))
  expect_null(basis_synthesis(basis))
  expect_false(basis$projection$left_inverse_validated)
  expect_equal(vectorize_space(basis, c(1, 2, 3, 4, 5)), c(1, 2))
  expect_error(reconstruct_space(basis, c(1, 2)), "no decoder")
})

test_that("a basis with neither operator is refused", {
  err <- tryCatch(
    basis_space(basis_parent(), c("c1", "c2"), encoder = NULL, decoder = NULL),
    error = function(e) e
  )
  expect_s3_class(err, "fmridataset_error_space_mismatch")
  expect_match(conditionMessage(err), "encoder, a decoder, or both")
})

test_that("a synthesis-only basis has a stable digest and restricts", {
  make <- function() {
    basis_space_from_decoder(
      basis_parent(), c("c1", "c2"), rank_deficient_decoder(),
      encoder = "none"
    )
  }
  expect_identical(space_digest(make()), space_digest(make()))

  restricted <- restrict_space(make(), 1L)
  expect_equal(n_features(restricted), 1L)
  expect_equal(feature_ids(restricted), "c1")
  # Restriction must not manufacture an encoder the basis never had.
  expect_null(basis_analysis(restricted))
  expect_false(is.null(basis_synthesis(restricted)))
})

test_that("a two-way basis still recomputes its encoder on restriction", {
  basis <- basis_space_from_decoder(basis_parent(), c("c1", "c2"), full_rank_decoder())
  restricted <- restrict_space(basis, 1L)

  expect_false(is.null(basis_analysis(restricted)))
  expect_true(restricted$projection$left_inverse_validated)
})

test_that("a synthesis-only basis works as a frame's feature space", {
  basis <- basis_space_from_decoder(
    basis_parent(), c("c1", "c2"), rank_deficient_decoder(),
    encoder = "none"
  )
  scores <- matrix(as.double(1:8), 4, 2)
  frame <- fmri_frame(
    list(latent = scores),
    observations = data.frame(.obs_id = sprintf("t%d", 1:4)),
    space = basis
  )

  expect_equal(feature_ids(frame), c("c1", "c2"))
  expect_equal(collect_assay(frame), scores)
  expect_equal(feature_ids(frame[, "c2"]), "c2")
})
