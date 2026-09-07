# A feature-mapped read used to depend on the SHAPE of the request. Source
# columns that are zero for every requested target are pruned before the
# product, so which columns entered the arithmetic varied with the request --
# and because `0 * NA` is `NA`, a non-finite source value leaked into targets
# that carry no weight on it.
#
# t1 = a, t2 = b + c, source row = (NA, 10, 100):
#   source_read(fs)             gave (NA,  NA)
#   source_read(fs, features=2) gave      110
#
# A zero weight means the source feature is not part of the target, so the
# answer for a target must depend only on that target's own operator row.

map_fixture <- function(values, weights = rbind(c(1, 0, 0), c(0, 1, 1)),
                        rule = "linear") {
  from <- index_space(3, ids = c("a", "b", "c"), namespace = "nf")
  to <- index_space(nrow(weights),
    ids = paste0("t", seq_len(nrow(weights))),
    namespace = "nt"
  )
  feature_mapped_source(
    memory_source(values),
    feature_map(from, to, weights),
    rule = rule
  )
}

test_that("a target's value does not depend on which targets were requested", {
  fs <- map_fixture(matrix(c(NA, 10, 100), nrow = 1))

  full <- source_read(fs)
  expect_equal(source_read(fs, features = 1)[1, 1], full[1, 1])
  expect_equal(source_read(fs, features = 2)[1, 1], full[1, 2])

  # t2 weights only b and c, neither of which is NA.
  expect_equal(full[1, 2], 110)
  # t1 weights a, which is NA, so it stays NA.
  expect_true(is.na(full[1, 1]))
})

test_that("the same invariance holds through the frame API", {
  fs <- map_fixture(matrix(c(NA, 10, 100), nrow = 1))
  to <- index_space(2, ids = c("t1", "t2"), namespace = "nt")
  f <- fmri_frame(
    list(a = fs),
    observations = data.frame(.obs_id = "o1"),
    space = to
  )

  expect_equal(collect_assay(f)[1, 2], 110)
  expect_equal(collect_assay(f[, "t2"])[1, 1], 110)
  expect_true(is.na(collect_assay(f[, "t1"])[1, 1]))
})

test_that("block-wise results do not depend on block size", {
  fs <- map_fixture(matrix(c(NA, 10, 100), nrow = 1))
  to <- index_space(2, ids = c("t1", "t2"), namespace = "nt")
  f <- fmri_frame(
    list(a = fs),
    observations = data.frame(.obs_id = "o1"),
    space = to
  )

  one <- unlist(block_apply(f, function(v, ids) v, block_size = 1))
  two <- unlist(block_apply(f, function(v, ids) v, block_size = 2))
  expect_equal(one, two)
  expect_equal(unname(two[2]), 110)
})

test_that("finite data is unchanged and matches dense matrix algebra exactly", {
  weights <- rbind(c(1, 0, 0), c(0, 1, 1), c(0.5, 0.25, 0))
  x <- matrix(c(1, 10, 100, 2, 20, 200), nrow = 2, byrow = TRUE)
  fs <- map_fixture(x, weights)

  expect_identical(
    unname(as.matrix(source_read(fs))),
    unname(x %*% t(weights))
  )
})

test_that("non-finite values still propagate where they are genuinely weighted", {
  weights <- rbind(c(1, 0, 0), c(0, 1, 1))

  # NA in b: t1 does not weight b, t2 does.
  fs <- map_fixture(matrix(c(1, NA, 100), nrow = 1), weights)
  out <- source_read(fs)
  expect_equal(out[1, 1], 1)
  expect_true(is.na(out[1, 2]))

  # NaN behaves like NA.
  fs_nan <- map_fixture(matrix(c(1, NaN, 100), nrow = 1), weights)
  expect_true(is.na(source_read(fs_nan)[1, 2]))

  # An infinite contributing value is reported as non-finite, not as a
  # spurious 0 * Inf = NaN in an unrelated target.
  fs_inf <- map_fixture(matrix(c(Inf, 10, 100), nrow = 1), weights)
  out_inf <- source_read(fs_inf)
  expect_true(is.na(out_inf[1, 1]))
  expect_equal(out_inf[1, 2], 110)
})

test_that("request invariance holds for the independent_variance rule", {
  fs <- map_fixture(
    matrix(c(NA, 4, 9), nrow = 1),
    rule = "independent_variance"
  )

  full <- source_read(fs)
  expect_equal(source_read(fs, features = 2)[1, 1], full[1, 2])
  # squared weights: t2 = 1^2 * 4 + 1^2 * 9
  expect_equal(full[1, 2], 13)
  expect_true(is.na(full[1, 1]))
})

test_that("request invariance holds for a sparse operator", {
  weights <- Matrix::sparseMatrix(
    i = c(1, 2, 2), j = c(1, 2, 3), x = c(1, 1, 1), dims = c(2, 3)
  )
  fs <- map_fixture(matrix(c(NA, 10, 100), nrow = 1), weights)

  full <- source_read(fs)
  expect_equal(full[1, 2], 110)
  expect_equal(source_read(fs, features = 2)[1, 1], 110)
  expect_true(is.na(full[1, 1]))
})

test_that("multiple observations keep per-row independence", {
  weights <- rbind(c(1, 0, 0), c(0, 1, 1))
  x <- matrix(c(
    NA, 10, 100, # row 1: t1 NA, t2 110
    1, 2, 3 # row 2: t1 1,  t2 5
  ), nrow = 2, byrow = TRUE)
  fs <- map_fixture(x, weights)

  out <- source_read(fs)
  expect_true(is.na(out[1, 1]))
  expect_equal(out[1, 2], 110)
  expect_equal(out[2, 1], 1)
  expect_equal(out[2, 2], 5)

  # Row 2 is fully finite, so requesting it alone must agree.
  expect_equal(unname(source_read(fs, observations = 2)), matrix(c(1, 5), 1, 2))
})
