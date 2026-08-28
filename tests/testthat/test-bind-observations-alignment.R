# bind_observations() row-binds axis-block data positionally. Nothing used to
# check that the two frames agreed on what their block columns MEANT, so a
# frame whose motion regressors happened to be ordered (rotation, translation)
# had its values filed under (translation, rotation) -- silently, with the
# first frame's labels applied to every row.
#
# The same shape of defect applied to feature metadata, tables, and frame
# metadata: the bound frame kept the first frame's copy without checking the
# others agreed.

bind_space <- function() {
  volume_space(dim = c(2, 2, 2), affine = diag(4), support = 1:4, template = "t")
}

bind_frame <- function(ids, components, parcel = c("a", "a", "b", "b"),
                       block_values = NULL, tables = list(), metadata = list()) {
  sp <- bind_space()
  n <- length(ids)
  block <- axis_block(
    if (is.null(block_values)) matrix(seq_len(n * 2), n, 2) else block_values,
    components = tibble::tibble(.component_id = components)
  )
  fmri_frame(
    assays = list(beta = memory_source(matrix(seq_len(n * 4), n, 4))),
    observations = axis_frame(
      tibble::tibble(.obs_id = ids, grp = rep("g", n)),
      blocks = list(motion = block)
    ),
    features = feature_axis(
      tibble::tibble(.feature_id = feature_ids(sp), parcel = parcel),
      space = sp
    ),
    tables = tables,
    metadata = metadata
  )
}

block_matrix <- function(frame) {
  as.matrix(source_read(as_array_source(axis_block_data(obs_blocks(frame)$motion))))
}

test_that("bound block values follow their component IDs, not their positions", {
  a <- bind_frame(c("o1", "o2", "o3"), c("translation", "rotation"))
  b <- bind_frame(
    c("o4", "o5"), c("rotation", "translation"),
    block_values = matrix(c(100, 200, 300, 400), 2, 2)
  )

  # b's column 1 is rotation (100, 200); column 2 is translation (300, 400).
  ab <- bind_observations(a, b)

  expect_equal(block_component_ids(obs_blocks(ab)$motion), c("translation", "rotation"))

  bound <- block_matrix(ab)
  # a's rows are unchanged.
  expect_equal(unname(bound[1:3, ]), matrix(seq_len(6), 3, 2))
  # b's rows are permuted so translation and rotation land under their labels.
  expect_equal(unname(bound[4:5, 1]), c(300, 400)) # translation
  expect_equal(unname(bound[4:5, 2]), c(100, 200)) # rotation
})

test_that("binding already-aligned blocks leaves the data untouched", {
  a <- bind_frame(c("o1", "o2", "o3"), c("translation", "rotation"))
  b <- bind_frame(
    c("o4", "o5"), c("translation", "rotation"),
    block_values = matrix(c(100, 200, 300, 400), 2, 2)
  )

  bound <- block_matrix(bind_observations(a, b))
  expect_equal(unname(bound[4:5, 1]), c(100, 200))
  expect_equal(unname(bound[4:5, 2]), c(300, 400))
})

test_that("blocks with different component identities are refused", {
  a <- bind_frame(c("o1", "o2"), c("translation", "rotation"))
  b <- bind_frame(c("o3", "o4"), c("translation", "scaling"))

  expect_error(
    bind_observations(a, b),
    class = "fmridataset_error_alignment"
  )
  expect_error(bind_observations(a, b), "scaling")
})

test_that("bind is order-insensitive for component alignment", {
  a <- bind_frame(c("o1", "o2"), c("translation", "rotation"),
    block_values = matrix(c(1, 2, 3, 4), 2, 2)
  )
  b <- bind_frame(c("o3", "o4"), c("rotation", "translation"),
    block_values = matrix(c(10, 20, 30, 40), 2, 2)
  )

  ab <- block_matrix(bind_observations(a, b))
  ba <- block_matrix(bind_observations(b, a))

  # Same values, each under its own label, whichever frame leads.
  expect_equal(unname(ab[3:4, 1]), c(30, 40)) # b translation
  expect_equal(unname(ab[3:4, 2]), c(10, 20)) # b rotation
  # Leading with b, the component order is (rotation, translation), so a's
  # rotation lands in column 1 and its translation in column 2.
  expect_equal(unname(ba[3:4, 1]), c(3, 4)) # a rotation
  expect_equal(unname(ba[3:4, 2]), c(1, 2)) # a translation
  expect_equal(
    block_component_ids(obs_blocks(bind_observations(b, a))$motion),
    c("rotation", "translation")
  )
})

test_that("frames disagreeing on feature metadata are refused", {
  a <- bind_frame(c("o1", "o2"), c("translation", "rotation"))
  b <- bind_frame(c("o3", "o4"), c("translation", "rotation"),
    parcel = c("ZZZ", "ZZZ", "YYY", "YYY")
  )

  expect_error(
    bind_observations(a, b),
    class = "fmridataset_error_alignment"
  )
  expect_error(bind_observations(a, b), "feature metadata")
})

test_that("frames disagreeing on tables or metadata are refused", {
  a <- bind_frame(c("o1", "o2"), c("translation", "rotation"),
    tables = list(events = tibble::tibble(onset = 1)),
    metadata = list(source = "from-A")
  )
  b_tables <- bind_frame(c("o3", "o4"), c("translation", "rotation"),
    tables = list(events = tibble::tibble(onset = 99)),
    metadata = list(source = "from-A")
  )
  b_meta <- bind_frame(c("o5", "o6"), c("translation", "rotation"),
    tables = list(events = tibble::tibble(onset = 1)),
    metadata = list(source = "from-B")
  )

  expect_error(bind_observations(a, b_tables), "tables")
  expect_error(bind_observations(a, b_meta), "metadata")
})

test_that("frames that agree on all annotations still bind", {
  a <- bind_frame(c("o1", "o2"), c("translation", "rotation"),
    tables = list(events = tibble::tibble(onset = 1)),
    metadata = list(source = "shared")
  )
  b <- bind_frame(c("o3", "o4"), c("translation", "rotation"),
    tables = list(events = tibble::tibble(onset = 1)),
    metadata = list(source = "shared")
  )

  ab <- bind_observations(a, b)
  expect_equal(observation_ids(ab), c("o1", "o2", "o3", "o4"))
  expect_equal(features(ab)$parcel, c("a", "a", "b", "b"))
})
