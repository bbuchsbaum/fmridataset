# row_sharded_source() rejected any shard with zero observations, which made
# bind_observations() refuse a zero-observation frame:
#
#   bind_observations(A, Z)
#   #> fmridataset_error_alignment: Row-sharded sources cannot contain shards
#   #>   with zero observations.
#
# Such frames are legal everywhere else in the model - they subset, collect,
# plan, and FDS round-trip - so binding one should be an identity. An empty
# shard makes two consecutive boundaries equal, and findInterval() resolves a
# row to the last boundary at or below it, so an empty shard is never routed
# to. These tests pin that routing rather than the old restriction.

rows_source <- function(n) memory_source(matrix(as.double(seq_len(n * 2)), n, 2))
empty_source <- function() memory_source(matrix(numeric(0), 0, 2))

test_that("an empty shard is allowed in any position and never routed to", {
  cases <- list(
    first = list(a = empty_source(), b = rows_source(2), c = rows_source(2)),
    middle = list(a = rows_source(2), b = empty_source(), c = rows_source(2)),
    last = list(a = rows_source(2), b = rows_source(2), c = empty_source())
  )
  reference <- source_read(row_sharded_source(list(
    a = rows_source(2), b = rows_source(2)
  )))

  for (label in names(cases)) {
    sharded <- row_sharded_source(cases[[label]])
    expect_equal(source_shape(sharded), c(4L, 2L), info = label)
    expect_equal(source_read(sharded), reference, info = label)
  }
})

test_that("an all-empty shard set is a valid zero-observation source", {
  sharded <- row_sharded_source(list(a = empty_source(), b = empty_source()))

  expect_equal(source_shape(sharded), c(0L, 2L))
  expect_equal(dim(source_read(sharded)), c(0L, 2L))
})

test_that("routing skips the empty shard", {
  sharded <- row_sharded_source(list(
    a = rows_source(3), b = empty_source(), c = rows_source(2)
  ))

  located <- locate_source_rows(sharded, 1:5)
  expect_equal(
    located$.shard_id,
    c("shard-000001", "shard-000001", "shard-000001", "shard-000003", "shard-000003")
  )
  expect_false("shard-000002" %in% located$.shard_id)

  # Row 4 is the first row of the third shard.
  expect_equal(unname(source_read(sharded, observations = 4L)), matrix(c(1, 3), 1, 2))
  # Reordered selection still resolves per shard.
  expect_equal(
    unname(source_read(sharded, observations = c(5L, 1L))[, 1]),
    c(2, 1)
  )
})

test_that("binding a zero-observation frame is an identity", {
  space <- index_space(2, ids = c("f1", "f2"))
  make <- function(ids) {
    fmri_frame(
      list(a = memory_source(matrix(as.double(seq_len(2 * length(ids))), length(ids), 2))),
      observations = data.frame(.obs_id = ids),
      space = space
    )
  }
  populated <- make(c("o1", "o2"))
  empty <- make(character(0))

  expect_equal(dim(empty), c(0L, 2L))

  for (bound in list(
    bind_observations(populated, empty),
    bind_observations(empty, populated)
  )) {
    expect_equal(observation_ids(bound), c("o1", "o2"))
    expect_equal(collect_assay(bound), collect_assay(populated))
  }
})

test_that("genuinely incompatible shards are still refused", {
  a <- rows_source(3)

  expect_error(row_sharded_source(list(a, memory_source(matrix(1:15, 3, 5)))), "feature count")
  expect_error(
    row_sharded_source(list(a, memory_source(matrix(1:4, 2, 2), dtype = "float32"))),
    "dtype"
  )
  expect_error(row_sharded_source(list(a, a), shard_ids = c("same", "same")), "unique")
})
