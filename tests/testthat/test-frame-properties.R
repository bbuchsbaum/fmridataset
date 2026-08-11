test_that("built-in sources and spaces satisfy reusable conformance", {
  m <- matrix(seq_len(30), 5, 6)
  expect_array_source_conformance(memory_source(m), m)
  expect_array_source_conformance(counting_source(memory_source(m)), m)

  expect_feature_space_conformance(index_space(6))
  expect_feature_space_conformance(volume_space(c(2, 2, 2), support = 1:6))
})

test_that("randomized synchronized views equal dense reference slicing", {
  set.seed(20260811)
  fx <- make_frame_fixture()

  for (iteration in seq_len(50)) {
    rows <- sample(seq_len(nrow(fx$beta)), sample.int(nrow(fx$beta) + 1L, 1L) - 1L)
    cols <- sample(seq_len(ncol(fx$beta)), sample.int(ncol(fx$beta) + 1L, 1L) - 1L)
    view <- fx$frame[rows, cols]

    expect_equal(
      collect_assay(view),
      fx$beta[rows, cols, drop = FALSE],
      info = paste("random view", iteration)
    )
    expect_identical(observation_ids(view), observation_ids(fx$frame)[rows])
    expect_identical(feature_ids(view), feature_ids(fx$frame)[cols])
  }
})

test_that("row-bound sources preserve row order and push down selections", {
  a <- matrix(1:12, 3, 4)
  b <- matrix(101:108, 2, 4)
  source <- row_bound_source(list(memory_source(a), memory_source(b)))
  reference <- rbind(a, b)

  expect_array_source_conformance(source, reference)
  expect_equal(source_read(source, c(5, 2, 4), c(4, 1)), reference[c(5, 2, 4), c(4, 1)])
})

test_that("row-sharded sources expose stable manifests and exact row mappings", {
  shards <- list(
    memory_source(matrix(1:12, 3, 4)),
    memory_source(matrix(21:28, 2, 4)),
    memory_source(matrix(41:52, 3, 4))
  )
  source <- row_sharded_source(
    shards,
    shard_ids = c("sub-01_run-1", "sub-01_run-2", "sub-02_run-1"),
    shard_data = data.frame(subject = c("sub-01", "sub-01", "sub-02"))
  )

  expect_s3_class(source, "row_sharded_source")
  expect_invisible(validate_array_source(source))
  expect_identical(
    shard_manifest(source),
    data.frame(
      .shard_id = c("sub-01_run-1", "sub-01_run-2", "sub-02_run-1"),
      .start = c(1L, 4L, 6L),
      .end = c(3L, 5L, 8L),
      .n_observation = c(3L, 2L, 3L),
      .source_fingerprint = vapply(shards, source_fingerprint, character(1)),
      subject = c("sub-01", "sub-01", "sub-02")
    )
  )
  expect_identical(
    locate_source_rows(source, c(8L, 1L, 6L, 1L)),
    data.frame(
      .request_position = 1:4,
      .observation = c(8L, 1L, 6L, 1L),
      .shard_index = c(3L, 1L, 3L, 1L),
      .shard_id = c("sub-02_run-1", "sub-01_run-1", "sub-02_run-1", "sub-01_run-1"),
      .local_observation = c(3L, 1L, 1L, 1L)
    )
  )
})

test_that("row-sharded reads touch only selected shards once", {
  matrices <- list(
    matrix(1:12, 3, 4),
    matrix(21:28, 2, 4),
    matrix(41:52, 3, 4)
  )
  children <- lapply(matrices, function(x) counting_source(memory_source(x)))
  source <- row_sharded_source(children, shard_ids = c("a", "b", "c"))
  observations <- c(8L, 1L, 6L, 1L)
  features <- c(4L, 1L, 4L)
  reference <- do.call(rbind, matrices)[observations, features, drop = FALSE]

  expect_equal(source_read(source, observations, features), reference)
  expect_identical(vapply(children, function(x) source_counts(x)$reads, numeric(1)), c(1, 0, 1))
  expect_identical(vapply(children, function(x) source_counts(x)$values, numeric(1)), c(6, 0, 6))
})

test_that("row-sharded descriptors serialize and append immutably", {
  a <- memory_source(matrix(1:12, 3, 4))
  b <- memory_source(matrix(21:28, 2, 4))
  first <- row_sharded_source(list(a), shard_ids = "a")
  appended <- append_source_shards(first, list(b), shard_ids = "b")
  restored <- unserialize(serialize(appended, NULL))

  expect_identical(source_shape(first), c(3L, 4L))
  expect_identical(source_shape(appended), c(5L, 4L))
  expect_identical(shard_manifest(first)$.shard_id, "a")
  expect_identical(shard_manifest(restored)$.shard_id, c("a", "b"))
  expect_identical(source_fingerprint(restored), source_fingerprint(appended))
  expect_false(contains_runtime_state(restored))
  expect_equal(delarr::collect(as_delarr(restored)), rbind(a$data, b$data))
})

test_that("row-sharded sources reject ambiguous or incompatible manifests", {
  a <- memory_source(matrix(1:12, 3, 4))
  wrong_features <- memory_source(matrix(1:15, 3, 5))
  wrong_dtype <- memory_source(matrix(1:8, 2, 4), dtype = "float32")
  empty <- memory_source(matrix(numeric(), 0, 4))

  expect_error(row_sharded_source(list(a, a), shard_ids = c("same", "same")), "unique")
  expect_error(row_sharded_source(list(a, wrong_features)), "feature count")
  expect_error(row_sharded_source(list(a, wrong_dtype)), "dtype")
  expect_error(row_sharded_source(list(empty)), "zero observations")
  expect_error(
    row_sharded_source(list(a), shard_data = data.frame(.start = 1L)),
    "reserved"
  )
})

test_that("frame binding rejects shape-only spatial matches", {
  obs1 <- data.frame(.obs_id = "a")
  obs2 <- data.frame(.obs_id = "b")
  x <- fmri_frame(list(x = matrix(1:3, 1)), obs1, space = index_space(3))
  y <- fmri_frame(list(x = matrix(4:6, 1)), obs2, space = index_space(3))

  expect_error(bind_observations(x, y), class = "fmridataset_error_space_mismatch")
})

test_that("frame binding preserves lazy selectors from frame views", {
  fx <- make_frame_fixture()
  first <- fx$frame[c(7L, 2L), ]
  second <- fx$frame[c(4L, 5L), ]
  bound <- bind_observations(first, second)

  expect_identical(
    observation_ids(bound),
    c(observation_ids(first), observation_ids(second))
  )
  expect_s3_class(assay(bound, "beta")$source, "row_sharded_source")
  expect_equal(
    collect_assay(bound, "beta"),
    rbind(collect_assay(first, "beta"), collect_assay(second, "beta"))
  )
})
