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

test_that("frame binding rejects shape-only spatial matches", {
  obs1 <- data.frame(.obs_id = "a")
  obs2 <- data.frame(.obs_id = "b")
  x <- fmri_frame(list(x = matrix(1:3, 1)), obs1, space = index_space(3))
  y <- fmri_frame(list(x = matrix(4:6, 1)), obs2, space = index_space(3))

  expect_error(bind_observations(x, y), class = "fmridataset_error_space_mismatch")
})
