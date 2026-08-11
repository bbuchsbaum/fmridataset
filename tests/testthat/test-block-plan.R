test_that("block planners are metadata-only and honor layout budgets", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame

  plans <- lapply(
    c("balanced", "imagewise", "featurewise"),
    function(layout) {
      plan_blocks(
        x,
        layout = layout,
        memory_budget = 64,
        target_block_bytes = 64
      )
    }
  )
  names(plans) <- c("balanced", "imagewise", "featurewise")

  expect_true(all(vapply(
    assays(x),
    function(value) source_counts(value$source)$bytes,
    numeric(1)
  ) == 0))
  for (plan in plans) {
    manifest <- block_manifest(plan)
    expect_s3_class(plan, "frame_block_plan")
    expect_true(all(manifest$.bytes <= 64))
    expect_equal(sum(manifest$.n_observation * manifest$.n_feature), 42)
    expect_identical(plan$max_block_bytes, max(manifest$.bytes))
    expect_false(contains_runtime_state(plan))
    expect_identical(
      unserialize(serialize(plan, NULL)),
      plan
    )
  }

  expect_true(all(block_manifest(plans$imagewise)$.n_feature == ncol(x)))
  expect_true(all(block_manifest(plans$featurewise)$.n_observation == nrow(x)))
  expect_true(plans$balanced$block_shape[[1L]] < nrow(x))
  expect_true(plans$balanced$block_shape[[2L]] < ncol(x))
})

test_that("block plan execution covers each cell exactly once", {
  fx <- make_frame_fixture(instrument = TRUE)
  x <- fx$frame
  plan <- plan_blocks(
    x,
    layout = "balanced",
    memory_budget = 64,
    target_block_bytes = 64
  )
  realized <- matrix(NA_real_, nrow(x), ncol(x))

  pieces <- execute_block_plan(x, plan, function(values, observation_ids, feature_ids, block) {
    realized[
      block$.observation_start:block$.observation_end,
      block$.feature_start:block$.feature_end
    ] <<- values
    list(
      observations = observation_ids,
      features = feature_ids,
      dimensions = dim(values)
    )
  })

  expect_equal(realized, fx$beta)
  expect_length(pieces, nrow(block_manifest(plan)))
  expect_identical(
    source_counts(assays(x)$beta$source)$reads,
    as.numeric(nrow(block_manifest(plan)))
  )
  expect_true(all(vapply(pieces, function(piece) length(piece$observations), integer(1)) > 0L))
})

test_that("profile planners reject impossible atomic reads", {
  x <- make_frame_fixture()$frame

  expect_error(
    plan_blocks(x, layout = "imagewise", memory_budget = 40, target_block_bytes = 40),
    class = "fmridataset_error_budget"
  )
  expect_error(
    plan_blocks(x, layout = "featurewise", memory_budget = 48, target_block_bytes = 48),
    class = "fmridataset_error_budget"
  )
  expect_error(
    plan_blocks(x, layout = "balanced", memory_budget = 7, target_block_bytes = 7),
    class = "fmridataset_error_budget"
  )
})

test_that("balanced planning scales source chunks without changing their aspect", {
  shape <- c(200000L, 50000L)
  chunks <- c(64L, 4096L)
  capacity <- 1024^2

  expect_identical(
    fmridataset:::.plan_block_shape(shape, chunks, "balanced", capacity),
    c(128L, 8192L)
  )
})

test_that("block execution rejects plans for another selection", {
  x <- make_frame_fixture()$frame
  plan <- plan_blocks(x, memory_budget = 64, target_block_bytes = 64)

  expect_error(
    execute_block_plan(x[1:3, ], plan, identity),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    execute_block_plan(x, plan, identity, assay = "variance"),
    class = "fmridataset_error_source_contract"
  )
})
