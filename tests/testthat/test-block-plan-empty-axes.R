# A zero-length axis makes .plan_block_shape() return c(0L, 0L), so the OTHER
# axis reached .axis_block_ranges() with block_size 0 and a non-zero length,
# and seq.int(1L, n, by = 0L) raised a raw base-R error:
#
#   plan_blocks(f[, integer(0)])
#   #> simpleError: invalid '(to - from)/by'
#
# Such frames are legal - any filter_obs() or select_features() that matches
# nothing produces one, and they subset, collect, and round-trip correctly
# everywhere else. There is nothing to read, so the plan holds no blocks.

empty_axis_frame <- function() {
  fmri_frame(
    list(a = matrix(rnorm(200), 20, 10)),
    observations = data.frame(.obs_id = sprintf("o%02d", 1:20))
  )
}

test_that("frames emptied on either axis plan zero blocks in every layout", {
  full <- empty_axis_frame()
  cases <- list(
    "no features" = full[, integer(0)],
    "no observations" = full[integer(0), ],
    "neither" = full[integer(0), integer(0)],
    "single row, no features" = full[1, integer(0)]
  )

  for (label in names(cases)) {
    frame <- cases[[label]]
    for (layout in c("balanced", "imagewise", "featurewise")) {
      plan <- plan_blocks(frame, layout = layout)
      expect_s3_class(plan, "frame_block_plan")
      expect_equal(plan$n_blocks, 0L, info = paste(label, layout))
      expect_equal(plan$total_bytes, 0, info = paste(label, layout))
    }
  }
})

test_that("an empty frame still collects and reports its shape", {
  full <- empty_axis_frame()
  empty <- full[, integer(0)]

  expect_equal(dim(empty), c(20L, 0L))
  expect_equal(dim(collect_assay(empty)), c(20L, 0L))
  expect_equal(length(feature_ids(empty)), 0L)
})

test_that("executing an empty plan yields no results rather than erroring", {
  frame <- empty_axis_frame()[, integer(0)]
  plan <- plan_blocks(frame)

  expect_equal(length(execute_block_plan(frame, plan, function(block, ...) block)), 0L)
  expect_equal(length(block_apply(frame, function(values, ids) values)), 0L)
})

test_that("selections that match nothing plan cleanly", {
  frame <- empty_axis_frame()

  expect_equal(plan_blocks(filter_obs(frame, rep(FALSE, 20)))$n_blocks, 0L)
  expect_equal(plan_blocks(select_features(frame, rep(FALSE, 10)))$n_blocks, 0L)
})

test_that("non-empty frames are unaffected", {
  frame <- empty_axis_frame()

  for (layout in c("balanced", "imagewise", "featurewise")) {
    plan <- plan_blocks(frame, layout = layout)
    expect_gt(plan$n_blocks, 0L)
    expect_equal(plan$total_values, 200)
  }
})
