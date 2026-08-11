test_that("axis_frame generates persistent unique IDs", {
  x <- axis_frame(data.frame(value = 1:3))

  expect_s3_class(x, "axis_frame")
  expect_length(axis_ids(x), 3)
  expect_true(all(grepl("^obs-", axis_ids(x))))
  expect_equal(anyDuplicated(axis_ids(x)), 0L)
  expect_identical(axis_ids(x[3:1]), rev(axis_ids(x)))
})

test_that("axis_frame preserves supplied IDs and factors exactly", {
  d <- data.frame(
    .obs_id = c("a", "b"),
    condition = factor(c("B", "A"), levels = c("A", "B"), ordered = TRUE)
  )
  x <- axis_frame(d)

  expect_identical(axis_ids(x), c("a", "b"))
  expect_identical(levels(axis_data(x)$condition), c("A", "B"))
  expect_true(is.ordered(axis_data(x)$condition))
})

test_that("axis_frame rejects invalid IDs", {
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", "a"))),
    class = "fmridataset_error_alignment"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", NA_character_))),
    class = "fmridataset_error_alignment"
  )
})

test_that("axis blocks validate their leading dimension and components", {
  b <- axis_block(
    matrix(1:6, 3, 2),
    components = data.frame(.component_id = c("x", "y")),
    role = "continuous"
  )
  x <- axis_frame(data.frame(v = 1:3), blocks = list(embed = b))

  expect_identical(dim(axis_block_data(axis_blocks(x)$embed)), c(3L, 2L))
  expect_identical(block_component_ids(axis_blocks(x)$embed), c("x", "y"))
  expect_error(
    axis_frame(data.frame(v = 1:2), blocks = list(embed = b)),
    class = "fmridataset_error_alignment"
  )
})
