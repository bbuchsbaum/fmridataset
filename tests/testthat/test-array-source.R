test_that("memory_source implements the source contract", {
  m <- matrix(seq_len(12), 3, 4)
  x <- memory_source(m)

  expect_s3_class(x, "array_source")
  expect_identical(source_shape(x), c(3L, 4L))
  expect_identical(source_dtype(x), "float64")
  expect_true(all(c("row_slice", "column_slice", "block_slice") %in% source_capabilities(x)))
  expect_equal(source_read(x, c(3, 1), c(4, 2)), m[c(3, 1), c(4, 2), drop = FALSE])
})

test_that("memory_source handles empty slices", {
  x <- memory_source(matrix(1:6, 2, 3))
  expect_identical(dim(source_read(x, integer(), 1:2)), c(0L, 2L))
  expect_identical(dim(source_read(x, 1:2, integer())), c(2L, 0L))
})

test_that("counting_source records requested assay bytes", {
  x <- counting_source(memory_source(matrix(1:12, 3, 4)))

  expect_equal(source_counts(x)$bytes, 0)
  source_shape(x)
  expect_equal(source_counts(x)$bytes, 0)
  source_read(x, 1:2, 1:3)
  expect_equal(source_counts(x)$reads, 1)
  expect_equal(source_counts(x)$values, 6)
  expect_gt(source_counts(x)$bytes, 0)
})

test_that("fault_source fails at the configured lifecycle stage", {
  x <- fault_source(memory_source(matrix(1:4, 2)), stage = "read")
  expect_error(source_read(x, 1, 1), class = "fmridataset_error_backend_io")
})

test_that("array sources construct runtime delarr plans", {
  m <- matrix(seq_len(12), 3, 4)
  x <- as_delarr(memory_source(m))
  expect_equal(delarr::collect(x[3:1, c(4, 2)]), m[3:1, c(4, 2), drop = FALSE])
})
