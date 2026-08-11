test_that("memory_source implements the source contract", {
  m <- matrix(seq_len(12), 3, 4)
  x <- memory_source(m)

  expect_s3_class(x, "array_source")
  expect_identical(source_shape(x), c(3L, 4L))
  expect_identical(source_dtype(x), "float64")
  expect_true(all(c("row_slice", "column_slice", "block_slice") %in% source_capabilities(x)))
  expect_equal(source_read(x, c(3, 1), c(4, 2)), m[c(3, 1), c(4, 2), drop = FALSE])
})

test_that("source contracts expose complete normalized descriptors", {
  x <- memory_source(matrix(as.double(1:12), 3, 4), chunks = c(2, 3))
  descriptor <- source_descriptor(x)

  expect_identical(
    names(descriptor),
    c("shape", "dtype", "chunks", "capabilities", "fingerprint")
  )
  expect_identical(descriptor$shape, c(3L, 4L))
  expect_identical(descriptor$dtype, "float64")
  expect_identical(descriptor$chunks, c(2L, 3L))
  expect_true(all(c("block_slice", "serializable") %in% descriptor$capabilities))
  expect_match(descriptor$fingerprint, "^[0-9a-f]{64}$")
  expect_invisible(validate_array_source(x))
})

test_that("source contract validation rejects unsafe or ambiguous descriptors", {
  x <- memory_source(matrix(1:6, 2, 3))

  bad_dtype <- x
  bad_dtype$dtype <- "mystery"
  expect_error(
    validate_array_source(bad_dtype),
    class = "fmridataset_error_source_contract"
  )
  bad_chunks <- x
  bad_chunks$chunks <- c(3L, 1L)
  expect_error(
    validate_array_source(bad_chunks),
    class = "fmridataset_error_source_contract"
  )
  bad_capabilities <- x
  bad_capabilities$capabilities <- "row_slice"
  expect_error(
    validate_array_source(bad_capabilities),
    class = "fmridataset_error_source_contract"
  )
  unsafe <- x
  unsafe$loader <- function() NULL
  expect_error(
    validate_array_source(unsafe),
    class = "fmridataset_error_source_contract"
  )
  expect_error(memory_source(matrix(1:4, 2), dtype = "unknown"), "Unsupported")
})

test_that("fingerprints encode logical source transformations", {
  x <- memory_source(matrix(1:12, 3, 4), chunks = c(2, 2))
  same <- unserialize(serialize(x, NULL))
  reordered <- source_view(x, observations = c(3, 1), features = c(4, 2))
  changed_chunks <- memory_source(matrix(1:12, 3, 4), chunks = c(3, 4))

  expect_identical(source_fingerprint(x), source_fingerprint(same))
  expect_false(identical(source_fingerprint(x), source_fingerprint(reordered)))
  expect_false(identical(source_fingerprint(x), source_fingerprint(changed_chunks)))
  expect_identical(
    source_fingerprint(counting_source(x)),
    source_fingerprint(x)
  )
  expect_false(identical(
    source_fingerprint(fault_source(x, "read")),
    source_fingerprint(fault_source(x, "open"))
  ))
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
