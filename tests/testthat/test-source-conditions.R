test_that("source_error() builds the three protocol conditions with their fields", {
  stale <- source_error(
    "Store changed.", type = "stale",
    source = "a.h5", expected = "x", actual = "y"
  )
  expect_s3_class(stale, "fmridataset_error_source_stale")
  expect_s3_class(stale, "fmridataset_error")
  expect_identical(stale$source, "a.h5")
  expect_identical(stale$expected, "x")
  expect_identical(stale$actual, "y")

  io <- source_error("Read failed.", type = "io", file = "a.h5", operation = "read")
  expect_s3_class(io, "fmridataset_error_backend_io")
  expect_identical(io$operation, "read")

  contract <- source_error("Bad dtype.", type = "contract", field = "dtype")
  expect_s3_class(contract, "fmridataset_error_source_contract")
  expect_identical(contract$field, "dtype")

  caught <- tryCatch(stop(stale), fmridataset_error_source_stale = function(e) e$actual)
  expect_identical(caught, "y")
})

test_that("source_error() rejects malformed messages and unnamed fields", {
  expect_error(source_error("", type = "io"), class = "fmridataset_error_source_contract")
  expect_error(
    source_error("Bad.", type = "io", "unnamed"),
    class = "fmridataset_error_source_contract"
  )
  expect_error(source_error("Bad.", type = "other"))
})

test_that("a descriptor without protocol methods fails with a contract error naming them", {
  bogus <- structure(list(shape = c(2L, 2L)), class = c("bogus_source", "array_source"))
  err <- expect_error(
    validate_array_source(bogus),
    class = "fmridataset_error_source_contract"
  )
  expect_identical(err$field, "methods")
  expect_true(all(c("source_shape", "source_read") %in% err$missing))
  expect_match(conditionMessage(err), "bogus_source")

  expect_error(
    validate_array_source(list(shape = c(2L, 2L))),
    class = "fmridataset_error_source_contract"
  )
})

test_that("equal-valued memory sources share a fingerprint only under a shared revision", {
  values <- matrix(seq_len(6), nrow = 2)
  a <- memory_source(values)
  b <- memory_source(values)
  expect_false(identical(source_fingerprint(a), source_fingerprint(b)))

  a_rev <- memory_source(values, revision = "run-1")
  b_rev <- memory_source(values, revision = "run-1")
  c_rev <- memory_source(values, revision = "run-2")
  expect_identical(source_fingerprint(a_rev), source_fingerprint(b_rev))
  expect_false(identical(source_fingerprint(a_rev), source_fingerprint(c_rev)))
  expect_identical(content_hash(a_rev), content_hash(c_rev))
})
