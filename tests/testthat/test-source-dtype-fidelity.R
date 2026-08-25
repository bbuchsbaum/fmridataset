# Composing sources preallocated matrix(NA_real_, ...) regardless of dtype, so
# reading a logical assay through row_bound_source() or row_sharded_source()
# handed back doubles while source_dtype() still reported "logical". The
# descriptor and the data disagreed, which matters because the descriptor is
# what budgets, manifests, and downstream consumers are entitled to trust.

logical_source <- function(n_obs = 3, n_feat = 2) {
  memory_source(matrix(
    rep(c(TRUE, FALSE), length.out = n_obs * n_feat),
    n_obs, n_feat
  ))
}

test_that("realized mode agrees with the declared dtype", {
  expect_equal(fmridataset:::.realized_dtype_mode("logical"), "logical")
  expect_equal(fmridataset:::.realized_dtype_mode("complex64"), "complex")
  expect_equal(fmridataset:::.realized_dtype_mode("complex128"), "complex")
  for (dtype in c("uint8", "int32", "float32", "float64", "int64")) {
    expect_equal(fmridataset:::.realized_dtype_mode(dtype), "double", info = dtype)
  }
  expect_error(
    fmridataset:::.realized_dtype_mode("float128"),
    class = "fmridataset_error_source_contract"
  )
})

test_that("row_bound_source preserves a logical assay", {
  child <- logical_source()
  bound <- row_bound_source(list(child, child))

  expect_identical(source_dtype(bound), "logical")
  expect_identical(storage.mode(source_read(bound)), "logical")
  expect_identical(source_read(bound)[1:3, ], source_read(child))
  expect_identical(source_read(bound)[4:6, ], source_read(child))
})

test_that("row_sharded_source preserves a logical assay", {
  child <- logical_source()
  sharded <- row_sharded_source(list(a = child, b = child))

  expect_identical(source_dtype(sharded), "logical")
  expect_identical(storage.mode(source_read(sharded)), "logical")
})

test_that("empty selections keep the declared type", {
  bound <- row_bound_source(list(logical_source(), logical_source()))

  expect_identical(
    storage.mode(source_read(bound, observations = integer(0))),
    "logical"
  )
  expect_identical(
    storage.mode(source_read(bound, features = integer(0))),
    "logical"
  )
  expect_equal(dim(source_read(bound, observations = integer(0))), c(0L, 2L))
})

test_that("numeric assays are unaffected", {
  child <- memory_source(matrix(as.double(1:6), 3, 2))
  bound <- row_bound_source(list(child, child))

  expect_identical(source_dtype(bound), "float64")
  expect_identical(storage.mode(source_read(bound)), "double")
  expect_equal(source_read(bound)[1:3, ], source_read(child))
})

test_that("reordered and partial reads keep the declared type", {
  child <- logical_source(n_obs = 3, n_feat = 2)
  bound <- row_bound_source(list(child, child))

  out <- source_read(bound, observations = c(5L, 1L), features = 2L)
  expect_identical(storage.mode(out), "logical")
  expect_equal(dim(out), c(2L, 1L))
})
