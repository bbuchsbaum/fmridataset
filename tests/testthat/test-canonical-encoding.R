.canonical_golden_values <- function() {
  sparse <- Matrix::sparseMatrix(
    i = c(2L, 1L, 2L), j = c(1L, 2L, 2L), x = c(3, -0, NA_real_),
    dims = c(2L, 2L), dimnames = list(c("r1", "r2"), c("c1", "c2"))
  )
  list(
    null = NULL,
    integer = c(1L, NA_integer_, -2147483647L),
    doubles = c(0, -0, NA_real_, NaN, Inf, -Inf, pi),
    utf8 = c("caf\u00e9", "e\u0301", NA_character_),
    factor = factor(c("b", "a", NA), levels = c("a", "b")),
    ordered = ordered(c("low", "high"), levels = c("low", "high")),
    matrix = matrix(c(1, 2, 3, 4), 2L, dimnames = list(c("r1", "r2"), NULL)),
    sparse = sparse,
    sequence = list("b", "a"),
    record = list(beta = 2L, alpha = list(z = NULL, a = TRUE)),
    missing_field = list(alpha = 1L),
    null_field = list(alpha = 1L, beta = NULL)
  )
}

test_that("canonicalization v1 publishes explicit byte-level rules", {
  contract <- canonicalization_contract()
  expect_identical(contract$encoding, "fmridataset-tagged-binary")
  expect_identical(contract$unicode, "NFC")
  expect_identical(contract$byte_order, "big-endian")
  expect_identical(contract$named_field_order, "lexicographic")
  expect_identical(contract$sequence_order, "preserved")
  expect_identical(contract$negative_zero, "preserved")
  expect_identical(contract$nan, "canonical-payload")
})

test_that("canonical bytes normalize records strings and sparse storage", {
  expect_type(canonical_bytes(list(a = 1L)), "raw")
  expect_identical(
    canonical_bytes(list(beta = 2L, alpha = 1L)),
    canonical_bytes(list(alpha = 1L, beta = 2L))
  )
  expect_false(identical(
    canonical_bytes(list("beta", "alpha")),
    canonical_bytes(list("alpha", "beta"))
  ))
  expect_identical(
    canonical_bytes("\u00e9"),
    canonical_bytes("e\u0301")
  )
  expect_false(identical(canonical_bytes(0), canonical_bytes(-0)))
  expect_false(identical(canonical_bytes(NA_real_), canonical_bytes(NaN)))
  expect_false(identical(canonical_bytes(list(a = 1L)),
                         canonical_bytes(list(a = 1L, b = NULL))))

  x <- Matrix::sparseMatrix(i = c(2L, 1L), j = c(1L, 2L), x = c(3, 4))
  y <- methods::as(x, "TsparseMatrix")
  expect_identical(canonical_bytes(x), canonical_bytes(y))
})

test_that("canonical dimensions factors and ordered factors are semantic", {
  expect_false(identical(canonical_bytes(1:4), canonical_bytes(matrix(1:4, 2L))))
  expect_false(identical(
    canonical_bytes(factor("a", levels = c("a", "b"))),
    canonical_bytes(factor("a", levels = c("b", "a")))
  ))
  expect_false(identical(
    canonical_bytes(factor("a", levels = c("a", "b"))),
    canonical_bytes(ordered("a", levels = c("a", "b")))
  ))
})

test_that("canonical bytes reject runtime state", {
  expect_error(canonical_bytes(globalenv()), class = "fmridataset_error_identity")
  expect_error(canonical_bytes(function() NULL), class = "fmridataset_error_identity")
})

test_that("canonical bytes handle classed list and atomic storage", {
  version <- package_version("4.5.1")
  instant <- as.POSIXct("2026-08-12 12:34:56", tz = "UTC")

  expect_length(canonical_sha256(version), 1L)
  expect_length(canonical_sha256(instant), 1L)
  expect_identical(
    canonical_sha256(version),
    canonical_sha256(unserialize(serialize(version, NULL)))
  )
  expect_identical(
    canonical_sha256(instant),
    canonical_sha256(unserialize(serialize(instant, NULL)))
  )
})

test_that("canonical bytes reject unsupported S4 objects deterministically", {
  methods::setClass("canonical_test_s4", slots = c(value = "numeric"))
  value <- methods::new("canonical_test_s4", value = 1)

  expect_error(
    canonical_bytes(value),
    class = "fmridataset_error_identity",
    regexp = "not arbitrary S4"
  )
})

test_that("published canonical v1 golden vectors match bytes and SHA-256", {
  path <- system.file(
    "golden", "r-canonical-v1.tsv", package = "fmridataset"
  )
  if (!nzchar(path)) path <- testthat::test_path("..", "..", "inst", "golden", "r-canonical-v1.tsv")
  expect_true(file.exists(path))
  golden <- utils::read.delim(path, colClasses = "character", check.names = FALSE)
  values <- .canonical_golden_values()
  expect_identical(golden$name, names(values))
  for (name in names(values)) {
    bytes <- canonical_bytes(values[[name]])
    expect_identical(paste(sprintf("%02x", as.integer(bytes)), collapse = ""),
                     golden$bytes_hex[golden$name == name], info = name)
    expect_identical(canonical_sha256(values[[name]]),
                     golden$sha256[golden$name == name], info = name)
  }
})
