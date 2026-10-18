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
  expect_false(identical(
    canonical_bytes(list(a = 1L)),
    canonical_bytes(list(a = 1L, b = NULL))
  ))

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
    "golden", "r-canonical-v1.tsv",
    package = "fmridataset"
  )
  if (!nzchar(path)) path <- testthat::test_path("..", "..", "inst", "golden", "r-canonical-v1.tsv")
  expect_true(file.exists(path))
  golden <- utils::read.delim(path, colClasses = "character", check.names = FALSE)
  values <- .canonical_golden_values()
  expect_identical(golden$name, names(values))
  for (name in names(values)) {
    bytes <- canonical_bytes(values[[name]])
    expect_identical(paste(sprintf("%02x", as.integer(bytes)), collapse = ""),
      golden$bytes_hex[golden$name == name],
      info = name
    )
    expect_identical(canonical_sha256(values[[name]]),
      golden$sha256[golden$name == name],
      info = name
    )
  }
})

test_that("sparse canonical bytes depend on values, not storage layout", {
  zero <- Matrix::sparseMatrix(
    i = integer(), j = integer(), x = double(), dims = c(3L, 3L)
  )
  explicit_identity <- methods::as(
    methods::as(diag(3), "CsparseMatrix"), "generalMatrix"
  )
  expect_identical(canonical_bytes(Matrix::Diagonal(3)), canonical_bytes(explicit_identity))
  expect_false(identical(canonical_bytes(Matrix::Diagonal(3)), canonical_bytes(zero)))
  unit_triangular <- methods::new("dtCMatrix",
    Dim = c(3L, 3L), diag = "U", uplo = "U",
    p = c(0L, 0L, 0L, 0L), i = integer(), x = double()
  )
  expect_identical(canonical_bytes(unit_triangular), canonical_bytes(explicit_identity))

  general <- Matrix::sparseMatrix(
    i = c(1L, 2L, 1L, 2L), j = c(1L, 1L, 2L, 2L), x = c(2, 1, 1, 2), dims = c(2L, 2L)
  )
  expect_identical(
    canonical_bytes(Matrix::forceSymmetric(general)),
    canonical_bytes(general)
  )

  stored_zero <- Matrix::sparseMatrix(
    i = c(1L, 2L), j = c(1L, 2L), x = c(1, 0), dims = c(2L, 2L)
  )
  expect_identical(canonical_bytes(stored_zero), canonical_bytes(Matrix::drop0(stored_zero)))
  stored_negative_zero <- Matrix::sparseMatrix(
    i = c(1L, 2L), j = c(1L, 2L), x = c(1, -0), dims = c(2L, 2L)
  )
  expect_false(identical(
    canonical_bytes(stored_negative_zero),
    canonical_bytes(Matrix::drop0(stored_zero))
  ))
})

test_that("lists cannot forge the reserved sparse matrix record", {
  real <- Matrix::sparseMatrix(i = 1L, j = 1L, x = 2, dims = c(1L, 1L))
  fake <- list(
    .canonical_type = "sparse_matrix", value_type = "double",
    dimensions = c(1L, 1L), dimnames = list(NULL, NULL), i = 1L, j = 1L, x = 2
  )
  expect_length(canonical_sha256(real), 1L)
  expect_error(canonical_bytes(fake), class = "fmridataset_error_identity")
  expect_error(
    canonical_bytes(list(outer = list(.canonical_type = "x"))),
    class = "fmridataset_error_identity"
  )
})

test_that("canonical strings are NFC and repair ill-formed UTF-8 like ICU", {
  canon <- fmridataset:::.canonical_utf8
  codepoints <- function(x) lapply(x, function(value) as.numeric(utf8ToInt(value)))
  from_hex <- function(hex) {
    value <- rawToChar(as.raw(strtoi(strsplit(hex, " ")[[1]], 16L)))
    Encoding(value) <- "UTF-8"
    value
  }

  composed <- canon(c(
    "é", "Å", "각", "abc", "", NA,
    "ẛ̣"
  ))
  expect_identical(
    codepoints(composed),
    list(0xE9, 0xC5, 0xAC01, c(0x61, 0x62, 0x63), numeric(), NA_real_,
      c(0x1E9B, 0x0323))
  )
  expect_identical(Encoding(composed[c(1, 4)]), c("UTF-8", "unknown"))
  expect_identical(canon(c(a = "é"))[["a"]], "é")
  expect_identical(canon(character()), character())

  ill_formed <- vapply(c(
    "61 ff 62", "e2 82", "e2 82 61", "c0 af", "ed a0 80", "f4 90 80 80",
    "f0 9f 98 61", "80 80", "e0 80 80", "c3", "65 cc 81 ff"
  ), from_hex, character(1), USE.NAMES = FALSE)
  expect_identical(
    codepoints(canon(ill_formed)),
    list(
      c(0x61, 0xFFFD, 0x62), 0xFFFD, c(0xFFFD, 0x61), rep(0xFFFD, 2),
      rep(0xFFFD, 3), rep(0xFFFD, 4), c(0xFFFD, 0x61), rep(0xFFFD, 2),
      rep(0xFFFD, 3), 0xFFFD, c(0xE9, 0xFFFD)
    )
  )

  bytes <- "\xc3\xa9"
  Encoding(bytes) <- "bytes"
  expect_error(canon(bytes), class = "fmridataset_error_identity")
})

test_that("canonical NFC matches the stringi/ICU normalizer it replaced", {
  skip_if_not_installed("stringi")
  reference <- function(x) {
    present <- !is.na(x)
    x[present] <- stringi::stri_trans_nfc(enc2utf8(x[present]))
    Encoding(x[present]) <- "UTF-8"
    x
  }
  cp <- c(1:0xD7FF, 0xE000:0x10FFFF)
  singles <- c(intToUtf8(cp, multiple = TRUE), NA)
  expect_identical(fmridataset:::.canonical_utf8(singles), reference(singles))

  withr::local_seed(20261018)
  pool <- as.raw(c(
    0x41, 0x61, 0x80:0xBF, 0xC0:0xC3, 0xCC, 0xCD, 0xDF, 0xE0:0xE2,
    0xEC:0xEF, 0xF0:0xF5, 0xFF
  ))
  random <- vapply(seq_len(5000), function(i) {
    value <- rawToChar(sample(pool, sample.int(10L, 1L), replace = TRUE))
    Encoding(value) <- "UTF-8"
    value
  }, character(1))
  marks <- c(0x41:0x5A, 0x300:0x36F, 0x1100:0x11FF, 0xAC00:0xAC40, 0x0F71:0x0F84)
  sequences <- vapply(seq_len(5000), function(i) {
    intToUtf8(sample(marks, sample.int(6L, 1L), replace = TRUE))
  }, character(1))
  values <- c(random, sequences)
  expect_identical(
    lapply(fmridataset:::.canonical_utf8(values), charToRaw),
    lapply(reference(values), charToRaw)
  )
})
