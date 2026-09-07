# Source fingerprints are cheap revision identities of descriptors; content
# hashes are explicit O(n) value identities. These tests pin the policy in
# ADR-009: neither is ever substituted for the other, and nothing hashes a
# payload unless asked.

test_that("memory_source construction never hashes the payload", {
  m <- matrix(as.double(seq_len(2000L * 200L)), 2000L, 200L)

  seen <- list()
  local_mocked_bindings(
    .content_hash_block_bytes = function(block, mode) stop("payload was hashed"),
    .canonical_digest = function(x) {
      seen[[length(seen) + 1L]] <<- names(x)
      canonical_sha256(x)
    },
    .package = "fmridataset"
  )
  x <- memory_source(m)

  expect_false(any(vapply(seen, function(fields) "data" %in% fields, logical(1))))
  expect_true(any(vapply(seen, function(fields) "identity" %in% fields, logical(1))))
  expect_null(x$content_digest)
  expect_match(x$identity, "^object:")
  expect_identical(x$identity_basis, "object")
  expect_error(memory_source(m, identity = "content"), "payload was hashed")
})

test_that("memory_source fingerprints are per object, not per value", {
  m <- matrix(1:12, 3, 4)
  a <- memory_source(m)
  b <- memory_source(m)

  expect_false(identical(source_fingerprint(a), source_fingerprint(b)))
  expect_identical(
    source_fingerprint(unserialize(serialize(a, NULL))),
    source_fingerprint(a)
  )
  expect_identical(content_hash(a), content_hash(b))

  revised <- memory_source(m, revision = "v2")
  expect_false(identical(source_fingerprint(revised), source_fingerprint(a)))
  expect_identical(revised$revision, "v2")
  expect_error(
    memory_source(m, revision = c("a", "b")),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    memory_source(m, revision = ""),
    class = "fmridataset_error_source_contract"
  )
})

test_that("memory_source identity = 'content' is the explicit opt-in", {
  m <- matrix(as.double(1:12), 3, 4)
  a <- memory_source(m, identity = "content")
  b <- memory_source(matrix(1:12, 3, 4), identity = "content")
  other <- memory_source(m + 1, identity = "content")

  expect_identical(source_fingerprint(a), source_fingerprint(b))
  expect_identical(a$identity, paste0("content:", content_hash(a)))
  expect_false(identical(source_fingerprint(a), source_fingerprint(other)))
  expect_false(identical(
    source_fingerprint(a),
    source_fingerprint(memory_source(m, identity = "content", chunks = c(1, 1)))
  ))
})

test_that("content hashes agree across equal values and differ otherwise", {
  m <- matrix(as.double(1:24), 6, 4)
  whole <- memory_source(m)
  chunked <- memory_source(m, chunks = c(2, 3))
  integer_valued <- memory_source(matrix(1:24, 6, 4))
  narrow <- memory_source(m, dtype = "float32")
  bound <- row_bound_source(list(
    memory_source(m[1:2, , drop = FALSE]),
    memory_source(m[3:3, , drop = FALSE]),
    memory_source(m[4:6, , drop = FALSE])
  ))
  sharded <- row_sharded_source(
    list(memory_source(m[1:4, , drop = FALSE]), memory_source(m[5:6, , drop = FALSE])),
    shard_ids = c("a", "b")
  )
  full_view <- source_view(whole, observations = 1:6, features = 1:4)

  reference <- content_hash(whole)
  expect_match(reference, "^[0-9a-f]{64}$")
  expect_identical(content_hash(chunked), reference)
  expect_identical(content_hash(integer_valued), reference)
  expect_identical(content_hash(narrow), reference)
  expect_identical(content_hash(bound), reference)
  expect_identical(content_hash(sharded), reference)
  expect_identical(content_hash(full_view), reference)
  expect_identical(content_hash(counting_source(whole)), reference)
  expect_identical(content_hash(m), reference)
  expect_identical(content_hash(source_open(whole)), reference)

  expect_false(identical(content_hash(memory_source(m + 1)), reference))
  expect_false(identical(content_hash(memory_source(t(m))), reference))
  expect_false(identical(content_hash(memory_source(matrix(m, 4, 6))), reference))

  reordered <- source_view(whole, observations = c(6, 1, 3), features = c(4, 2))
  expect_identical(
    content_hash(reordered),
    content_hash(memory_source(m[c(6, 1, 3), c(4, 2)]))
  )
  expect_false(identical(content_hash(reordered), reference))
})

test_that("content hashes distinguish modes and special values", {
  expect_false(identical(
    content_hash(memory_source(matrix(c(NA, 1, 2, 3), 2))),
    content_hash(memory_source(matrix(c(NaN, 1, 2, 3), 2)))
  ))
  expect_identical(
    content_hash(memory_source(matrix(c(0 / 0, 1, 2, 3), 2))),
    content_hash(memory_source(matrix(c(NaN, 1, 2, 3), 2)))
  )
  expect_identical(
    content_hash(memory_source(matrix(c(-(0 / 0), 1, 2, 3), 2))),
    content_hash(memory_source(matrix(c(NaN, 1, 2, 3), 2)))
  )
  expect_false(identical(
    content_hash(memory_source(matrix(c(-0, 1), 1))),
    content_hash(memory_source(matrix(c(0, 1), 1)))
  ))
  expect_false(identical(
    content_hash(memory_source(matrix(c(TRUE, FALSE), 1))),
    content_hash(memory_source(matrix(c(1, 0), 1)))
  ))
  expect_identical(
    content_hash(memory_source(matrix(c(TRUE, NA, FALSE, TRUE), 2))),
    content_hash(memory_source(matrix(c(TRUE, NA, FALSE, TRUE), 2), chunks = c(1, 1)))
  )
  z <- matrix(complex(real = 1:4, imaginary = 4:1), 2)
  expect_identical(
    content_hash(memory_source(z, dtype = "complex128")),
    content_hash(memory_source(z, dtype = "complex64", chunks = c(1, 2)))
  )
  expect_false(identical(
    content_hash(memory_source(z, dtype = "complex128")),
    content_hash(memory_source(Re(z)))
  ))
  expect_false(identical(
    content_hash(memory_source(matrix(numeric(), 0, 3))),
    content_hash(memory_source(matrix(numeric(), 3, 0)))
  ))
})

test_that("content hashing streams bounded, chunk-aligned blocks that touch each value once", {
  m <- matrix(as.double(seq_len(100)), 20, 5)
  x <- counting_source(memory_source(m, chunks = c(4, 5)))
  reference <- content_hash(memory_source(m))

  # 160 bytes holds 20 doubles: four whole rows, one observation chunk.
  expect_identical(content_hash(x, block_bytes = 160), reference)
  counts <- source_counts(x)
  expect_equal(counts$reads, 5)
  expect_equal(counts$values, 100)
  expect_equal(counts$opens, 1)
  expect_equal(counts$closes, 1)

  # 8 bytes holds one value: a row no longer fits, so feature ranges stream.
  reset_source_counts(x)
  expect_identical(content_hash(x, block_bytes = 8), reference)
  expect_equal(source_counts(x)$reads, 100)
  expect_equal(source_counts(x)$values, 100)

  # A generous bound reads everything at once and still agrees.
  reset_source_counts(x)
  expect_identical(content_hash(x, block_bytes = 1e6), reference)
  expect_equal(source_counts(x)$reads, 1)
  expect_equal(source_counts(x)$values, 100)

  expect_error(content_hash(x, block_bytes = 0), class = "fmridataset_error_budget")
})

test_that("content hashing crosses leaf boundaries independently of block size", {
  leaf <- content_hash_contract()$leaf_values
  n_col <- 7L
  n_row <- as.integer(ceiling(leaf * 1.5 / n_col)) + 3L
  m <- matrix(as.double(seq_len(n_row * n_col)) / 3, n_row, n_col)
  x <- memory_source(m)

  reference <- content_hash(x)
  expect_identical(content_hash(x, block_bytes = 8 * n_col * 11), reference)
  expect_identical(content_hash(x, block_bytes = 8 * n_col * leaf), reference)
  expect_identical(
    content_hash(row_bound_source(list(
      memory_source(m[seq_len(5000L), , drop = FALSE]),
      memory_source(m[-seq_len(5000L), , drop = FALSE])
    )), block_bytes = 8 * n_col * 999),
    reference
  )
})

test_that("row-bound content hashing streams shard by shard", {
  parts <- list(
    matrix(as.double(1:12), 3, 4),
    matrix(as.double(13:16), 1, 4),
    matrix(as.double(17:36), 5, 4)
  )
  children <- lapply(parts, function(part) counting_source(memory_source(part, chunks = c(2, 4))))
  bound <- row_bound_source(children)

  expect_identical(
    content_hash(bound, block_bytes = 8 * 4 * 2),
    content_hash(memory_source(do.call(rbind, parts)))
  )
  expect_equal(
    vapply(children, function(child) source_counts(child)$reads, numeric(1)),
    c(2, 1, 3)
  )
  expect_equal(
    vapply(children, function(child) source_counts(child)$values, numeric(1)),
    c(12, 4, 20)
  )
  expect_equal(vapply(children, function(child) source_counts(child)$opens, numeric(1)), c(1, 1, 1))
  expect_equal(vapply(children, function(child) source_counts(child)$closes, numeric(1)), c(1, 1, 1))
})

test_that("content hashes are receipts accepted by the identity contract", {
  x <- memory_source(matrix(as.double(1:6), 2, 3))
  receipt <- content_hash(x)
  identity <- identity_descriptor(x, domain = "content", content_digest = receipt)

  expect_identical(identity$domain, "content")
  expect_identical(identity$digest, receipt)
  expect_identical(identity$content_digest, receipt)
  expect_false(identical(identity_descriptor(x)$digest, receipt))
  expect_identical(content_hash_contract()$id, "org.fmridataset.content-hash/v1")
})

test_that("in-memory mutation is invisible to fingerprints and visible to content hashes", {
  x <- memory_source(matrix(as.double(1:6), 2, 3))
  before_fingerprint <- source_fingerprint(x)
  before_content <- content_hash(x)
  x$data[1L, 1L] <- 100

  expect_identical(source_fingerprint(x), before_fingerprint)
  expect_false(identical(content_hash(x), before_content))
})

test_that("sparse entity and lifted source fingerprints are cached at construction", {
  data <- Matrix::sparseMatrix(i = c(1, 3, 4), j = c(1, 2, 2), x = c(1.5, 2, 3), dims = c(4, 2))
  sparse <- fmridataset:::.sparse_entity_source(data)
  lifted <- fmridataset:::.row_index_source(sparse, c(4L, NA, 1L))
  view <- source_view(sparse, observations = c(2L, 4L))
  rebuilt <- fmridataset:::.sparse_entity_source(data)
  expected <- list(
    sparse = source_fingerprint(sparse),
    lifted = source_fingerprint(lifted),
    view = source_fingerprint(view)
  )

  # Deterministic across constructions: a resolved entity block lifts to the
  # same fingerprint every time it is rebuilt.
  expect_identical(source_fingerprint(rebuilt), expected$sparse)
  expect_identical(content_hash(sparse), content_hash(memory_source(as.matrix(data))))

  # And O(1) afterwards: no canonical digest is computed by any later call,
  # including through the wrappers that compose the sparse source.
  local_mocked_bindings(
    .canonical_digest = function(x) stop("fingerprint recomputed"),
    .package = "fmridataset"
  )
  for (i in 1:3) {
    expect_identical(source_fingerprint(sparse), expected$sparse)
    expect_identical(source_fingerprint(lifted), expected$lifted)
    expect_identical(source_fingerprint(view), expected$view)
  }
})
