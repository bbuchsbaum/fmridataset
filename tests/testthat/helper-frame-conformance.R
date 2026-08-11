expect_array_source_conformance <- function(source, reference) {
  expect_s3_class(source, "array_source")
  expect_identical(source_shape(source), as.integer(dim(reference)))
  expect_length(source_dtype(source), 1)
  expect_length(source_chunks(source), 2)
  expect_true(all(source_chunks(source) > 0))
  expect_true("block_slice" %in% source_capabilities(source))
  expect_true(nzchar(source_fingerprint(source)))

  rows <- unique(c(nrow(reference), 1L))
  cols <- unique(c(ncol(reference), 1L))
  expect_equal(
    source_read(source, rows, cols),
    reference[rows, cols, drop = FALSE]
  )
  expect_identical(dim(source_read(source, integer(), cols)), c(0L, length(cols)))
  expect_identical(dim(source_read(source, rows, integer())), c(length(rows), 0L))
}

expect_feature_space_conformance <- function(space) {
  expect_s3_class(space, "feature_space")
  expect_identical(length(feature_ids(space)), n_features(space))
  expect_equal(anyDuplicated(feature_ids(space)), 0L)
  expect_false(anyNA(feature_ids(space)))
  expect_true(nzchar(space_digest(space)))
  expect_identical(nrow(feature_data(space)), n_features(space))

  selected <- rev(seq_len(n_features(space)))[seq_len(min(3L, n_features(space)))]
  restricted <- restrict_space(space, selected)
  expect_identical(feature_ids(restricted), feature_ids(space)[selected])
  expect_identical(n_features(restricted), length(selected))
}

contains_runtime_state <- function(x) {
  if (is.environment(x) || is.function(x) || typeof(x) == "externalptr") return(TRUE)
  if (is.pairlist(x)) return(any(vapply(as.list(x), contains_runtime_state, logical(1))))
  if (is.list(x)) return(any(vapply(unclass(x), contains_runtime_state, logical(1))))
  FALSE
}
