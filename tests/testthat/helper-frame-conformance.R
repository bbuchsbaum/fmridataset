# Executable contracts every array source and every feature space must satisfy.
#
# These are the gates a new backend or space has to pass. They are deliberately
# stated as the behaviour the model promises rather than the behaviour any one
# implementation happens to have, so a type that cannot meet them fails loudly
# instead of quietly opting out.

expect_array_source_conformance <- function(source, reference) {
  expect_s3_class(source, "array_source")
  expect_invisible(validate_array_source(source))
  expect_identical(source_shape(source), as.integer(dim(reference)))

  descriptor <- source_descriptor(source)
  expect_identical(descriptor$shape, source_shape(source))
  expect_length(descriptor$dtype, 1)
  expect_true(descriptor$dtype %in% fmridataset:::.supported_source_dtypes)
  expect_length(source_chunks(source), 2)
  expect_true(all(source_chunks(source) > 0))
  expect_true(all(source_chunks(source) <= pmax(1L, source_shape(source))))
  expect_true("block_slice" %in% source_capabilities(source))
  expect_true("serializable" %in% source_capabilities(source))
  expect_equal(anyDuplicated(source_capabilities(source)), 0L)
  expect_true(nzchar(source_fingerprint(source)))
  expect_false(contains_runtime_state(source))

  # A descriptor must survive a serialization round trip whole, not just in its
  # fingerprint: the whole point of a serializable source is that it can be
  # reconstructed somewhere else.
  restored <- unserialize(serialize(source, NULL))
  expect_identical(source_fingerprint(restored), source_fingerprint(source))
  expect_identical(source_shape(restored), source_shape(source))
  expect_identical(source_dtype(restored), source_dtype(source))
  expect_identical(source_chunks(restored), source_chunks(source))

  # Lifecycle. Opening and closing must be safe even for sources that hold
  # nothing, and must not disturb the descriptor.
  handle <- source_open(source)
  expect_true(inherits(handle, "array_source_handle"))
  expect_silent(source_close(handle))
  expect_identical(source_fingerprint(source), descriptor$fingerprint)

  n_row <- nrow(reference)
  n_col <- ncol(reference)
  rows <- unique(c(n_row, 1L))
  cols <- unique(c(n_col, 1L))

  expect_equal(source_read(source, rows, cols), reference[rows, cols, drop = FALSE])
  expect_equal(
    source_read(source, rev(rows), rev(cols)),
    reference[rev(rows), rev(cols), drop = FALSE]
  )

  # Reads are pure: the same request twice gives the same answer, and the
  # answer does not depend on the shape of the request that produced it.
  expect_equal(source_read(source, rows, cols), source_read(source, rows, cols))
  if (n_row && n_col) {
    full <- source_read(source)
    expect_equal(full, unname(as.matrix(reference)), ignore_attr = TRUE)
    expect_equal(source_read(source, 1L, 1L)[1, 1], full[1, 1])
  }

  # An arbitrary permutation, not merely a reversal.
  if (n_row > 2L && n_col > 2L) {
    perm_rows <- c(2L, n_row, 1L)
    perm_cols <- c(n_col, 2L, 1L)
    expect_equal(
      source_read(source, perm_rows, perm_cols),
      reference[perm_rows, perm_cols, drop = FALSE]
    )
  }

  # One normalization law for every axis in the package: an element appears
  # at most once in a selection, on raw sources exactly as on frames, so the
  # positional and the ID-bearing layers can never disagree about a read.
  if (n_row && n_col) {
    expect_error(
      source_read(source, c(1L, 1L), cols),
      class = "fmridataset_error_alignment"
    )
    expect_error(
      source_read(source, rows, c(1L, 1L)),
      class = "fmridataset_error_alignment"
    )
  }

  # Views compose. A range, an arbitrary set of positions, and a select-all
  # nested in any order read exactly what the equivalent flat positions read,
  # and the nesting collapses to one view over the root source.
  if (n_row > 2L && n_col > 2L) {
    inner_rows <- 2:n_row
    inner_cols <- seq_len(n_col - 1L)
    ranged <- source_view(source, observations = inner_rows, features = inner_cols)
    expect_equal(source_read(ranged), reference[inner_rows, inner_cols, drop = FALSE])

    picked <- source_view(ranged, observations = c(n_row - 1L, 1L))
    picked_rows <- inner_rows[c(n_row - 1L, 1L)]
    expect_false(inherits(picked$source, "source_view"))
    expect_equal(source_read(picked), reference[picked_rows, inner_cols, drop = FALSE])

    everything <- source_view(picked)
    expect_identical(source_fingerprint(everything), source_fingerprint(picked))
    expect_equal(source_read(everything), source_read(picked))

    deeper <- source_view(everything, observations = 2:1, features = c(n_col - 1L, 1L))
    expect_false(inherits(deeper$source, "source_view"))
    expect_equal(
      source_read(deeper),
      reference[picked_rows[2:1], c(n_col - 1L, 1L), drop = FALSE]
    )
    expect_equal(
      source_read(deeper, 1L, 2L),
      reference[picked_rows[[2L]], 1L, drop = FALSE]
    )
    expect_identical(
      source_fingerprint(deeper),
      source_fingerprint(source_view(
        source,
        observations = picked_rows[2:1], features = c(n_col - 1L, 1L)
      ))
    )
    expect_identical(dim(source_read(source_view(deeper, observations = integer()))), c(0L, 2L))
  }

  # Empty selections on either axis, and on both.
  expect_identical(dim(source_read(source, integer(), cols)), c(0L, length(cols)))
  expect_identical(dim(source_read(source, rows, integer())), c(length(rows), 0L))
  expect_identical(dim(source_read(source, integer(), integer())), c(0L, 0L))

  # Out-of-range selections are refused rather than recycled or truncated.
  expect_error(source_read(source, n_row + 1L, 1L))
  expect_error(source_read(source, 1L, n_col + 1L))

  # Content identity is a property of the values, not of the source that
  # delivers them: every conforming source hashes exactly like an in-memory
  # copy of its reference, and the result is a legal content receipt.
  # The fingerprint stays untouched by the O(n) read.
  content <- content_hash(source)
  expect_identical(content, content_hash(memory_source(unname(as.matrix(reference)))))
  expect_identical(
    identity_descriptor(source, domain = "content", content_digest = content)$digest,
    content
  )
  expect_identical(source_fingerprint(source), descriptor$fingerprint)
}

expect_feature_space_conformance <- function(space) {
  expect_s3_class(space, "feature_space")
  n <- n_features(space)

  expect_identical(length(feature_ids(space)), n)
  expect_equal(anyDuplicated(feature_ids(space)), 0L)
  expect_false(anyNA(feature_ids(space)))
  expect_true(all(nzchar(feature_ids(space))))
  expect_true(nzchar(space_digest(space)))
  expect_false(is.null(native_shape(space)))

  # Feature metadata is one row per feature and agrees with the IDs.
  data <- feature_data(space)
  expect_identical(nrow(data), n)
  if (".feature_id" %in% names(data)) {
    expect_identical(as.character(data$.feature_id), feature_ids(space))
  }

  # Identity is a property of the space, not of the object: an identical
  # construction and a serialization round trip must both agree.
  expect_identical(space_digest(unserialize(serialize(space, NULL))), space_digest(space))

  selected <- rev(seq_len(n))[seq_len(min(3L, n))]
  restricted <- restrict_space(space, selected)
  expect_identical(feature_ids(restricted), feature_ids(space)[selected])
  expect_identical(n_features(restricted), length(selected))

  # Restriction composes: narrowing twice equals narrowing once by the
  # composed index.
  if (length(selected) > 1L) {
    inner <- c(2L, 1L)
    expect_identical(
      feature_ids(restrict_space(restricted, inner)),
      feature_ids(space)[selected[inner]]
    )
  }

  # Restricting to everything, in order, is an identity on IDs.
  if (n) {
    expect_identical(feature_ids(restrict_space(space, seq_len(n))), feature_ids(space))
  }

  # An empty restriction is a legal space with no features, not an error and
  # not a length-one recycling artefact.
  empty <- restrict_space(space, integer(0))
  expect_identical(n_features(empty), 0L)
  expect_identical(length(feature_ids(empty)), 0L)
  expect_identical(nrow(feature_data(empty)), 0L)
  expect_true(nzchar(space_digest(empty)))
}

contains_runtime_state <- function(x) {
  if (is.environment(x) || is.function(x) || typeof(x) == "externalptr") {
    return(TRUE)
  }
  if (is.pairlist(x)) {
    return(any(vapply(as.list(x), contains_runtime_state, logical(1))))
  }
  if (is.list(x)) {
    return(any(vapply(unclass(x), contains_runtime_state, logical(1))))
  }
  FALSE
}
