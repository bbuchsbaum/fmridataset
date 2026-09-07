# Regression tests for the review findings fixed together on the
# fix/review-findings branch. Each block names the finding it pins.

.review_second_frame <- function(fx, offset = 1) {
  fr <- fx$frame
  obs <- observations(fr)
  obs$.obs_id <- sprintf("other-%03d", seq_len(nrow(obs)))
  fmri_frame(
    assays = list(
      beta = memory_source(fx$beta + offset),
      variance = memory_source(fx$variance + offset)
    ),
    observations = axis_frame(obs, blocks = obs_blocks(fr)),
    features = feature_axis(fr),
    entities = entities(fr),
    active_assay = "beta"
  )
}

# 1. bind_observations() compares entity registries semantically ------------

test_that("frames reopened from FDS bind with each other and with in-memory frames", {
  testthat::skip_if_not_installed("fmristore")
  fx <- make_frame_fixture()
  fr <- fx$frame
  fr2 <- .review_second_frame(fx)

  p1 <- file.path(tempfile("review-bind-a"), "a.h5")
  p2 <- file.path(tempfile("review-bind-b"), "b.h5")
  dir.create(dirname(p1))
  dir.create(dirname(p2))
  r1 <- open_frame(write_frame(fr, p1))
  r2 <- open_frame(write_frame(fr2, p2))

  # The reopened entity block is an array source with a per-file fingerprint,
  # so the registry digests differ although every value agrees.
  expect_s3_class(
    axis_block_data(entity_blocks(entity(r1, "stimulus"))$visual_pca),
    "array_source"
  )
  expect_false(identical(entity_registry_digest(r1), entity_registry_digest(r2)))

  both <- bind_observations(r1, r2)
  expect_s3_class(both, "fmri_frame")
  expect_identical(nrow(both), 14L)
  expect_s3_class(bind_observations(r1, fr2), "fmri_frame")
  expect_s3_class(bind_observations(fr, r2), "fmri_frame")
  expect_equal(
    unname(collect_assay(both)),
    unname(rbind(fx$beta, fx$beta + 1))
  )
})

test_that("bind_observations() still refuses genuinely different entity registries", {
  fx <- make_frame_fixture()
  fr <- fx$frame
  other <- .review_second_frame(fx)

  changed <- entity(fr, "stimulus")
  changed_data <- entity_data(changed)
  changed_data$category[[1L]] <- "house"
  different_data <- entity_frame(
    changed_data,
    key = "stimulus_id", blocks = entity_blocks(changed)
  )
  other_data <- fmri_frame(
    assays = assays(other), observations = observation_axis(other),
    features = feature_axis(other), entities = list(stimulus = different_data)
  )
  err <- expect_error(
    bind_observations(fr, other_data),
    class = "fmridataset_error_entity"
  )
  expect_match(conditionMessage(err), "incompatible entity registries")
  expect_match(conditionMessage(err), "entity data")

  block <- entity_blocks(changed)$visual_pca
  values <- axis_block_data(block)
  values[1L, 1L] <- values[1L, 1L] + 1
  different_values <- entity_frame(
    entity_data(changed),
    key = "stimulus_id",
    blocks = list(visual_pca = axis_block(
      memory_source(values),
      components = block_components(block), role = block$role
    ))
  )
  other_values <- fmri_frame(
    assays = assays(other), observations = observation_axis(other),
    features = feature_axis(other), entities = list(stimulus = different_values)
  )
  err <- expect_error(
    bind_observations(fr, other_values),
    class = "fmridataset_error_entity"
  )
  expect_match(conditionMessage(err), "different values")

  missing_entity <- fmri_frame(
    assays = assays(other), observations = observation_axis(other),
    features = feature_axis(other)
  )
  expect_error(
    bind_observations(fr, missing_entity),
    class = "fmridataset_error_entity"
  )
})

# 2. Wrapper fingerprints are computed once, at construction -----------------

.fingerprint_calls <- new.env(parent = emptyenv())
.fingerprint_calls$n <- 0L

local({
  ns <- asNamespace("fmridataset")
  cls <- "fingerprint_counting_source"
  registerS3method("source_shape", cls, function(x, ...) source_shape(x$source), envir = ns)
  registerS3method("source_dtype", cls, function(x, ...) source_dtype(x$source), envir = ns)
  registerS3method("source_chunks", cls, function(x, ...) source_chunks(x$source), envir = ns)
  registerS3method(
    "source_capabilities", cls,
    function(x, ...) source_capabilities(x$source),
    envir = ns
  )
  registerS3method("source_fingerprint", cls, function(x, ...) {
    .fingerprint_calls$n <- .fingerprint_calls$n + 1L
    source_fingerprint(x$source)
  }, envir = ns)
  registerS3method("source_open", cls, function(x, ...) source_open(x$source, ...), envir = ns)
  registerS3method("source_close", cls, function(x, ...) source_close(x$source, ...), envir = ns)
  registerS3method(
    "source_read", cls,
    function(x, observations = NULL, features = NULL, ...) {
      source_read(x$source, observations = observations, features = features, ...)
    },
    envir = ns
  )
})

.fingerprint_counting_source <- function(source) {
  structure(
    list(source = as_array_source(source)),
    class = c("fingerprint_counting_source", "array_source")
  )
}

.expect_fingerprint_cached <- function(wrapper) {
  .fingerprint_calls$n <- 0L
  first <- source_fingerprint(wrapper)
  for (i in 1:4) expect_identical(source_fingerprint(wrapper), first)
  expect_identical(.fingerprint_calls$n, 0L)
  expect_identical(wrapper$fingerprint, first)
  # The stored value is what the method would compute from the descriptor.
  fresh <- wrapper
  fresh$fingerprint <- NULL
  expect_identical(source_fingerprint(fresh), first)
  # Serialization preserves the cached fingerprint without recomputing it.
  .fingerprint_calls$n <- 0L
  restored <- unserialize(serialize(wrapper, NULL))
  expect_identical(restored$fingerprint, first)
  expect_identical(source_fingerprint(restored), first)
  expect_identical(.fingerprint_calls$n, 0L)
  invisible(first)
}

test_that("feature_mapped_source() fingerprints once at construction", {
  child <- .fingerprint_counting_source(memory_source(matrix(1:12, 3, 4)))
  from <- index_space(4L, ids = sprintf("f%d", 1:4))
  to <- index_space(2L, ids = c("c1", "c2"))
  map <- feature_map(from, to, matrix(1, 2, 4))
  .fingerprint_calls$n <- 0L
  wrapper <- feature_mapped_source(child, map)
  expect_lte(.fingerprint_calls$n, 2L)
  .expect_fingerprint_cached(wrapper)
})

test_that("validity_masked_source() fingerprints once at construction", {
  child <- .fingerprint_counting_source(memory_source(matrix(1:12, 3, 4)))
  from <- index_space(4L, ids = sprintf("f%d", 1:4))
  bank <- mask_bank(matrix(c(TRUE, TRUE, FALSE, TRUE), 1L, 4L), from)
  .fingerprint_calls$n <- 0L
  wrapper <- validity_masked_source(child, rep(bank$mask_ids[[1L]], 3L), bank)
  expect_lte(.fingerprint_calls$n, 2L)
  .expect_fingerprint_cached(wrapper)
})

test_that("fault_source() and row_sharded_source() fingerprint once at construction", {
  child <- .fingerprint_counting_source(memory_source(matrix(1:12, 3, 4)))
  .fingerprint_calls$n <- 0L
  faulty <- fault_source(child, stage = "read")
  expect_lte(.fingerprint_calls$n, 2L)
  .expect_fingerprint_cached(faulty)

  .fingerprint_calls$n <- 0L
  sharded <- row_sharded_source(list(child, child), shard_ids = c("a", "b"))
  expect_lte(.fingerprint_calls$n, 4L)
  .expect_fingerprint_cached(sharded)

  bound <- row_bound_source(list(child, child))
  .expect_fingerprint_cached(bound)
  # row_bound_source() assigns the deterministic default shard IDs.
  expect_identical(
    source_fingerprint(bound),
    source_fingerprint(row_sharded_source(list(child, child)))
  )
})

# 3. validate_array_source() resolves methods from the caller's scope ---------

test_that("an extension class whose methods live in the calling scope validates", {
  source_shape.review_local_source <- function(x, ...) x$shape
  source_dtype.review_local_source <- function(x, ...) "float64"
  source_chunks.review_local_source <- function(x, ...) x$shape
  source_capabilities.review_local_source <- function(x, ...) {
    c("block_slice", "serializable")
  }
  source_fingerprint.review_local_source <- function(x, ...) "review-local"
  source_open.review_local_source <- function(x, ...) {
    structure(list(source = x), class = "array_source_handle")
  }
  source_close.review_local_source <- function(x, ...) invisible(TRUE)
  source_read.review_local_source <- function(x, observations = NULL, features = NULL, ...) {
    matrix(0, 2, 2)
  }
  s <- structure(list(shape = c(2L, 2L)), class = c("review_local_source", "array_source"))
  expect_identical(source_shape(s), c(2L, 2L))
  expect_identical(validate_array_source(s), s)

  # A descriptor that violates the contract is still reported as such, with
  # the method resolved from this scope rather than reported as missing.
  bad <- structure(list(shape = c(2L, -1L)), class = c("review_local_source", "array_source"))
  err <- expect_error(validate_array_source(bad), class = "fmridataset_error_source_contract")
  expect_identical(err$field, "shape")
})

test_that("a class with no protocol methods is still named as implementing nothing", {
  bogus <- structure(list(), class = c("review_bogus_source", "array_source"))
  err <- expect_error(validate_array_source(bogus), class = "fmridataset_error_source_contract")
  expect_identical(err$field, "methods")
  expect_setequal(err$missing, c(
    "source_shape", "source_dtype", "source_chunks", "source_capabilities",
    "source_fingerprint", "source_open", "source_read", "source_close"
  ))
  expect_match(conditionMessage(err), "review_bogus_source")
})

# 4a. Vectorized canonical string encoding is byte-for-byte unchanged ---------

test_that("vectorized canonical string encoding matches the per-element reference", {
  reference <- function(x) {
    unlist(lapply(x, function(value) {
      if (is.na(value)) {
        return(charToRaw("0"))
      }
      bytes <- charToRaw(enc2utf8(value))
      c(charToRaw("1"), writeBin(length(bytes), raw(), size = 4L, endian = "big"), bytes)
    }), use.names = FALSE)
  }
  cases <- list(
    NA_character_,
    c(NA_character_, NA_character_),
    "",
    c("", NA, "a"),
    c("café", "é", NA_character_),
    c("中文", "x", NA, ""),
    sprintf("feature-%06d", seq_len(2000L))
  )
  for (value in cases) {
    expect_identical(.canonical_string_vector_bytes(value), reference(value))
  }
  expect_identical(.canonical_string_vector_bytes(character()), raw())
  expect_identical(
    canonical_bytes(c("a", NA, "b")),
    c(
      charToRaw(paste0(canonicalization_contract()$id, "\n")),
      charToRaw("c"), as.raw(c(0, 0, 0, 3)), reference(c("a", NA, "b")),
      charToRaw("A"), as.raw(c(0, 0, 0, 0))
    )
  )
})

# 4b. Axis frames cache the digest of their IDs ------------------------------

test_that("axis frames carry their ID digest and subsets recompute it", {
  axis <- axis_frame(data.frame(.obs_id = c("a", "b", "c"), value = 1:3))
  expect_identical(axis$id_digest$n, 3L)
  expect_identical(axis$id_digest$sha256, .canonical_digest(c("a", "b", "c")))
  expect_identical(.axis_digest(axis), .canonical_digest(c("a", "b", "c")))

  subset <- axis[c(3L, 1L)]
  expect_identical(subset$id_digest$n, 2L)
  expect_identical(.axis_digest(subset), .canonical_digest(c("c", "a")))

  # A cached value that no longer describes the axis is ignored.
  stale <- axis
  stale$id_digest$n <- 5L
  expect_identical(.axis_digest(stale), .canonical_digest(c("a", "b", "c")))
  stale$id_digest <- NULL
  expect_identical(.axis_digest(stale), .canonical_digest(c("a", "b", "c")))

  space <- index_space(3L, ids = c("x", "y", "z"))
  features <- feature_axis(feature_data(space), space = space)
  expect_identical(.axis_digest(features), .canonical_digest(c("x", "y", "z")))
  entity <- entity_frame(data.frame(k = c("e1", "e2")), key = "k")
  expect_identical(.axis_digest(entity), .canonical_digest(c("e1", "e2")))
})

test_that("view assay descriptors carry the digests of the visible axes", {
  fx <- make_frame_fixture()
  view <- fx$frame[c(2L, 5L), c(6L, 1L)]
  descriptor <- assay(view)
  expect_identical(descriptor$observation_digest, .canonical_digest(observation_ids(view)))
  expect_identical(descriptor$feature_digest, .canonical_digest(feature_ids(view)))
  whole <- assay(fx$frame[, ])
  expect_identical(whole$feature_digest, .canonical_digest(feature_ids(fx$frame)))
})

# 5. explain() inspects ephemeral frames; identity_descriptor() refuses ------

test_that("explain() reports durability instead of aborting on ephemeral IDs", {
  ephemeral <- fmri_frame(
    list(a = memory_source(matrix(1:4, 2, 2))),
    observations = data.frame(.obs_id = c("a", "b"))
  )
  summary <- explain(ephemeral)
  expect_false(summary$ids_durable)
  expect_null(summary$digests$semantic)
  expect_type(summary$digests$schema, "character")
  expect_identical(summary$digests$observation, .canonical_digest(c("a", "b")))

  durable <- fmri_frame(
    list(a = memory_source(matrix(1:4, 2, 2))),
    observations = data.frame(.obs_id = c("a", "b")),
    space = index_space(2L, ids = c("x", "y"))
  )
  summary <- explain(durable)
  expect_true(summary$ids_durable)
  expect_identical(summary$digests$semantic, fds_manifest_digest(fds_frame_manifest(durable)))

  err <- expect_error(
    identity_descriptor(ephemeral, domain = "semantic"),
    class = "fmridataset_error_identity"
  )
  expect_identical(err$axes, "feature")
  expect_match(conditionMessage(err), "feature axis carries ephemeral IDs")
  expect_match(conditionMessage(err), "space = ")
  expect_error(identity_descriptor(ephemeral), class = "fmridataset_error_identity")

  ephemeral_obs <- fmri_frame(
    list(a = memory_source(matrix(1:4, 2, 2))),
    observations = axis_frame(data.frame(value = 1:2), id_policy = "ephemeral"),
    space = index_space(2L, ids = c("x", "y"))
  )
  expect_false(explain(ephemeral_obs)$ids_durable)
  err <- expect_error(
    identity_descriptor(ephemeral_obs, domain = "semantic"),
    class = "fmridataset_error_identity"
  )
  expect_identical(err$axes, "observation")
  expect_match(conditionMessage(err), "observation axis carries ephemeral IDs")
})

# 6. Runtime state is rejected at construction --------------------------------

test_that("axis, block, and assay metadata reject runtime state at construction", {
  err <- expect_error(
    axis_frame(data.frame(.obs_id = c("a", "b")), metadata = list(f = function() 1)),
    class = "fmridataset_error_alignment"
  )
  expect_identical(err$field, "metadata")
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", "b")), metadata = list(e = new.env())),
    class = "fmridataset_error_alignment"
  )

  err <- expect_error(
    axis_block(matrix(1:4, 2, 2), metadata = list(f = function() 1)),
    class = "fmridataset_error_alignment"
  )
  expect_identical(err$field, "metadata")
  expect_error(
    axis_block(matrix(1:4, 2, 2), metadata = list(e = new.env())),
    class = "fmridataset_error_alignment"
  )

  annotated <- structure(
    list(source = memory_source(matrix(1:4, 2, 2)), metadata = list(e = new.env())),
    class = "aligned_assay"
  )
  err <- expect_error(
    fmri_frame(
      list(a = annotated),
      observations = data.frame(.obs_id = c("a", "b")),
      space = index_space(2L, ids = c("x", "y"))
    ),
    class = "fmridataset_error_alignment"
  )
  expect_identical(err$field, "metadata")
  expect_identical(err$assay, "a")

  # Entity frames keep their own error class for the same rule.
  expect_error(
    entity_frame(data.frame(k = "a"), key = "k", metadata = list(f = function() 1)),
    class = "fmridataset_error_entity"
  )
  # A block is validated by its own constructor before any entity frame sees it.
  expect_error(
    axis_block(matrix(1, 1, 1), metadata = list(f = function() 1)),
    class = "fmridataset_error_alignment"
  )
  # Serializable metadata still passes.
  expect_s3_class(
    axis_block(matrix(1:4, 2, 2), metadata = list(note = "ok")),
    "axis_block"
  )
})

# 7. Every selection over an empty axis has one canonical form ---------------

test_that("an empty axis has one canonical selection form", {
  expect_identical(.normalize_selection(NULL, 0L)$form, "all")
  expect_identical(.normalize_selection(integer(), 0L)$form, "all")
  expect_identical(.normalize_selection(logical(), 0L)$form, "all")
  expect_identical(.normalize_selection(NULL, 0L), .normalize_selection(integer(), 0L))
  expect_identical(.selection_range(0L, 1L, 0L)$form, "all")
  # The empty selection on a non-empty axis stays explicit.
  expect_identical(.normalize_selection(integer(), 3L)$form, "positions")
  expect_length(.selection_expand(.normalize_selection(NULL, 0L)), 0L)

  fz <- fmri_frame(
    list(a = memory_source(matrix(numeric(), 0, 3))),
    observations = data.frame(.obs_id = character()),
    space = index_space(3L, ids = c("a", "b", "c"))
  )
  plan <- plan_blocks(fz)
  expect_identical(.frame_plan_fingerprint(fz, "a"), .frame_plan_fingerprint(fz[integer(), ], "a"))
  expect_identical(
    source_fingerprint(assay(fz[integer(), ])$source),
    source_fingerprint(assay(fz[, ])$source)
  )
  calls <- 0L
  expect_no_error(
    execute_block_plan(fz[integer(), ], plan, function(values, ...) calls <<- calls + 1L)
  )
  expect_identical(calls, 0L)
  expect_identical(dim(collect_assay(fz[integer(), ])), c(0L, 3L))
})

# 8. Non-finite and out-of-range numeric selectors are rejected ---------------

test_that("non-finite and out-of-range numeric selectors fail with structured reasons", {
  for (value in list(Inf, -Inf, c(1, Inf))) {
    err <- expect_error(.normalize_selection(value, 5L), class = "fmridataset_error_alignment")
    expect_identical(err$reason, "non_finite")
  }
  for (value in list(3e9, -3e9, c(1, 1e10))) {
    err <- expect_error(.normalize_selection(value, 5L), class = "fmridataset_error_alignment")
    expect_identical(err$reason, "out_of_bounds")
    expect_identical(err$axis_length, 5L)
  }
  # The integer boundary itself is still a legal position.
  expect_identical(
    .selection_expand(.normalize_selection(2147483647, .Machine$integer.max)),
    .Machine$integer.max
  )

  fx <- make_frame_fixture()
  err <- expect_error(fx$frame[Inf, ], class = "fmridataset_error_alignment")
  expect_identical(err$reason, "non_finite")
  err <- expect_error(fx$frame[, 3e9], class = "fmridataset_error_alignment")
  expect_identical(err$reason, "out_of_bounds")
  err <- expect_error(
    source_read(memory_source(matrix(1:4, 2, 2)), observations = -Inf),
    class = "fmridataset_error_alignment"
  )
  expect_identical(err$reason, "non_finite")
})

# 10. Factor columns in bound typed tables ------------------------------------

test_that("bound typed tables compare factor cells under one declared level set", {
  fx <- make_frame_fixture()
  fr <- fx$frame
  obs2 <- observations(fr)
  obs2$.obs_id <- sprintf("other-%03d", seq_len(nrow(obs2)))
  with_table <- function(observations, table) {
    fmri_frame(
      list(beta = memory_source(fx$beta)),
      observations = observations,
      features = feature_axis(fr), tables = list(contrasts = table)
    )
  }
  levels_ab <- c("a", "b")
  first <- with_table(observations(fr), auxiliary_table(
    data.frame(contrast_id = c("c1", "c2"), grp = factor(c("a", "b"), levels = levels_ab)),
    key = "contrast_id", role = "contrasts"
  ))
  # Equal labels under the same level set merge, whatever the row order.
  same_levels <- with_table(axis_frame(obs2), auxiliary_table(
    data.frame(contrast_id = c("c3", "c2"), grp = factor(c("a", "b"), levels = levels_ab)),
    key = "contrast_id", role = "contrasts"
  ))
  merged <- table_data(bind_observations(first, same_levels)$tables$contrasts)
  expect_identical(as.character(merged$contrast_id), c("c1", "c2", "c3"))
  expect_identical(as.character(merged$grp), c("a", "b", "a"))
  expect_identical(levels(merged$grp), levels_ab)

  # A different level set is a schema difference, refused before the merge.
  other_levels <- with_table(axis_frame(obs2), auxiliary_table(
    data.frame(contrast_id = c("c2", "c3"), grp = factor(c("b", "c"), levels = c("b", "c"))),
    key = "contrast_id", role = "contrasts"
  ))
  err <- expect_error(bind_observations(first, other_levels), class = "fmridataset_error_schema")
  expect_match(conditionMessage(err), "grp")

  # Equal level sets with a genuinely different label for a shared key conflict.
  conflicting <- with_table(axis_frame(obs2), auxiliary_table(
    data.frame(contrast_id = c("c2", "c3"), grp = factor(c("a", "a"), levels = levels_ab)),
    key = "contrast_id", role = "contrasts"
  ))
  err <- expect_error(bind_observations(first, conflicting), class = "fmridataset_error_table")
  expect_identical(err$keys, "c2")
})
