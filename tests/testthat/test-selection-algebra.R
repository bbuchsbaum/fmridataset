# The selection algebra (inst/architecture/ADR-010-selection-algebra.md).
#
# One normalization law for every axis in the package, one compact normalized
# form stored by views and hashed by fingerprints and plans, and composition of
# nested selections without materializing the axis they pass through.

# ------------------------------------------------------------ the one law

# Every case is stated once, parameterized by the axis length and its IDs, and
# run against every owner of a selectable axis. `positions` is the expected
# resolved selection; `reason` is the structured failure the owner must raise.
.selection_law_cases <- function(n, ids) {
  list(
    list(name = "character IDs resolve in request order", selector = ids[c(3L, 1L)], positions = c(3L, 1L)),
    list(name = "logical masks keep axis order", selector = seq_len(n) %in% c(1L, 3L), positions = c(1L, 3L)),
    list(name = "numeric positions may reorder", selector = c(3, 1), positions = c(3L, 1L)),
    list(name = "whole doubles are positions", selector = c(2, 3), positions = 2:3),
    list(name = "negative positions exclude", selector = -1, positions = seq_len(n)[-1L]),
    list(name = "zero is dropped", selector = c(0, 2, 0), positions = 2L),
    list(name = "empty is a legal zero-length axis", selector = integer(), positions = integer()),
    list(name = "unknown IDs are rejected", selector = c(ids[[1L]], "no-such-id"), reason = "unknown_id"),
    list(name = "repeated IDs are rejected", selector = ids[c(1L, 1L)], reason = "duplicate"),
    list(name = "missing IDs are rejected", selector = c(ids[[1L]], NA_character_), reason = "missing"),
    list(name = "logical length must match", selector = c(TRUE, FALSE), reason = "length"),
    list(name = "logical NA is rejected", selector = c(TRUE, NA, rep(FALSE, n - 2L)), reason = "missing"),
    list(name = "fractional positions are rejected", selector = 1.5, reason = "non_integer"),
    list(name = "missing positions are rejected", selector = c(1L, NA_integer_), reason = "missing"),
    list(name = "repeated positions are rejected", selector = c(1, 1), reason = "duplicate"),
    list(name = "mixed signs are rejected", selector = c(-1, 2), reason = "mixed_sign"),
    list(name = "positions past the axis are rejected", selector = n + 1, reason = "out_of_bounds"),
    list(name = "negative positions past the axis are rejected", selector = -(n + 1), reason = "out_of_bounds"),
    list(name = "unsupported selector types are rejected", selector = list(1), reason = "unsupported_type")
  )
}

.expect_selection_law <- function(owner) {
  cases <- .selection_law_cases(owner$n, owner$ids %||% sprintf("positional-%d", seq_len(owner$n)))
  for (case in cases) {
    label <- sprintf("%s: %s", owner$name, case$name)
    expected_reason <- case$reason
    if (is.null(owner$ids) && is.character(case$selector)) {
      # A raw source has no IDs; character selectors are refused for that
      # reason before any other rule applies.
      expected_reason <- "positional_axis"
    }
    if (is.null(expected_reason) && !length(case$positions) && !is.null(owner$empty_reason)) {
      expected_reason <- owner$empty_reason
    }
    if (is.null(expected_reason)) {
      expect_identical(owner$select(case$selector), case$positions, label = label)
    } else {
      err <- expect_error(owner$select(case$selector), class = owner$error_class, label = label)
      expect_identical(err$reason, expected_reason, label = label)
    }
  }
}

test_that("frames, views, sources, collections, axes, and entities share one normalization law", {
  fx <- make_frame_fixture()
  x <- fx$frame
  obs_ids <- observation_ids(x)
  feat_ids <- feature_ids(x)
  positions_matrix <- matrix(seq_len(7L), ncol = 1L)
  source <- memory_source(positions_matrix)
  stimulus <- entities(x)$stimulus
  collection <- fmri_collection(list(a = x, b = x[1:3, ], c = x[4:7, ]))
  view <- x[2:7, ]

  owners <- list(
    list(
      name = "frame observations", n = 7L, ids = obs_ids,
      error_class = "fmridataset_error_alignment",
      select = function(i) match(observation_ids(x[i, ]), obs_ids)
    ),
    list(
      name = "frame features", n = 6L, ids = feat_ids,
      error_class = "fmridataset_error_alignment",
      select = function(j) match(feature_ids(x[, j]), feat_ids)
    ),
    list(
      name = "nested view observations", n = 6L, ids = obs_ids[2:7],
      error_class = "fmridataset_error_alignment",
      select = function(i) match(observation_ids(view[i, ]), obs_ids[2:7])
    ),
    list(
      name = "source read", n = 7L, ids = NULL,
      error_class = "fmridataset_error_alignment",
      select = function(i) as.integer(source_read(source, i)[, 1L])
    ),
    list(
      name = "source view", n = 7L, ids = NULL,
      error_class = "fmridataset_error_alignment",
      select = function(i) as.integer(source_read(source_view(source, observations = i))[, 1L])
    ),
    list(
      name = "collection frames", n = 3L, ids = c("a", "b", "c"),
      error_class = "fmridataset_error_collection",
      empty_reason = "empty_collection",
      select = function(i) match(collection_ids(collection[i]), c("a", "b", "c"))
    ),
    list(
      name = "entity frame", n = 3L, ids = entity_ids(stimulus),
      error_class = "fmridataset_error_alignment",
      select = function(i) match(entity_ids(stimulus[i]), entity_ids(stimulus))
    ),
    list(
      name = "axis frame", n = 7L, ids = obs_ids,
      error_class = "fmridataset_error_alignment",
      select = function(i) match(axis_ids(observation_axis(x)[i]), obs_ids)
    )
  )
  for (owner in owners) .expect_selection_law(owner)
})

test_that("study entity filters restrict the registry through the same law", {
  fx <- make_frame_fixture()
  study <- fmri_study(frames = list(main = fx$frame), entities = entities(fx$frame))

  filtered <- filter_entities(study, stimulus, category != "scene")
  kept <- entities(filtered)$stimulus

  expect_identical(entity_ids(kept), c("stim-1", "stim-3"))
  expect_identical(entity_data(kept)$category, c("face", "object"))
  expect_identical(nrow(axis_block_data(entity_blocks(kept)$visual_pca)), 2L)
})

# ------------------------------------------------------- normalized forms

test_that("selections canonicalize to all, range, or positions", {
  norm <- fmridataset:::.normalize_selection

  expect_identical(norm(NULL, 5L)$form, "all")
  expect_identical(norm(1:5, 5L)$form, "all")
  expect_identical(norm(rep(TRUE, 5L), 5L)$form, "all")
  expect_identical(norm(c(0, 1, 2, 3, 4, 5), 5L)$form, "all")

  ranged <- norm(2:4, 5L)
  expect_identical(ranged$form, "range")
  expect_identical(c(ranged$start, ranged$end), c(2L, 4L))
  expect_identical(norm(-c(1, 5), 5L), ranged)
  expect_identical(norm(c(FALSE, TRUE, TRUE, TRUE, FALSE), 5L), ranged)
  expect_identical(norm(3, 5L)$form, "range")

  picked <- norm(c(4, 2), 5L)
  expect_identical(picked$form, "positions")
  expect_identical(picked$positions, c(4L, 2L))
  expect_identical(norm(integer(), 5L)$form, "positions")
  expect_identical(fmridataset:::.selection_length(norm(integer(), 5L)), 0L)

  expect_identical(norm(c("b", "a"), 3L, ids = c("a", "b", "c"))$positions, c(2L, 1L))
  expect_identical(norm(c("a", "b", "c"), 3L, ids = c("a", "b", "c"))$form, "all")

  # A normalized selection passes through untouched, and must address its axis.
  expect_identical(norm(ranged, 5L), ranged)
  err <- expect_error(norm(ranged, 6L), class = "fmridataset_error_alignment")
  expect_identical(err$reason, "axis_length")
})

test_that("composition never materializes all or range axes", {
  norm <- fmridataset:::.normalize_selection
  compose <- fmridataset:::.selection_compose
  expand <- fmridataset:::.selection_expand
  n <- 1000L
  inner_range <- norm(101:900, n)

  composed <- compose(norm(1:100, 800L), inner_range)
  expect_identical(composed$form, "range")
  expect_identical(c(composed$start, composed$end), c(101L, 200L))
  expect_identical(compose(norm(NULL, 800L), inner_range), inner_range)
  expect_identical(compose(norm(5:1, n), norm(NULL, n))$positions, 5:1)
  expect_identical(compose(norm(c(3, 1), 800L), inner_range)$positions, c(103L, 101L))
  expect_identical(
    compose(norm(2:1, 3L), norm(c(9, 4, 7), n))$positions,
    c(4L, 9L)
  )

  # Composition agrees with expanding and indexing, for every pair of forms.
  forms <- list(all = NULL, range = 3:8, positions = c(9, 2, 5, 7, 4, 6))
  for (inner in forms) {
    inner_selection <- norm(inner, 10L)
    m <- fmridataset:::.selection_length(inner_selection)
    for (outer in list(NULL, seq.int(2L, m - 1L), c(m, 1L, 2L))) {
      outer_selection <- norm(outer, m)
      expect_identical(
        expand(compose(outer_selection, inner_selection)),
        expand(inner_selection)[expand(outer_selection)]
      )
    }
  }

  err <- expect_error(compose(norm(1:3, 3L), inner_range), class = "fmridataset_error_alignment")
  expect_identical(err$reason, "compose_length")
})

test_that("runs come straight from the normalized form", {
  norm <- fmridataset:::.normalize_selection
  runs <- fmridataset:::.selection_runs

  expect_identical(runs(norm(NULL, 4L)), list(c(1L, 4L)))
  expect_identical(runs(norm(NULL, 0L)), list())
  expect_identical(runs(norm(2:3, 4L)), list(c(2L, 3L)))
  expect_identical(runs(norm(c(9, 1, 2, 3, 7), 10L)), list(c(1L, 3L), c(7L, 7L), c(9L, 9L)))
  expect_identical(runs(norm(integer(), 4L)), list())
})

# -------------------------------------------------------- nested views

test_that("nested frame views compose into one compact view over the base", {
  x <- make_frame_fixture()$frame

  ranged <- x[2:7, ][2:4, ]
  expect_identical(ranged$base, x)
  expect_identical(ranged$observation$form, "range")
  expect_identical(c(ranged$observation$start, ranged$observation$end), c(3L, 5L))
  expect_identical(ranged$feature$form, "all")
  expect_identical(observation_ids(ranged), observation_ids(x)[3:5])

  picked <- ranged[3:1, feature_ids(x)[c(6L, 2L)]]
  expect_identical(picked$base, x)
  expect_identical(picked$observation$positions, c(5L, 4L, 3L))
  expect_identical(picked$feature$positions, c(6L, 2L))
  expect_equal(collect_assay(picked), collect_assay(x)[c(5L, 4L, 3L), c(6L, 2L), drop = FALSE])

  # Selecting everything at every level leaves a select-all view, and the
  # view's assay source is one view over the root source.
  everything <- x[, ][, ][, ]
  expect_identical(everything$observation$form, "all")
  expect_identical(everything$feature$form, "all")
  assay_source <- assay(everything)$source
  expect_s3_class(assay_source, "source_view")
  expect_false(inherits(assay_source$source, "source_view"))
  expect_identical(source_fingerprint(assay_source), source_fingerprint(assay(x[, ])$source))
})

test_that("equal selections agree in fingerprint however they were expressed", {
  x <- make_frame_fixture()$frame
  ids <- observation_ids(x)

  by_position <- x[2:4, ]
  by_id <- x[ids[2:4], ]
  by_mask <- x[ids %in% ids[2:4], ]
  by_negation <- x[-c(1L, 5:7), ]
  by_nesting <- x[2:7, ][1:3, ]

  fingerprints <- vapply(
    list(by_position, by_id, by_mask, by_negation, by_nesting),
    function(view) source_fingerprint(assay(view)$source),
    character(1)
  )
  expect_length(unique(fingerprints), 1L)

  plans <- vapply(
    list(by_position, by_id, by_mask, by_negation, by_nesting),
    function(view) plan_blocks(view)$selection_fingerprint,
    character(1)
  )
  expect_length(unique(plans), 1L)
  expect_false(identical(plans[[1L]], plan_blocks(x[4:2, ])$selection_fingerprint))
  expect_identical(
    fds_manifest_digest(fds_frame_manifest(by_id)),
    fds_manifest_digest(fds_frame_manifest(by_nesting))
  )
})

test_that("semantic digests are unaffected by how a selection is stored", {
  first <- make_frame_fixture()$frame
  second <- make_frame_fixture()$frame

  expect_identical(
    fds_manifest_digest(fds_frame_manifest(first)),
    fds_manifest_digest(fds_frame_manifest(second))
  )
  expect_identical(
    fds_manifest_digest(fds_frame_manifest(first[, ])),
    fds_manifest_digest(fds_frame_manifest(first))
  )
  expect_identical(
    fds_manifest_digest(fds_frame_manifest(first[c(3L, 1L), 2:4])),
    fds_manifest_digest(fds_frame_manifest(second[c(3L, 1L), 2:4]))
  )
})

test_that("views serialize with their selection and fingerprint intact", {
  x <- make_frame_fixture()$frame
  view <- x[c(7L, 2L), 2:4]
  restored <- unserialize(serialize(view, NULL))

  expect_identical(restored$observation, view$observation)
  expect_identical(restored$feature, view$feature)
  expect_identical(
    source_fingerprint(assay(restored)$source),
    source_fingerprint(assay(view)$source)
  )
  expect_equal(collect_assay(restored), collect_assay(view))

  source_restored <- unserialize(serialize(assay(view)$source, NULL))
  expect_identical(source_fingerprint(source_restored), source_fingerprint(assay(view)$source))
  expect_equal(source_read(source_restored), source_read(assay(view)$source))
})

test_that("explain reports the normalized selection without expanding it", {
  x <- make_frame_fixture()$frame

  whole <- explain(x)$selection
  expect_identical(whole$observation$form, "all")
  expect_identical(whole$observation$count, 7L)
  expect_null(whole$observation$positions)

  ranged <- explain(x[2:5, c(6L, 1L)])$selection
  expect_identical(ranged$observation$form, "range")
  expect_identical(c(ranged$observation$start, ranged$observation$end), c(2L, 5L))
  expect_identical(ranged$feature$form, "positions")
  expect_identical(ranged$feature$count, 2L)
  expect_null(ranged$feature$positions)
})

# -------------------------------------------- descriptor size independence

test_that("select-all and range views do not scale with the axis length", {
  # A Zarr descriptor declares its shape without touching a store, so it is a
  # memory-free virtual array of any size.
  small <- zarr_array_source(
    "virtual-small.zarr",
    shape = c(10L, 10L), dtype = "float64", chunks = c(10L, 10L)
  )
  large <- zarr_array_source(
    "virtual-large.zarr",
    shape = c(1e6, 1e6), dtype = "float64", chunks = c(1000L, 1000L)
  )

  small_view <- source_view(small)
  large_view <- source_view(large)
  overhead <- function(view, base) as.numeric(object.size(view)) - as.numeric(object.size(base))
  expect_identical(overhead(large_view, large), overhead(small_view, small))
  expect_lt(as.numeric(object.size(large_view)), 10000)
  expect_identical(source_shape(large_view), c(1000000L, 1000000L))

  ranged <- source_view(large, observations = 5:999999, features = 1000:2000)
  expect_identical(ranged$observations$form, "range")
  expect_identical(ranged$features$form, "range")
  expect_lt(as.numeric(object.size(ranged)), 10000)

  nested <- source_view(ranged, observations = 2:999994, features = 1:1000)
  expect_lt(as.numeric(object.size(nested)), 10000)
  expect_identical(c(nested$observations$start, nested$observations$end), c(6L, 999998L))

  # Spelling out the full axis is canonicalized away, so it fingerprints and
  # serializes exactly like the select-all view.
  spelled <- source_view(large, observations = seq_len(1e6))
  expect_identical(source_fingerprint(spelled), source_fingerprint(large_view))
  expect_identical(length(serialize(spelled, NULL)), length(serialize(large_view, NULL)))

  elapsed <- system.time(for (i in 1:50) source_view(large))[["elapsed"]]
  expect_lt(elapsed, 5)
})

test_that("frame selections hash and plan without expanding a select-all axis", {
  observations <- data.frame(.obs_id = sprintf("observation-%06d", seq_len(100000L)))
  spatial <- index_space(2L, ids = c("left", "right"), namespace = "selection-scale")
  source <- counting_source(memory_source(matrix(0, nrow = 100000L, ncol = 2L)))
  frame <- fmri_frame(list(signal = source), observations, space = spatial)

  selection <- fmridataset:::.frame_selection(frame)
  expect_identical(selection$observations$form, "all")
  expect_lt(as.numeric(object.size(fmridataset:::.selection_descriptor(selection$observations))), 1000)
  expect_true(fmridataset:::.has_complete_feature_selection(frame))
  expect_true(fmridataset:::.has_complete_feature_selection(frame[1:10, ]))
  expect_false(fmridataset:::.has_complete_feature_selection(frame[, 1L]))

  ranged <- frame[2:99999, ]
  expect_identical(ranged$observation$form, "range")
  expect_lt(as.numeric(object.size(ranged$observation)), 2000)
  expect_lt(as.numeric(object.size(fds_frame_manifest(frame[, ])$assays)), 5000)
  expect_identical(source_counts(source)$bytes, 0)
})

# --------------------------------------------------------- pushdown forms

test_that("every built-in source declares its selector pushdown forms honestly", {
  forms <- fmridataset:::.source_pushdown_forms
  m <- matrix(as.double(1:24), 6L, 4L)
  memory <- memory_source(m)
  all_forms <- c("all", "range", "positions")

  expect_identical(forms(memory), all_forms)
  expect_identical(forms(counting_source(memory)), all_forms)
  expect_identical(forms(fault_source(memory, "read")), all_forms)
  expect_identical(forms(source_view(memory, observations = c(5L, 1L))), all_forms)
  expect_identical(forms(row_bound_source(list(memory, memory_source(m[1:2, ])))), all_forms)
  expect_identical(forms(row_sharded_source(list(memory, memory_source(m[1:2, ])))), all_forms)

  from <- index_space(4L, ids = sprintf("s%d", 1:4), namespace = "from")
  to <- index_space(2L, ids = c("t1", "t2"), namespace = "to")
  weights <- rbind(c(1, 1, 0, 0), c(0, 0, 1, 1))
  mapped <- feature_mapped_source(memory, feature_map(from, to, weights))
  expect_identical(forms(mapped), all_forms)

  zarr <- zarr_array_source(
    "fixture.zarr",
    shape = c(6L, 7L), dtype = "float64", chunks = c(2L, 3L)
  )
  expect_identical(forms(zarr), c("all", "range"))
  expect_false("pushdown:positions" %in% source_capabilities(zarr))
  expect_identical(forms(source_view(zarr, observations = 2:3)), c("all", "range"))
  expect_identical(forms(counting_source(zarr)), c("all", "range"))

  # A composition can only push down what every child can.
  mixed <- row_bound_source(list(memory_source(m[1:3, 1:7 %% 4 + 1L]), zarr))
  expect_identical(forms(mixed), c("all", "range"))

  expect_identical(forms(fmridataset:::.sparse_entity_source(Matrix::Matrix(m, sparse = TRUE))), all_forms)
  expect_identical(forms(fmridataset:::.row_index_source(memory, c(2L, NA, 1L))), all_forms)

  # Declarations are ordinary capability strings, so they are serializable
  # and pass the source contract.
  expect_invisible(validate_array_source(memory))
  expect_true(all(grepl("^pushdown:(all|range|positions)$", grep("^pushdown:", source_capabilities(memory), value = TRUE))))
  expect_identical(forms(unserialize(serialize(zarr, NULL))), c("all", "range"))
})

test_that("NIfTI sources declare every pushdown form", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  source <- nifti_array_source(path, path)
  expect_identical(fmridataset:::.source_pushdown_forms(source), c("all", "range", "positions"))
  expect_identical(
    fmridataset:::.source_pushdown_forms(source_view(source, features = 1L)),
    c("all", "range", "positions")
  )
})

# ----------------------------------------------------- zero-length axes

test_that("zero-length selections plan zero blocks and collect empty matrices", {
  x <- make_frame_fixture()$frame

  no_rows <- x[integer(), ]
  no_cols <- x[, integer()]
  nothing <- x[integer(), integer()]
  nested <- x[2:4, ][integer(), 2:3]

  for (view in list(no_rows, no_cols, nothing, nested)) {
    plan <- plan_blocks(view)
    expect_identical(plan$n_blocks, 0L)
    expect_identical(nrow(block_manifest(plan)), 0L)
    expect_length(execute_block_plan(view, plan, function(values, ...) values), 0L)
  }
  expect_identical(dim(collect_assay(no_rows)), c(0L, 6L))
  expect_identical(dim(collect_assay(no_cols)), c(7L, 0L))
  expect_identical(dim(collect_assay(nothing)), c(0L, 0L))
  expect_identical(dim(collect_assay(nested)), c(0L, 2L))
  expect_identical(dim(source_read(assay(nested)$source)), c(0L, 2L))
  expect_identical(explain(nested)$selection$observation$count, 0L)
})
