# Every silent-wrong-answer defect this package has had lived at a JOIN:
# combining subjects, binding frames, mapping features, chunking a read. The
# suite had thousands of expectations and caught none of them, because it
# tested each operation in isolation and never the laws relating them.
#
# These are those laws, swept over shapes and selections rather than asserted
# at one example. Each names the defect class it would have caught.

set.seed(20260828)

property_frame <- function(n_obs, n_feat, runs = NULL, blocks = TRUE) {
  space <- index_space(n_feat, ids = sprintf("f%02d", seq_len(n_feat)), namespace = "prop")
  observations <- data.frame(
    .obs_id = sprintf("o%03d", seq_len(n_obs)),
    grp = rep_len(c("a", "b"), n_obs),
    stringsAsFactors = FALSE
  )
  if (!is.null(runs)) observations$run_id <- runs

  axis <- if (blocks) {
    axis_frame(
      observations,
      blocks = list(motion = axis_block(
        matrix(as.double(seq_len(n_obs * 2)), n_obs, 2),
        components = tibble::tibble(.component_id = c("translation", "rotation"))
      ))
    )
  } else {
    axis_frame(observations)
  }

  fmri_frame(
    assays = list(a = memory_source(matrix(as.double(seq_len(n_obs * n_feat)), n_obs, n_feat))),
    observations = axis,
    features = feature_axis(
      tibble::tibble(.feature_id = feature_ids(space), parcel = rep_len(c("p", "q"), n_feat)),
      space = space
    )
  )
}

shapes <- list(c(6, 4), c(1, 3), c(5, 1), c(9, 7))

# A frame whose assay is a feature-mapped source, with a non-finite value in a
# zero-weighted source column. The laws below must hold for a composed lazy
# source exactly as they do for a materialised one -- that is precisely what
# the feature-map defect broke.
mapped_frame <- function(n_obs = 5) {
  from <- index_space(4, ids = sprintf("s%d", 1:4), namespace = "mfrom")
  to <- index_space(3, ids = sprintf("t%d", 1:3), namespace = "mto")
  weights <- rbind(c(1, 0, 0, 0), c(0, 1, 1, 0), c(0, 0, 0, 1))

  values <- matrix(as.double(seq_len(n_obs * 4)), n_obs, 4)
  values[1, 1] <- NA_real_ # weighted only by t1, so t2 and t3 must be unaffected

  fmri_frame(
    assays = list(a = feature_mapped_source(
      memory_source(values), feature_map(from, to, weights)
    )),
    observations = data.frame(.obs_id = sprintf("o%03d", seq_len(n_obs))),
    space = to
  )
}

test_that("selection commutes with collection", {
  # If collect(f[i, j]) ever differs from collect(f)[i, j], a read depends on
  # the shape of the request that produced it. That is the feature-map defect.
  for (shape in shapes) {
    frame <- property_frame(shape[[1]], shape[[2]])
    full <- collect_assay(frame)

    for (trial in seq_len(4)) {
      rows <- sample.int(shape[[1]], size = sample.int(shape[[1]], 1))
      cols <- sample.int(shape[[2]], size = sample.int(shape[[2]], 1))
      expect_equal(
        collect_assay(frame[rows, cols]),
        full[rows, cols, drop = FALSE],
        info = paste(shape, collapse = "x")
      )
    }
  }

  # The same law over a feature-mapped source, where a non-finite value sits in
  # a column that only ONE target feature weights. This is the exact shape of
  # the defect: collect(f)[i, j] and collect(f[i, j]) disagreed.
  mapped <- mapped_frame()
  mapped_full <- collect_assay(mapped)
  for (cols in list(1L, 2L, 3L, c(2L, 3L), c(3L, 1L), 1:3)) {
    expect_equal(
      collect_assay(mapped[, cols]),
      mapped_full[, cols, drop = FALSE],
      info = paste("mapped cols", paste(cols, collapse = ","))
    )
  }
  # t2 and t3 carry no weight on the NA column, so they must be finite.
  expect_true(all(is.finite(mapped_full[, 2:3])))
  expect_true(is.na(mapped_full[1, 1]))
})

test_that("block results do not depend on block size", {
  # Chunked and unchunked runs of the same computation must agree, or a lazy
  # engine is not referentially transparent.
  for (shape in shapes) {
    frame <- property_frame(shape[[1]], shape[[2]])
    reference <- collect_assay(frame)

    for (block_size in unique(c(1L, 2L, shape[[2]], shape[[2]] + 3L))) {
      pieces <- block_apply(frame, function(values, ids) values, block_size = block_size)
      expect_equal(
        do.call(cbind, pieces),
        reference,
        ignore_attr = TRUE,
        info = paste(paste(shape, collapse = "x"), "block", block_size)
      )
    }
  }

  # And over a feature-mapped source, where block width changed the numbers.
  mapped <- mapped_frame()
  mapped_reference <- collect_assay(mapped)
  for (block_size in 1:4) {
    expect_equal(
      do.call(cbind, block_apply(mapped, function(values, ids) values, block_size = block_size)),
      mapped_reference,
      ignore_attr = TRUE,
      info = paste("mapped block", block_size)
    )
  }
})

test_that("views compose", {
  # f[i, ][k, ] must equal f[i[k], ]; otherwise nested lazy selection drifts.
  for (shape in shapes) {
    if (shape[[1]] < 2) next
    frame <- property_frame(shape[[1]], shape[[2]])

    for (trial in seq_len(4)) {
      outer <- sample.int(shape[[1]])
      inner <- sample.int(length(outer), size = max(1L, length(outer) %/% 2L))

      expect_equal(
        collect_assay(frame[outer, ][inner, ]),
        collect_assay(frame[outer[inner], ]),
        info = paste(shape, collapse = "x")
      )
      expect_identical(
        observation_ids(frame[outer, ][inner, ]),
        observation_ids(frame[outer[inner], ])
      )
    }
  }
})

test_that("partitioning the observation axis and rebinding is an identity", {
  # This is the bind law. A partition reassembled must give back exactly what
  # was taken apart -- values, IDs, feature metadata, and block components.
  for (shape in shapes) {
    n <- shape[[1]]
    frame <- property_frame(n, shape[[2]])
    if (n < 2) next

    for (cut in unique(c(1L, n %/% 2L, n - 1L))) {
      first <- frame[seq_len(cut), ]
      second <- frame[seq(cut + 1L, n), ]
      rebound <- bind_observations(first, second)

      expect_equal(collect_assay(rebound), collect_assay(frame),
        info = paste(paste(shape, collapse = "x"), "cut", cut)
      )
      expect_identical(observation_ids(rebound), observation_ids(frame))
      expect_identical(feature_ids(rebound), feature_ids(frame))
      expect_equal(observations(rebound)$grp, observations(frame)$grp)
      expect_identical(
        block_component_ids(obs_blocks(rebound)$motion),
        block_component_ids(obs_blocks(frame)$motion)
      )
    }
  }
})

test_that("rebinding a partition preserves block values against their labels", {
  # The bind defect filed one frame's values under another frame's component
  # labels. Reassembling a partition must reproduce the block exactly.
  frame <- property_frame(8, 3)
  reference <- as.matrix(source_read(as_array_source(
    axis_block_data(obs_blocks(frame)$motion)
  )))

  for (cut in c(1L, 4L, 7L)) {
    rebound <- bind_observations(frame[seq_len(cut), ], frame[seq(cut + 1L, 8L), ])
    bound <- as.matrix(source_read(as_array_source(
      axis_block_data(obs_blocks(rebound)$motion)
    )))
    expect_equal(bound, reference, info = paste("cut", cut))
  }
})

test_that("a three-way partition rebinds in any grouping", {
  # Binding is associative on a partition: (a+b)+c must equal a+(b+c).
  frame <- property_frame(9, 4)
  a <- frame[1:3, ]
  b <- frame[4:6, ]
  c <- frame[7:9, ]

  left <- bind_observations(bind_observations(a, b), c)
  right <- bind_observations(a, bind_observations(b, c))

  expect_equal(collect_assay(left), collect_assay(frame))
  expect_equal(collect_assay(right), collect_assay(frame))
  expect_identical(observation_ids(left), observation_ids(right))
})

test_that("a plan partitions the frame exactly once", {
  # Every value must appear in exactly one block, whatever the budget.
  for (shape in shapes) {
    frame <- property_frame(shape[[1]], shape[[2]])
    realized <- fmridataset:::.realized_dtype_bytes(assay(frame)$dtype)

    for (budget in realized * c(1L, 3L, 8L, 64L)) {
      plan <- plan_blocks(frame, memory_budget = budget, target_block_bytes = budget)
      expect_equal(
        plan$total_values, prod(as.double(shape)),
        info = paste(paste(shape, collapse = "x"), "budget", budget)
      )
      expect_lte(plan$max_peak_bytes, budget)
    }
  }
})

test_that("the temporal schema follows selection", {
  # Run structure is derived from observation metadata, so it must track any
  # selection that preserves order rather than going stale.
  frame <- property_frame(9, 3, runs = rep(c("r1", "r2", "r3"), each = 3))
  expect_equal(unname(temporal_schema(frame)$run_lengths), c(3L, 3L, 3L))

  kept <- frame[c(1, 2, 4, 7, 8, 9), ]
  schema <- temporal_schema(kept)
  expect_equal(unname(schema$run_lengths), c(2L, 1L, 3L))
  expect_true(schema$contiguous)

  reordered <- frame[c(1, 4, 2, 7), ]
  expect_false(temporal_schema(reordered)$contiguous)
  expect_equal(sum(temporal_schema(reordered)$run_lengths), 4L)
})

test_that("an FDS manifest round trip is an identity on a reordered view", {
  frame <- property_frame(6, 4)
  view <- frame[c(5, 1, 3), c(4, 2)]

  manifest <- fds_frame_manifest(view)
  # Every array the manifest declares must be bound, the observation block
  # included: a frame is its assays AND its aligned axis blocks.
  expect_setequal(
    names(manifest$arrays),
    c("assays/a", "axis/observation/blocks/motion")
  )
  restored <- frame_from_fds_manifest(
    manifest,
    bindings = list(
      "assays/a" = memory_source(collect_assay(view)),
      "axis/observation/blocks/motion" = memory_source(as.matrix(source_read(
        as_array_source(axis_block_data(obs_blocks(view)$motion))
      )))
    )
  )

  expect_identical(observation_ids(restored), observation_ids(view))
  expect_identical(feature_ids(restored), feature_ids(view))
  expect_equal(collect_assay(restored), collect_assay(view))
  expect_identical(space_digest(space(restored)), space_digest(space(view)))
})
