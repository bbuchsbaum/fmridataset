# The frame model carries no acquisition timing of its own; run structure, TR
# and censoring are ordinary observation metadata. read_bids_bold() already
# emits them that way, but nothing validated the convention, so no consumer
# could rely on it. These tests pin the contract.

temporal_frame <- function(runs, TR = 2, censor = NULL, extra = NULL) {
  n <- length(runs)
  obs <- data.frame(
    .obs_id = sprintf("t%03d", seq_len(n)),
    run_id = runs,
    TR = TR,
    stringsAsFactors = FALSE
  )
  if (!is.null(censor)) obs$censor <- censor
  if (!is.null(extra)) obs <- cbind(obs, extra)
  fmri_frame(list(bold = matrix(rnorm(n * 2), n, 2)), observations = obs)
}

test_that("run structure is derived in first-appearance order", {
  schema <- temporal_schema(temporal_frame(rep(c("b", "a"), each = 3)))

  expect_s3_class(schema, "frame_temporal_schema")
  expect_equal(schema$n_runs, 2L)
  # Order of appearance, not sorted order.
  expect_equal(names(schema$run_lengths), c("b", "a"))
  expect_equal(unname(schema$run_lengths), c(3L, 3L))
  expect_equal(schema$block_ids, c(1L, 1L, 1L, 2L, 2L, 2L))
  expect_true(schema$contiguous)
})

test_that("uneven runs and a single run are described correctly", {
  uneven <- temporal_schema(temporal_frame(c("a", "a", "a", "b", "c", "c")))
  expect_equal(unname(uneven$run_lengths), c(3L, 1L, 2L))
  expect_equal(uneven$n_runs, 3L)

  single <- temporal_schema(temporal_frame(rep("only", 4)))
  expect_equal(single$n_runs, 1L)
  expect_equal(unname(single$run_lengths), 4L)
  expect_true(single$contiguous)
})

test_that("a sampling frame round-trips through fmrihrf", {
  frame <- temporal_frame(rep(c("r1", "r2"), each = 3))
  sf <- as_sampling_frame(frame)

  expect_s3_class(sf, "sampling_frame")
  expect_equal(sf$blocklens, c(3, 3))
  expect_equal(unique(sf$TR), 2)
  expect_equal(
    as.integer(fmrihrf::blockids(sf)),
    temporal_schema(frame)$block_ids
  )
})

test_that("TR may differ between runs but not within one", {
  varying <- temporal_frame(rep(c("a", "b"), each = 3), TR = rep(c(2, 3), each = 3))
  schema <- temporal_schema(varying)
  expect_equal(unname(schema$TR), c(2, 3))
  expect_equal(as_sampling_frame(varying)$TR, c(2, 3))

  inconsistent <- temporal_frame(rep("a", 4), TR = c(2, 2, 3, 2))
  expect_error(temporal_schema(inconsistent), class = "fmridataset_error_temporal")
  expect_error(temporal_schema(inconsistent), "one repetition time")
})

test_that("invalid TR values are refused", {
  for (bad in list(c(2, 2, -1, 2), c(2, 2, 0, 2), c(2, 2, NA, 2), c(2, 2, Inf, 2))) {
    expect_error(
      temporal_schema(temporal_frame(rep("a", 4), TR = bad)),
      class = "fmridataset_error_temporal"
    )
  }
  expect_error(
    temporal_schema(temporal_frame(rep("a", 4), TR = rep("2", 4))),
    "must be numeric"
  )
})

test_that("a frame without TR still describes its runs", {
  n <- 4
  frame <- fmri_frame(
    list(bold = matrix(rnorm(n * 2), n, 2)),
    observations = data.frame(
      .obs_id = sprintf("t%d", seq_len(n)),
      run_id = rep(c("a", "b"), each = 2)
    )
  )
  schema <- temporal_schema(frame)

  expect_null(schema$TR)
  expect_equal(unname(schema$run_lengths), c(2L, 2L))
  expect_error(as_sampling_frame(frame), "repetition times")
})

test_that("censoring is validated when present", {
  censored <- temporal_frame(rep("a", 4), censor = c(TRUE, FALSE, FALSE, TRUE))
  expect_equal(temporal_schema(censored)$censor, c(TRUE, FALSE, FALSE, TRUE))

  expect_null(temporal_schema(temporal_frame(rep("a", 4)))$censor)
  expect_error(
    temporal_schema(temporal_frame(rep("a", 4), censor = c(TRUE, NA, FALSE, TRUE))),
    "missing values"
  )
  expect_error(
    temporal_schema(temporal_frame(rep("a", 4), censor = c(1, 0, 0, 1))),
    "must be logical"
  )
})

test_that("missing or empty run labels are refused", {
  expect_error(
    temporal_schema(temporal_frame(c("a", NA, "b", "b"))),
    "missing value"
  )
  expect_error(
    temporal_schema(temporal_frame(c("a", "", "b", "b"))),
    "empty run labels"
  )
})

test_that("a reordered view is legal but cannot become a sampling frame", {
  frame <- temporal_frame(rep(c("r1", "r2"), each = 3))
  interleaved <- frame[c(1, 4, 2, 5, 3, 6), ]

  schema <- temporal_schema(interleaved)
  expect_false(schema$contiguous)
  expect_equal(schema$block_ids, c(1L, 2L, 1L, 2L, 1L, 2L))
  # Run lengths still count correctly even when interleaved.
  expect_equal(unname(schema$run_lengths), c(3L, 3L))

  expect_error(as_sampling_frame(interleaved), class = "fmridataset_error_temporal")
  expect_error(as_sampling_frame(interleaved), "run-length encoding")
})

test_that("a contiguity-preserving subset still converts", {
  frame <- temporal_frame(rep(c("r1", "r2"), each = 3))
  kept <- frame[c(1, 2, 4, 5), ]

  schema <- temporal_schema(kept)
  expect_true(schema$contiguous)
  expect_equal(unname(schema$run_lengths), c(2L, 2L))
  expect_equal(as_sampling_frame(kept)$blocklens, c(2, 2))
})

test_that("a single-run subset converts", {
  frame <- temporal_frame(rep(c("r1", "r2"), each = 3))
  one_run <- frame[4:6, ]

  expect_equal(temporal_schema(one_run)$n_runs, 1L)
  expect_equal(as_sampling_frame(one_run)$blocklens, 3)
})

test_that("a zero-observation frame has an empty schema", {
  schema <- temporal_schema(temporal_frame(rep("a", 3))[integer(0), ])

  expect_equal(schema$n_runs, 0L)
  expect_equal(length(schema$run_ids), 0L)
  expect_true(schema$contiguous)
})

test_that("the run column is discovered from a declared run entity", {
  # Two sessions each holding BIDS run-1: the labels collide, the scans do not.
  n <- 3
  observations <- data.frame(
    .obs_id = sprintf("v%02d", seq_len(2 * n)),
    scan_id = rep(c("ses-1_run-1", "ses-2_run-1"), each = n),
    run_id = rep("run-1", 2 * n),
    TR = 2,
    stringsAsFactors = FALSE
  )
  frame <- fmri_frame(
    list(bold = matrix(rnorm(2 * n * 2), 2 * n, 2)),
    observations = observations,
    entities = list(
      run = entity_frame(
        data.frame(scan_id = c("ses-1_run-1", "ses-2_run-1"), stringsAsFactors = FALSE),
        key = "scan_id", entity_type = "run"
      )
    ),
    relations = list(observation_run = key_relation("scan_id", target = "run"))
  )

  schema <- temporal_schema(frame)
  expect_equal(schema$columns$run, "scan_id")
  expect_equal(schema$n_runs, 2L)

  # Guessing the BIDS label would have merged two acquisitions into one.
  expect_equal(temporal_schema(frame, run_col = "run_id")$n_runs, 1L)
})

test_that("without a declared entity, scan_id is preferred over run_id", {
  n <- 4
  both <- fmri_frame(
    list(a = matrix(rnorm(n * 2), n, 2)),
    observations = data.frame(
      .obs_id = sprintf("o%d", seq_len(n)),
      run_id = rep("same", n),
      scan_id = rep(c("x", "y"), each = 2),
      TR = 1,
      stringsAsFactors = FALSE
    )
  )
  expect_equal(temporal_schema(both)$columns$run, "scan_id")
  expect_equal(temporal_schema(both)$n_runs, 2L)
})

test_that("a frame with no run information reports what it looked for", {
  bare <- fmri_frame(
    list(a = matrix(1:8, 4, 2)),
    data.frame(.obs_id = sprintf("o%d", 1:4))
  )

  expect_false(has_temporal_schema(bare))
  expect_error(temporal_schema(bare), class = "fmridataset_error_temporal")
  expect_error(temporal_schema(bare), "scan_id")
  expect_error(temporal_schema(bare), "run_id")

  expect_error(temporal_schema(bare, run_col = "nope"), "no \"nope\" column")
})

test_that("has_temporal_schema reflects validity, not just presence", {
  expect_true(has_temporal_schema(temporal_frame(rep("a", 3))))
  # Present but invalid is not a usable schema.
  expect_false(has_temporal_schema(temporal_frame(rep("a", 4), TR = c(2, 2, 3, 2))))
})

test_that("the schema prints its runs", {
  out <- capture.output(print(temporal_schema(temporal_frame(rep(c("r1", "r2"), each = 3)))))

  expect_match(out[1], "2 runs")
  expect_match(paste(out, collapse = "\n"), "r1")
  expect_match(paste(out, collapse = "\n"), "TR 2 s")
})
