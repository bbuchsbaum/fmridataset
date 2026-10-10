.contract_space <- function(ids = c("f1", "f2", "f3")) {
  index_space(length(ids), ids = ids, namespace = "frame-input-contracts")
}

.contract_frame <- function(ids = c("o1", "o2"), assays = c("bold", "z"),
                            features = NULL) {
  n <- length(ids)
  values <- lapply(seq_along(assays), function(k) matrix(seq_len(n * 3L) * k, n, 3L))
  names(values) <- assays
  fmri_frame(
    values,
    data.frame(.obs_id = ids, run = seq_len(n)),
    features = features,
    space = if (is.null(features)) .contract_space() else NULL
  )
}

test_that("binding an ephemeral axis yields an ephemeral axis", {
  durable <- .contract_frame()
  ephemeral <- fmri_frame(
    list(bold = matrix(1:3, 1L, 3L), z = matrix(4:6, 1L, 3L)),
    axis_frame(data.frame(run = 3L), id_policy = "ephemeral"),
    space = .contract_space()
  )
  for (bound in list(
    bind_observations(durable, ephemeral),
    bind_observations(ephemeral, durable)
  )) {
    expect_false(ids_are_durable(observation_axis(bound)))
    expect_identical(axis_id_policy(observation_axis(bound))$policy, "ephemeral")
    expect_false(explain(bound)$ids_durable)
    expect_error(fds_frame_manifest(bound), class = "fmridataset_error_identity")
    expect_error(
      identity_descriptor(bound, domain = "semantic"),
      class = "fmridataset_error_identity"
    )
  }

  both <- bind_observations(durable, .contract_frame(c("o3", "o4")))
  expect_true(ids_are_durable(observation_axis(both)))
})

test_that("aligned assays cannot be reused on different axes", {
  frame <- .contract_frame()
  expect_error(
    fmri_frame(assays(frame), data.frame(.obs_id = c("o2", "o1")), space = .contract_space()),
    "different observation axis",
    class = "fmridataset_error_alignment"
  )
  expect_error(
    fmri_frame(
      assays(frame), data.frame(.obs_id = c("o1", "o2")),
      space = .contract_space(c("f3", "f2", "f1"))
    ),
    "different feature axis",
    class = "fmridataset_error_alignment"
  )
  view <- frame[2:1, ]
  expect_error(
    fmri_frame(assays(view), data.frame(.obs_id = c("o1", "o2")), space = .contract_space()),
    class = "fmridataset_error_alignment"
  )

  # The same axes, or bare sources re-annotated explicitly, are accepted.
  same <- fmri_frame(assays(frame), observation_axis(frame), features = feature_axis(frame))
  expect_identical(collect_assay(same), collect_assay(frame))
  realigned <- fmri_frame(
    lapply(assays(view), function(value) value$source),
    data.frame(.obs_id = c("o2", "o1")),
    space = .contract_space()
  )
  expect_identical(collect_assay(realigned), collect_assay(view))
})

test_that("axis_frame refuses an explicit id that contradicts the ID column", {
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", "b")), id = c("x", "y")),
    "disagree",
    class = "fmridataset_error_identity"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = c("a", "b")), id = c("b", "a")),
    class = "fmridataset_error_identity"
  )
  agreed <- axis_frame(data.frame(.obs_id = c("a", "b")), id = c("a", "b"))
  expect_identical(axis_ids(agreed), c("a", "b"))
})

test_that("numeric axis IDs format exactly and fractional IDs are refused", {
  x <- axis_frame(data.frame(.obs_id = c(100000, 1e5 + 1, 0)))
  expect_identical(axis_ids(x), c("100000", "100001", "0"))
  expect_identical(axis_ids(axis_frame(data.frame(.obs_id = c(-0, 1e15)))), c("0", "1000000000000000"))
  expect_identical(axis_ids(axis_frame(data.frame(.obs_id = c(100000L, 2L)))), c("100000", "2"))
  expect_error(
    axis_frame(data.frame(.obs_id = c(1, 1.00000001))),
    "whole numbers",
    class = "fmridataset_error_identity"
  )
  expect_error(
    axis_frame(data.frame(value = 1:2), id = c(1.5, 2)),
    class = "fmridataset_error_identity"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = c(1, Inf))),
    class = "fmridataset_error_identity"
  )
  expect_error(
    axis_frame(data.frame(.obs_id = c(1, NA_real_))),
    class = "fmridataset_error_alignment"
  )
})

test_that("assay selection requires one existing assay name", {
  frame <- .contract_frame()
  view <- frame[2:1, c(3, 1)]
  for (x in list(frame, view)) {
    expect_identical(assay(x)$name, "bold")
    expect_identical(assay(x, "z")$name, "z")
    for (bad in list(c("bold", "source"), c("bold", "dtype"), 2, NA_character_, "", NULL)) {
      expect_error(assay(x, bad), class = "fmridataset_error_alignment")
      expect_error(collect_assay(x, bad), class = "fmridataset_error_alignment")
    }
    expect_error(assay(x, "missing"), "Unknown assay", class = "fmridataset_error_alignment")
  }
})

test_that("fmri_frame validates active_assay as one string", {
  build <- function(active) {
    fmri_frame(
      list(x = matrix(1:6, 2L, 3L), y = matrix(1:6, 2L, 3L)),
      data.frame(.obs_id = c("a", "b")),
      space = .contract_space(), active_assay = active
    )
  }
  expect_identical(active_assay(build("y")), "y")
  for (bad in list(c("x", "y"), NA_character_, 2, "")) {
    expect_error(build(bad), "active_assay", class = "fmridataset_error_alignment")
  }
  expect_error(build("missing"), "not present", class = "fmridataset_error_alignment")
})

test_that("frame slicing refuses single and extra indices", {
  frame <- .contract_frame()
  view <- frame[2:1, ]
  for (x in list(frame, view)) {
    expect_error(x[2], "x\\[i, \\]", class = "fmridataset_error_alignment")
    expect_error(x["o1"], class = "fmridataset_error_alignment")
    expect_error(x[1, 2, 3], "at most two", class = "fmridataset_error_alignment")
    expect_error(x[1, 1, drop = TRUE], "non-dropping", class = "fmridataset_error_alignment")
    expect_identical(dim(x[1, ]), c(1L, 3L))
    expect_identical(dim(x[, 2]), c(2L, 1L))
    expect_identical(dim(x[1, 2, drop = FALSE]), c(1L, 1L))
    expect_identical(dim(x[]), c(2L, 3L))
  }
})

test_that("subsetting a spatial feature axis restricts its space", {
  axis <- feature_axis(.contract_frame())
  sub <- axis[c("f3", "f1")]
  expect_s3_class(sub, "spatial_axis_frame")
  expect_identical(axis_ids(sub), c("f3", "f1"))
  expect_identical(feature_ids(space(sub)), c("f3", "f1"))
  expect_identical(axis_id_policy(sub), axis_id_policy(axis))
  expect_identical(axis_ids(axis[2:3]), feature_ids(space(axis[2:3])))
  expect_identical(axis[], axis)
  expect_identical(axis[1:3], axis)
  # The restricted axis builds a frame without contradicting its space.
  frame <- fmri_frame(list(bold = matrix(1:4, 2L, 2L)), data.frame(.obs_id = c("a", "b")), features = sub)
  expect_identical(feature_ids(frame), c("f3", "f1"))
})

test_that("binding requires exact agreement on retained feature metadata", {
  sp <- .contract_space()
  make <- function(id, w) {
    fmri_frame(
      list(bold = matrix(1:3, 1L, 3L)), data.frame(.obs_id = id),
      features = feature_axis(data.frame(.feature_id = c("f1", "f2", "f3"), w = w), space = sp)
    )
  }
  expect_error(
    bind_observations(make("a", c(1, 2, 3)), make("b", c(1, 2, 3 + 1e-9))),
    "feature metadata",
    class = "fmridataset_error_alignment"
  )
  bound <- bind_observations(make("a", c(1, 2, 3)), make("b", c(1, 2, 3)))
  expect_identical(dim(bound), c(2L, 3L))
})

test_that("as_fmri_frame refuses unsupported classes with a structured error", {
  expect_error(as_fmri_frame(1), "No as_fmri_frame method", class = "fmridataset_error_config")
})

test_that("legacy backend-era error helpers are gone", {
  ns <- asNamespace("fmridataset")
  for (name in c("fmridataset_error_backend_io", "fmridataset_error_config", "stop_fmridataset")) {
    expect_false(exists(name, envir = ns, inherits = FALSE))
  }
})
