.toy_volume <- function(n_time, seed = 1L, dim3 = c(2L, 2L, 2L)) {
  values <- array(
    as.double(seed * 1000 + seq_len(prod(dim3) * n_time)),
    c(dim3, n_time)
  )
  neuroim2::NeuroVec(values, neuroim2::NeuroSpace(c(dim3, n_time)))
}

.toy_mask <- function(dim3 = c(2L, 2L, 2L), active = c(1L, 4L, 6L, 7L)) {
  values <- array(FALSE, dim3)
  values[active] <- TRUE
  neuroim2::LogicalNeuroVol(values, neuroim2::NeuroSpace(dim3))
}

test_that("a matrix becomes a run-structured frame with derived IDs", {
  y <- matrix(as.double(1:24), nrow = 6, dimnames = list(NULL, c("a", "b", "c", "d")))
  frame <- as_fmri_frame(y, TR = c(2, 1.5), run_length = c(4, 2))

  expect_s3_class(frame, "fmri_frame")
  expect_identical(dim(frame), c(6L, 4L))
  expect_identical(feature_ids(frame), c("a", "b", "c", "d"))
  expect_identical(
    observation_ids(frame),
    c(sprintf("run-1_vol-%04d", 1:4), sprintf("run-2_vol-%04d", 1:2))
  )
  expect_equal(collect_assay(frame, dimnames = FALSE), unname(y))
  expect_identical(active_assay(frame), "signal")

  schema <- temporal_schema(frame)
  expect_equal(unname(schema$run_lengths), c(4L, 2L))
  expect_equal(unname(schema$TR), c(2, 1.5))
  expect_true(schema$contiguous)
  expect_true(schema$ordered)
  expect_identical(observations(frame)$volume_index, c(1:4, 1:2))
})

test_that("matrix defaults are one run, no TR, and generated feature IDs", {
  frame <- as_fmri_frame(matrix(0, nrow = 3, ncol = 2))

  expect_identical(feature_ids(frame), c("feature-1", "feature-2"))
  expect_false("TR" %in% names(observations(frame)))
  expect_equal(unname(temporal_schema(frame)$run_lengths), 3L)
  expect_length(frame$tables, 0L)
})

test_that("unusable column names fall back to generated feature IDs", {
  y <- matrix(0, nrow = 2, ncol = 2, dimnames = list(NULL, c("v", "v")))
  expect_identical(feature_ids(as_fmri_frame(y)), c("feature-1", "feature-2"))
})

test_that("a matrix can be placed on an explicit feature space", {
  space <- volume_space(c(2, 2, 1), support = c(1L, 4L), template = "toy")
  frame <- as_fmri_frame(matrix(1:6 + 0, nrow = 3), space = space)

  expect_true(same_space(space(frame), space)$same)
  expect_error(
    as_fmri_frame(matrix(0, 3, 3), space = space),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(
    as_fmri_frame(matrix(0, 3, 2), space = space, feature_ids = c("a", "b")),
    class = "fmridataset_error_config"
  )
})

test_that("constructor arguments are validated with structured errors", {
  y <- matrix(0, nrow = 4, ncol = 2)

  expect_error(as_fmri_frame(y, run_length = c(3, 2)), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, run_length = c(4, 0)), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, run_length = 2.5), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, TR = -1), class = "fmridataset_error_config")
  expect_error(
    as_fmri_frame(y, TR = c(1, 2, 3), run_length = c(2, 2)),
    class = "fmridataset_error_config"
  )
  expect_error(as_fmri_frame(y, feature_ids = c("a", "a")), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, feature_ids = "a"), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, events = "onsets"), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, extra = 1))
  expect_error(
    as_fmri_frame(matrix(letters[1:4], 2)),
    class = "fmridataset_error_config"
  )
})

test_that("censor accepts logical, 0/1, and index forms and rejects the rest", {
  y <- matrix(0, nrow = 4, ncol = 1)
  expected <- c(FALSE, TRUE, FALSE, TRUE)

  expect_identical(observations(as_fmri_frame(y, censor = expected))$censor, expected)
  expect_identical(observations(as_fmri_frame(y, censor = c(0, 1, 0, 1)))$censor, expected)
  expect_identical(observations(as_fmri_frame(y, censor = c(4, 2)))$censor, expected)
  expect_identical(temporal_schema(as_fmri_frame(y, censor = c(4, 2)))$censor, expected)

  expect_error(as_fmri_frame(y, censor = c(TRUE, FALSE)), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, censor = c(TRUE, NA, FALSE, TRUE)), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, censor = 5), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, censor = c(2, 2)), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, censor = 1.5), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(y, censor = "2"), class = "fmridataset_error_config")
})

test_that("events are stored as a keyed event table", {
  y <- matrix(0, nrow = 4, ncol = 1)
  events <- data.frame(onset = c(1, 3), condition = c("a", "b"))

  frame <- as_fmri_frame(y, TR = 1, events = events)
  stored <- frame$tables$events
  expect_s3_class(stored, "fmri_event_table")
  expect_identical(event_key(stored), "event_id")
  expect_identical(event_data(stored)$event_id, c("event-1", "event-2"))
  expect_identical(event_data(stored)$condition, c("a", "b"))

  keyed <- event_table(cbind(event_id = c("x", "y"), events), key = "event_id")
  expect_identical(as_fmri_frame(y, events = keyed)$tables$events, keyed)
  expect_length(as_fmri_frame(y, events = events[0, ])$tables, 0L)
})

test_that("NeuroVec runs become a frame over the mask's volume space", {
  run_1 <- .toy_volume(3L, seed = 1L)
  run_2 <- .toy_volume(2L, seed = 2L)
  mask <- .toy_mask()
  support <- which(as.vector(mask))

  frame <- as_fmri_frame(list(run_1, run_2), mask = mask, TR = 2)

  expect_s3_class(space(frame), "volume_space")
  expect_identical(feature_ids(frame), paste0("voxel-", support))
  expect_equal(unname(temporal_schema(frame)$run_lengths), c(3L, 2L))
  expected <- rbind(
    t(apply(as.array(run_1), 4L, function(volume) volume[support])),
    t(apply(as.array(run_2), 4L, function(volume) volume[support]))
  )
  expect_equal(collect_assay(frame, dimnames = FALSE), expected)

  single <- as_fmri_frame(run_1, mask = mask, TR = 2)
  expect_equal(collect_assay(single, dimnames = FALSE), expected[1:3, , drop = FALSE])
  expect_identical(observation_ids(single), sprintf("run-1_vol-%04d", 1:3))

  # A volume round-trips through the frame's space.
  map <- spatial_map(single, observation = 2L)
  expect_equal(as.vector(map)[support], expected[2L, ])
})

test_that("a volume_space is accepted as the NeuroVec mask", {
  run <- .toy_volume(2L)
  mask <- .toy_mask()
  from_vol <- as_fmri_frame(run, mask = mask)
  from_space <- as_fmri_frame(run, mask = space(from_vol))

  expect_true(same_space(space(from_vol), space(from_space))$same)
  expect_equal(collect_assay(from_space), collect_assay(from_vol))
})

test_that("NeuroVec input rejects masks that are missing, empty, or off-grid", {
  run <- .toy_volume(2L)

  expect_error(as_fmri_frame(run), class = "fmridataset_error_config")
  expect_error(as_fmri_frame(run, mask = "mask.nii"), class = "fmridataset_error_config")
  expect_error(
    as_fmri_frame(run, mask = .toy_mask(active = integer())),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(
    as_fmri_frame(run, mask = .toy_mask(dim3 = c(2L, 2L, 3L))),
    class = "fmridataset_error_space_mismatch"
  )
  # Same dimensions but another affine is a different space.
  shifted <- neuroim2::LogicalNeuroVol(
    array(TRUE, c(2L, 2L, 2L)),
    neuroim2::NeuroSpace(c(2L, 2L, 2L), origin = c(10, 0, 0))
  )
  expect_error(
    as_fmri_frame(run, mask = shifted),
    class = "fmridataset_error_space_mismatch"
  )
  expect_error(
    as_fmri_frame(list(run, .toy_volume(2L, dim3 = c(2L, 2L, 3L))), mask = .toy_mask()),
    class = "fmridataset_error_space_mismatch"
  )
})

test_that("lists must hold NeuroVec runs", {
  expect_error(as_fmri_frame(list(), mask = .toy_mask()), class = "fmridataset_error_config")
  expect_error(
    as_fmri_frame(list(.toy_volume(2L), matrix(0, 2, 2)), mask = .toy_mask()),
    class = "fmridataset_error_config"
  )
})

test_that("read_nifti_bold builds a lazy frame matching the NIfTI source", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  frame <- read_nifti_bold(c(path, path), mask = path, TR = 2)
  source <- nifti_source(c(path, path), path)

  expect_s3_class(assay(frame)$source, "nifti_array_source")
  expect_identical(source_fingerprint(assay(frame)$source), source_fingerprint(source))
  expect_true(same_space(space(frame), nifti_source_space(source))$same)
  expect_equal(unname(temporal_schema(frame)$run_lengths), c(4L, 4L))
  expect_equal(collect_assay(frame, dimnames = FALSE), source_read(source), tolerance = 0)

  # An in-memory mask and an explicit run split describe the same data.
  mask <- suppressWarnings(neuroim2::read_vol(path))
  resplit <- read_nifti_bold(c(path, path), mask = mask, run_length = c(2, 6))
  expect_equal(unname(temporal_schema(resplit)$run_lengths), c(2L, 6L))
  expect_identical(feature_ids(resplit), feature_ids(frame))

  expect_error(read_nifti_bold(path), class = "fmridataset_error_config")
  expect_error(
    read_nifti_bold(path, mask = path, run_length = 3),
    class = "fmridataset_error_config"
  )
})

test_that("read_nifti_bold reads no voxel values at construction", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  reads <- 0L
  testthat::local_mocked_bindings(
    .nifti_read_vec = function(...) {
      reads <<- reads + 1L
      stop("no voxel read is expected")
    },
    .package = "fmridataset"
  )
  frame <- read_nifti_bold(path, mask = path, TR = 2)
  expect_identical(dim(frame)[[1L]], 4L)
  expect_identical(reads, 0L)
})

test_that("a LatentNeuroVec becomes scores over a synthesis-only basis space", {
  skip_if_not_installed("fmrilatent")
  mask <- neuroim2::LogicalNeuroVol(
    array(TRUE, c(2L, 2L, 1L)), neuroim2::NeuroSpace(c(2L, 2L, 1L))
  )
  loadings <- Matrix::Matrix(
    matrix(c(1, 0, 0, 1, 1, 1, 2, -1), nrow = 4, byrow = TRUE),
    sparse = TRUE
  )
  scores <- Matrix::Matrix(matrix(c(1, 0, 0, 1, 2, -1), nrow = 3, byrow = TRUE))
  latent <- fmrilatent::LatentNeuroVec(
    basis = scores, loadings = loadings,
    space = neuroim2::NeuroSpace(c(2L, 2L, 1L, 3L)), mask = mask,
    offset = rep(0, 4)
  )

  frame <- as_fmri_frame(latent, TR = 2)

  expect_s3_class(space(frame), "basis_space")
  expect_identical(active_assay(frame), "scores")
  expect_identical(feature_ids(frame), c("component-1", "component-2"))
  expect_equal(collect_assay(frame, dimnames = FALSE), unname(as.matrix(scores)))
  expect_identical(n_features(parent_space(space(frame))), 4L)
  expect_equal(
    as.matrix(collect_assay(frame) %*% Matrix::t(basis_synthesis(space(frame)))),
    as.matrix(scores %*% Matrix::t(loadings)),
    ignore_attr = TRUE
  )

  named <- as_fmri_frame(latent, component_ids = c("pc1", "pc2"), run_length = c(1, 2))
  expect_identical(feature_ids(named), c("pc1", "pc2"))
  expect_equal(unname(temporal_schema(named)$run_lengths), c(1L, 2L))
  expect_error(
    as_fmri_frame(latent, component_ids = c("pc1", "pc1")),
    class = "fmridataset_error_config"
  )
  # A latent object is not a NeuroVec run.
  expect_error(
    as_fmri_frame(list(latent), mask = mask),
    class = "fmridataset_error_config"
  )
})
