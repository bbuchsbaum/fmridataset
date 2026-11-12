# Behaviour added because the vignettes had to work around its absence.

# Register throwaway S3 methods on the package namespace for this test only, so
# dispatch from inside the package finds them and later tests never see them.
.local_source_methods <- function(class, methods, env = parent.frame()) {
  ns <- asNamespace("fmridataset")
  table <- get(".__S3MethodsTable__.", envir = ns)
  for (generic in names(methods)) registerS3method(generic, class, methods[[generic]], envir = ns)
  withr::defer(rm(list = paste0(names(methods), ".", class), envir = table), envir = env)
}

.ergonomic_frame <- function() {
  fmri_frame(
    assays = list(bold = matrix(as.double(1:24), nrow = 4)),
    observations = data.frame(
      .obs_id = sprintf("vol-%d", 1:4), run_id = rep(c("r1", "r2"), each = 2L)
    ),
    space = volume_space(c(3, 2, 1), template = "toy"),
    tables = list(events = event_table(data.frame(event_id = "e1", onset = 0))),
    metadata = list(site = "A")
  )
}

test_that("same_space() is a logical that also carries its report", {
  a <- volume_space(c(2, 2, 1), template = "toy")
  b <- volume_space(c(2, 2, 1), affine = diag(c(2, 2, 2, 1)), template = "toy")
  same <- same_space(a, a)
  differ <- same_space(a, b)

  expect_true(if (same) TRUE else FALSE)
  expect_false(if (differ) TRUE else FALSE)
  expect_true(isTRUE(same))
  expect_true(same$same)
  expect_false(differ$same)
  expect_true(differ$same_class)
  expect_false(differ$same_digest)
  expect_true(differ$same_feature_ids)
  expect_identical(differ$reason, "Feature spaces differ: digest differs.")
  expect_identical(differ[["same_digest"]], FALSE)
  expect_setequal(names(as.list(differ)), names(differ))
  expect_match(capture.output(print(differ))[[1L]], "^<space_comparison> FALSE: digest differs")
  expect_identical(capture.output(print(same)), "<space_comparison> TRUE: the same feature space")
  expect_error(assert_same_space(a, b), "digest differs", class = "fmridataset_error_space_mismatch")
})

test_that("collect_assay() and as.matrix() label values with the axis IDs", {
  frame <- .ergonomic_frame()
  view <- frame[c("vol-4", "vol-1"), c("voxel-6", "voxel-2")]
  values <- collect_assay(view)
  expect_identical(dimnames(values), list(c("vol-4", "vol-1"), c("voxel-6", "voxel-2")))
  expect_identical(unname(values), matrix(c(24, 21, 8, 5), 2))
  expect_null(dimnames(collect_assay(view, dimnames = FALSE)))
  expect_identical(as.matrix(view), values)
  expect_identical(as.matrix(view, dimnames = FALSE), unname(values))
  expect_error(as.matrix(frame, memory_budget = 1), class = "fmridataset_error_budget")
})

test_that("frames and views report their counts", {
  frame <- .ergonomic_frame()
  expect_identical(n_observations(frame), 4L)
  expect_identical(n_features(frame), 6L)
  view <- select_features(frame, i == 99)
  expect_identical(n_features(view), 0L)
  expect_identical(n_observations(frame[2:1, ]), 2L)
})

test_that("frame metadata, tables, and provenance have accessors", {
  frame <- .ergonomic_frame()
  view <- frame[1:2, ]
  expect_s3_class(frame_metadata(frame), "unaligned_record")
  expect_identical(frame_metadata(view)$site, "A")
  expect_identical(names(frame_tables(view)), "events")
  expect_s3_class(frame_table(frame, "events"), "fmri_event_table")
  expect_error(frame_table(frame, "nope"), "\"events\"", class = "fmridataset_error_alignment")
  expect_error(frame_metadata(list()), class = "fmridataset_error_alignment")
  expect_null(provenance(frame))

  bound <- bind_observations(frame[1:2, ], frame[3:4, ])
  expect_s3_class(provenance(bound), "provenance_graph")
  expect_identical(provenance(bound), bound$provenance)
  expect_identical(length(frame_tables(fmri_frame(list(b = matrix(0, 1, 1)), data.frame(.obs_id = "a")))), 0L)
})

test_that("the semantic digest accepts a frame directly", {
  frame <- .ergonomic_frame()
  expect_identical(
    fds_frame_manifest_digest(frame),
    fds_frame_manifest_digest(fds_frame_manifest(frame))
  )
})

test_that("errors name what disagreed", {
  frame <- .ergonomic_frame()
  expect_error(frame[, c("voxel-1", "voxel-99")], "voxel-99", class = "fmridataset_error_alignment")
  expect_error(
    fmri_frame(
      list(bold = matrix(0, 2, 5)),
      data.frame(.obs_id = c("a", "b")),
      space = volume_space(c(3, 2, 1))
    ),
    "5 columns but the feature space has 6 features",
    class = "fmridataset_error_alignment"
  )
  one <- index_space(1, ids = "f1")
  site_a <- fmri_frame(list(b = matrix(0, 1, 1)), data.frame(.obs_id = "a"), space = one, metadata = list(site = "A", scanner = "X"))
  site_b <- fmri_frame(list(b = matrix(0, 1, 1)), data.frame(.obs_id = "b"), space = one, metadata = list(site = "B", scanner = "X"))
  expect_error(
    bind_observations(site_a, site_b, metadata_policy = "identical"),
    "different metadata in 'site'",
    class = "fmridataset_error_alignment"
  )
})

test_that("source methods receive normalized selectors", {
  seen <- new.env()
  reference <- matrix(as.double(1:12), 3)
  source_read.echo_source <- function(x, observations = NULL, features = NULL, ...) {
    seen$observations <- observations
    seen$features <- features
    reference[observations %||% seq_len(3), features %||% seq_len(4), drop = FALSE]
  }
  `%||%` <- function(x, y) if (is.null(x)) y else x
  source <- structure(memory_source(reference), class = c("echo_source", "memory_source", "array_source"))
  .local_source_methods("echo_source", list(source_read = source_read.echo_source))

  source_read(source, c(TRUE, FALSE, TRUE), -1L)
  expect_identical(seen$observations, c(1L, 3L))
  expect_identical(seen$features, 2:4)
  source_read(source, c(0, 2), NULL)
  expect_identical(seen$observations, 2L)
  expect_null(seen$features)
  source_read(source, 1:3, 4:1)
  expect_null(seen$observations)
  expect_identical(seen$features, 4:1)
})

test_that("descriptors need no close method of their own", {
  handle_closed <- new.env()
  reference <- matrix(as.double(1:6), 2)
  src <- structure(
    list(shape = c(2L, 3L), fingerprint = "toy-1", values = reference),
    class = c("closeless_source", "array_source")
  )
  methods <- list(
    source_shape = function(x, ...) x$shape,
    source_dtype = function(x, ...) "float64",
    source_chunks = function(x, ...) x$shape,
    source_capabilities = function(x, ...) c("block_slice", "serializable"),
    source_fingerprint = function(x, ...) x$fingerprint,
    source_open = function(x, ...) structure(list(source = x), class = c("closeless_handle", "array_source_handle")),
    source_read = function(x, observations = NULL, features = NULL, ...) {
      x$values[observations %||% seq_len(2), features %||% seq_len(3), drop = FALSE]
    }
  )
  `%||%` <- function(x, y) if (is.null(x)) y else x
  .local_source_methods("closeless_source", methods)
  .local_source_methods("closeless_handle", list(source_close = function(x, ...) {
    handle_closed$value <- TRUE
    invisible(TRUE)
  }))

  expect_invisible(validate_array_source(src))
  expect_invisible(source_close(src))
  source_close(source_open(src))
  expect_true(handle_closed$value)
})

test_that("check_array_source() certifies conforming sources and names failures", {
  reference <- matrix(as.double(1:20), 4)
  expect_no_error(report <- check_array_source(memory_source(reference), reference))
  expect_s3_class(report, "source_conformance")
  expect_true(all(report$passed))
  expect_identical(
    report$law,
    c("descriptor", "serialization", "lifecycle", "reads", "selectors", "views", "content")
  )
  sharded <- row_sharded_source(list(memory_source(reference[1:2, ]), memory_source(reference[3:4, ])))
  expect_no_error(check_array_source(sharded, reference))

  err <- tryCatch(check_array_source(memory_source(reference), reference * 2), error = identity)
  expect_s3_class(err, "fmridataset_error_source_contract")
  expect_match(conditionMessage(err), "reads: A corner read does not match the reference")
  quiet <- check_array_source(memory_source(reference), reference * 2, error = FALSE)
  expect_false(all(quiet$passed))
  expect_match(capture.output(print(quiet))[[1L]], "^<source_conformance> 3 of 7 laws pass")

  leaky <- memory_source(reference)
  leaky$loader <- function() NULL
  broken <- check_array_source(leaky, reference, error = FALSE)
  expect_false(broken$passed[[1L]])
  expect_true(all(is.na(broken$passed[-1L])))
  err <- tryCatch(check_array_source(leaky, reference), error = identity)
  expect_match(conditionMessage(err), "fails 1 conformance law:")
})

test_that("a single hemisphere label covers a one-hemisphere mesh", {
  left <- surface_space(sprintf("v%d", 1:4), hemisphere = "left")
  expect_identical(left$hemisphere, rep("left", 4))
  expect_error(surface_space(sprintf("v%d", 1:4), hemisphere = c("left", "right")),
    class = "fmridataset_error_space_mismatch"
  )
})

test_that("the bundled fMRIPrep tree imports with a bold assay", {
  skip_if_not(requireNamespace("bidser", quietly = TRUE) && utils::packageVersion("bidser") >= "0.5.0")
  dir <- withr::local_tempdir()
  untar(system.file("extdata", "fmriprep-toy.tar.gz", package = "fmridataset"), exdir = dir)
  bold <- read_bids_bold(file.path(dir, "fmriprep-toy"), subject = "01")
  expect_identical(active_assay(bold), "bold")
  expect_identical(dim(bold), c(6L, 3L))
  renamed <- read_bids_bold(file.path(dir, "fmriprep-toy"), subject = "01", assay = "signal")
  expect_identical(names(assays(renamed)), "signal")
})

test_that("studies and collections say how to save them", {
  frame <- .ergonomic_frame()
  study <- fmri_study(list(main = frame))
  expect_error(write_frame(study, tempfile()), "fds_study_manifest", class = "fmridataset_error_backend_io")
  expect_error(
    write_frame(fmri_collection(list(a = frame)), tempfile()),
    "not a fmri_collection",
    class = "fmridataset_error_backend_io"
  )
})

test_that("a study names the entities it needs to share", {
  subjects <- entity_frame(data.frame(subject_id = c("s1", "s2")), key = "subject_id")
  frame <- fmri_frame(
    list(b = matrix(0, 2, 1)),
    data.frame(.obs_id = c("a", "b"), subject_id = c("s1", "s2")),
    entities = list(subject = subjects),
    relations = list(os = key_relation("subject_id", target = "subject"))
  )
  expect_error(fmri_study(list(main = frame)), "entities = list\\(subject = ...\\)")
})
