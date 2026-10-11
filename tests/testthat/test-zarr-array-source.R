.fake_zarr_runtime <- function(data, chunks = dim(data), dtype = "float64") {
  state <- new.env(parent = emptyenv())
  state$data <- data
  state$shape <- as.integer(dim(data))
  state$chunks <- as.integer(chunks)
  state$dtype <- dtype
  state$reads <- list()
  state$closed <- FALSE
  state
}

test_that("Zarr sources are serializable logical descriptors", {
  reference <- matrix(as.double(1:30), 5, 6)
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(reference),
    dtype = "float64",
    chunks = c(2, 3)
  )

  expect_s3_class(source, "zarr_array_source")
  expect_invisible(validate_array_source(source))
  expect_identical(source_shape(source), c(5L, 6L))
  expect_identical(source_chunks(source), c(2L, 3L))
  expect_identical(source$physical_axes, c("observation", "feature"))
  expect_false(contains_runtime_state(source))
  restored <- unserialize(serialize(source, NULL))
  expect_identical(source_fingerprint(restored), source_fingerprint(source))
})

test_that("Zarr sources discover physical metadata without retaining handles", {
  runtime <- .fake_zarr_runtime(matrix(1:12, 3, 4), c(2, 3), "int32")

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      source <- zarr_array_source("fixture.zarr")
      expect_identical(source_shape(source), c(3L, 4L))
      expect_identical(source_dtype(source), "int32")
      expect_identical(source_chunks(source), c(2L, 3L))
      expect_true(runtime$closed)
      expect_false(contains_runtime_state(source))
    }
  )
})

test_that("Zarr source reads preserve arbitrary order and reject repeats", {
  reference <- matrix(as.double(1:42), 6, 7)
  runtime <- .fake_zarr_runtime(reference, c(2, 3))
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(reference),
    dtype = "float64",
    chunks = c(2, 3)
  )

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_read = function(handle, selection) {
      handle$reads[[length(handle$reads) + 1L]] <- selection
      handle$data[
        seq.int(selection[[1L]][[1L]], selection[[1L]][[2L]]),
        seq.int(selection[[2L]][[1L]], selection[[2L]][[2L]]),
        drop = FALSE
      ]
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      rows <- c(6L, 1L, 2L, 4L)
      cols <- c(7L, 2L, 1L)
      expect_error(
        source_read(source, c(6L, 1L, 6L), cols),
        class = "fmridataset_error_alignment"
      )
      expect_equal(
        source_read(source, rows, cols),
        reference[rows, cols, drop = FALSE],
        tolerance = 0
      )
      expect_true(runtime$closed)
      # The bounding box fits the target block size, so the scattered
      # selection is one backend read subset in memory.
      expect_length(runtime$reads, 1L)
      expect_true(all(vapply(runtime$reads, function(x) {
        all(vapply(x, function(axis) diff(axis) >= 0L, logical(1)))
      }, logical(1))))
    }
  )
})

test_that("Zarr source translates feature-first physical arrays", {
  logical <- matrix(as.double(1:20), 4, 5)
  physical <- t(logical)
  runtime <- .fake_zarr_runtime(physical, c(3, 2))
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(logical),
    dtype = "float64",
    chunks = c(2, 3),
    physical_axes = c("feature", "observation")
  )

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_read = function(handle, selection) {
      handle$data[
        seq.int(selection[[1L]][[1L]], selection[[1L]][[2L]]),
        seq.int(selection[[2L]][[1L]], selection[[2L]][[2L]]),
        drop = FALSE
      ]
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      expect_equal(
        source_read(source, c(4L, 1L), c(5L, 2L, 1L)),
        logical[c(4L, 1L), c(5L, 2L, 1L), drop = FALSE],
        tolerance = 0
      )
    }
  )
})

test_that("Zarr handles close and reject changed physical metadata", {
  reference <- matrix(as.double(1:12), 3, 4)
  runtime <- .fake_zarr_runtime(reference, c(2, 2))
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(reference),
    dtype = "float64",
    chunks = c(2, 2)
  )

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      handle <- source_open(source)
      expect_s3_class(handle, "zarr_array_source_handle")
      expect_false(runtime$closed)
      expect_true(source_close(handle))
      expect_true(runtime$closed)
    }
  )

  runtime$closed <- FALSE
  runtime$shape <- c(4L, 4L)
  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      condition <- expect_error(
        source_open(source),
        class = "fmridataset_error_source_stale"
      )
      expect_false(inherits(condition, "fmridataset_error_backend_io"))
      expect_identical(condition$source$type, "zarr_array_source")
      expect_identical(condition$source$uri, source$uri)
      expect_identical(condition$expected$shape, c(3L, 4L))
      expect_identical(condition$actual$shape, c(4L, 4L))
      expect_true(runtime$closed)
    }
  )
})

test_that("Zarr source validates contracts before runtime access", {
  expect_error(
    zarr_array_source("fixture.zarr", shape = c(2, 3), dtype = "float64"),
    "provided together"
  )
  expect_error(
    zarr_array_source(
      "fixture.zarr",
      shape = c(2, 3), dtype = "float64",
      chunks = c(2, 3), physical_axes = c("observation", "observation")
    ),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    zarr_array_source(
      "fixture.zarr",
      shape = c(2, 3), dtype = "string",
      chunks = c(2, 3)
    ),
    class = "fmridataset_error_source_contract"
  )
  with_mocked_bindings(
    .zarr_assert_available = function() {
      .frame_abort(
        "The optional zarr package is required to open a Zarr ArraySource.",
        "fmridataset_error_backend_io"
      )
    },
    .package = "fmridataset",
    {
      expect_error(
        source_read(
          zarr_array_source(
            "fixture.zarr",
            shape = c(2, 3), dtype = "float64",
            chunks = c(2, 3)
          ),
          1L, 1L
        ),
        "zarr package"
      )
    }
  )
})

test_that("Zarr source is a reconstructible delarr provider", {
  reference <- matrix(as.double(1:20), 4, 5)
  runtime <- .fake_zarr_runtime(reference, c(2, 3))
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(reference),
    dtype = "float64",
    chunks = c(2, 3)
  )

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) runtime,
    .zarr_provider_metadata = function(handle) {
      list(
        shape = handle$shape,
        chunks = handle$chunks,
        dtype = handle$dtype
      )
    },
    .zarr_provider_read = function(handle, selection) {
      handle$data[
        seq.int(selection[[1L]][[1L]], selection[[1L]][[2L]]),
        seq.int(selection[[2L]][[1L]], selection[[2L]][[2L]]),
        drop = FALSE
      ]
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      plan <- unserialize(serialize(as_delarr(source), NULL))
      expect_equal(
        delarr::collect(plan[c(4L, 1L), c(5L, 2L)]),
        reference[c(4L, 1L), c(5L, 2L), drop = FALSE],
        tolerance = 0
      )
    }
  )
})

test_that("Zarr source reads a real current-driver store", {
  skip_if_not_installed("zarr", minimum_version = "0.4.2")
  skip_if_not_installed("blosc")
  reference <- matrix(as.double(1:42), 6, 7)
  path <- tempfile("frame-source-", fileext = ".zarr")
  on.exit(unlink(path, recursive = TRUE), add = TRUE)
  zarr::as_zarr(reference, location = path)

  source <- zarr_array_source(path)
  expect_array_source_conformance(source, reference)
  rows <- c(6L, 1L, 2L, 5L)
  features <- c(7L, 2L, 1L, 4L)
  expect_equal(
    source_read(source, rows, features),
    reference[rows, features, drop = FALSE],
    tolerance = 0
  )
  expect_equal(
    delarr::collect(as_delarr(source)[rows, features]),
    reference[rows, features, drop = FALSE],
    tolerance = 0
  )
})

test_that("Zarr reads consume range selections as one chunk-aligned request", {
  reference <- matrix(as.double(1:42), 6, 7)
  runtime <- .fake_zarr_runtime(reference, c(2, 3))
  source <- zarr_array_source(
    "fixture.zarr",
    shape = dim(reference),
    dtype = "float64",
    chunks = c(2, 3)
  )

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) {
      runtime$closed <- FALSE
      runtime
    },
    .zarr_provider_metadata = function(handle) {
      list(shape = handle$shape, chunks = handle$chunks, dtype = handle$dtype)
    },
    .zarr_provider_read = function(handle, selection) {
      handle$reads[[length(handle$reads) + 1L]] <- selection
      handle$data[
        seq.int(selection[[1L]][[1L]], selection[[1L]][[2L]]),
        seq.int(selection[[2L]][[1L]], selection[[2L]][[2L]]),
        drop = FALSE
      ]
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      # A range view pushes its bounds down as one read; nesting a range in a
      # range stays one read over the composed bounds.
      ranged <- source_view(source, observations = 2:5, features = 3:7)
      expect_equal(source_read(ranged), reference[2:5, 3:7, drop = FALSE], tolerance = 0)
      expect_identical(runtime$reads, list(list(c(2L, 5L), c(3L, 7L))))

      runtime$reads <- list()
      nested <- source_view(ranged, observations = 2:3, features = 2:4)
      expect_equal(source_read(nested), reference[3:4, 4:6, drop = FALSE], tolerance = 0)
      expect_identical(runtime$reads, list(list(c(3L, 4L), c(4L, 6L))))

      # A select-all axis is one read of the whole axis.
      runtime$reads <- list()
      expect_equal(source_read(source_view(source, observations = 4L)), reference[4L, , drop = FALSE], tolerance = 0)
      expect_identical(runtime$reads, list(list(c(4L, 4L), c(1L, 7L))))

      # Arbitrary positions within the target block size are read as their
      # bounding box.
      runtime$reads <- list()
      expect_equal(
        source_read(source, c(6L, 1L, 2L), NULL),
        reference[c(6L, 1L, 2L), , drop = FALSE],
        tolerance = 0
      )
      expect_identical(runtime$reads, list(list(c(1L, 6L), c(1L, 7L))))

      # Above it, positions are decomposed into chunk-aligned regions; runs in
      # chunks that do not touch stay separate reads.
      withr::local_options(fmridataset.target_block_bytes = 8)
      runtime$reads <- list()
      expect_equal(
        source_read(source, c(6L, 1L, 2L), NULL),
        reference[c(6L, 1L, 2L), , drop = FALSE],
        tolerance = 0
      )
      expect_identical(runtime$reads, list(list(c(1L, 2L), c(1L, 7L)), list(c(6L, 6L), c(1L, 7L))))
    }
  )
})

test_that("Zarr reads coalesce scattered selections instead of one read per run pair", {
  logical_ref <- matrix(as.double(seq_len(6 * 7)), 6, 7)
  # Feature-first physical storage with logical chunks of 2 observations by
  # 3 features.
  runtime <- .fake_zarr_runtime(t(logical_ref), c(3, 2))
  runtime$nreads <- 0L

  with_mocked_bindings(
    .zarr_provider_open = function(uri, array_path) {
      runtime$closed <- FALSE
      runtime
    },
    .zarr_provider_metadata = function(handle) {
      list(shape = handle$shape, chunks = handle$chunks, dtype = handle$dtype)
    },
    .zarr_provider_read = function(handle, selection) {
      handle$nreads <- handle$nreads + 1L
      handle$reads[[length(handle$reads) + 1L]] <- selection
      handle$data[
        seq.int(selection[[1L]][[1L]], selection[[1L]][[2L]]),
        seq.int(selection[[2L]][[1L]], selection[[2L]][[2L]]),
        drop = FALSE
      ]
    },
    .zarr_provider_close = function(handle) handle$closed <- TRUE,
    .package = "fmridataset",
    {
      source <- zarr_array_source(
        "fixture.zarr",
        physical_axes = c("feature", "observation")
      )
      expect_identical(source_chunks(source), c(2L, 3L))
      rows <- c(5L, 1L, 3L)
      cols <- c(7L, 1L, 5L, 3L)

      # 3 observation runs x 4 feature runs used to be 12 reads; the bounding
      # box fits the default target block size, so it is now one.
      expect_equal(source_read(source, rows, cols), logical_ref[rows, cols], tolerance = 0)
      expect_identical(runtime$nreads, 1L)

      # Under a tiny target, runs merge only within chunks: features 1 and 3
      # share chunk 1, so the 4 feature runs become 3 regions (3 x 3 reads).
      withr::local_options(fmridataset.target_block_bytes = 8)
      runtime$nreads <- 0L
      expect_equal(source_read(source, rows, cols), logical_ref[rows, cols], tolerance = 0)
      expect_identical(runtime$nreads, 9L)

      # Empty selections never touch the backend and keep the declared type.
      runtime$nreads <- 0L
      empty <- source_read(source, integer(), cols)
      expect_identical(dim(empty), c(0L, 4L))
      expect_identical(storage.mode(empty), "double")
      expect_identical(runtime$nreads, 0L)
    }
  )
})
