test_that("memory_source implements the source contract", {
  m <- matrix(seq_len(12), 3, 4)
  x <- memory_source(m)

  expect_s3_class(x, "array_source")
  expect_identical(source_shape(x), c(3L, 4L))
  expect_identical(source_dtype(x), "float64")
  expect_true(all(c("row_slice", "column_slice", "block_slice") %in% source_capabilities(x)))
  expect_equal(source_read(x, c(3, 1), c(4, 2)), m[c(3, 1), c(4, 2), drop = FALSE])
})

test_that("source contracts expose complete normalized descriptors", {
  x <- memory_source(matrix(as.double(1:12), 3, 4), chunks = c(2, 3))
  descriptor <- source_descriptor(x)

  expect_identical(
    names(descriptor),
    c("shape", "dtype", "chunks", "capabilities", "fingerprint")
  )
  expect_identical(descriptor$shape, c(3L, 4L))
  expect_identical(descriptor$dtype, "float64")
  expect_identical(descriptor$chunks, c(2L, 3L))
  expect_true(all(c("block_slice", "serializable") %in% descriptor$capabilities))
  expect_match(descriptor$fingerprint, "^[0-9a-f]{64}$")
  expect_invisible(validate_array_source(x))
})

test_that("source contract validation rejects unsafe or ambiguous descriptors", {
  x <- memory_source(matrix(1:6, 2, 3))

  bad_dtype <- x
  bad_dtype$dtype <- "mystery"
  expect_error(
    validate_array_source(bad_dtype),
    class = "fmridataset_error_source_contract"
  )
  bad_chunks <- x
  bad_chunks$chunks <- c(3L, 1L)
  expect_error(
    validate_array_source(bad_chunks),
    class = "fmridataset_error_source_contract"
  )
  bad_capabilities <- x
  bad_capabilities$capabilities <- "row_slice"
  expect_error(
    validate_array_source(bad_capabilities),
    class = "fmridataset_error_source_contract"
  )
  unsafe <- x
  unsafe$loader <- function() NULL
  expect_error(
    validate_array_source(unsafe),
    class = "fmridataset_error_source_contract"
  )
  expect_error(memory_source(matrix(1:4, 2), dtype = "unknown"), "Unsupported")
})

test_that("fingerprints encode logical source transformations", {
  x <- memory_source(matrix(1:12, 3, 4), chunks = c(2, 2))
  same <- unserialize(serialize(x, NULL))
  reordered <- source_view(x, observations = c(3, 1), features = c(4, 2))
  changed_chunks <- memory_source(matrix(1:12, 3, 4), chunks = c(3, 4))

  expect_identical(source_fingerprint(x), source_fingerprint(same))
  expect_false(identical(source_fingerprint(x), source_fingerprint(reordered)))
  expect_false(identical(source_fingerprint(x), source_fingerprint(changed_chunks)))
  expect_identical(
    source_fingerprint(counting_source(x)),
    source_fingerprint(x)
  )
  expect_false(identical(
    source_fingerprint(fault_source(x, "read")),
    source_fingerprint(fault_source(x, "open"))
  ))
})

test_that("memory_source handles empty slices", {
  x <- memory_source(matrix(1:6, 2, 3))
  expect_identical(dim(source_read(x, integer(), 1:2)), c(0L, 2L))
  expect_identical(dim(source_read(x, 1:2, integer())), c(2L, 0L))
})

test_that("counting_source records requested assay bytes", {
  x <- counting_source(memory_source(matrix(1:12, 3, 4)))

  expect_equal(source_counts(x)$bytes, 0)
  source_shape(x)
  expect_equal(source_counts(x)$bytes, 0)
  source_read(x, 1:2, 1:3)
  expect_equal(source_counts(x)$reads, 1)
  expect_equal(source_counts(x)$values, 6)
  expect_gt(source_counts(x)$bytes, 0)
})

test_that("instrumentation descriptors remain serializable developer tools", {
  counted <- counting_source(memory_source(matrix(1:4, 2L)))
  faulted <- fault_source(memory_source(matrix(1:4, 2L)), stage = "read")

  expect_false(contains_runtime_state(counted))
  expect_false(contains_runtime_state(faulted))
  expect_identical(source_counts(counted)$bytes, 0)
  expect_identical(
    source_fingerprint(unserialize(serialize(counted, NULL))),
    source_fingerprint(counted)
  )
  expect_error(source_read(faulted, 1L, 1L), class = "fmridataset_error_backend_io")
})

test_that("fault_source fails at the configured lifecycle stage", {
  x <- fault_source(memory_source(matrix(1:4, 2)), stage = "read")
  expect_error(source_read(x, 1, 1), class = "fmridataset_error_backend_io")
})

test_that("fault_source handles keep the fault stage in the read and close path", {
  child <- counting_source(memory_source(matrix(as.double(1:4), 2)))

  read_handle <- source_open(fault_source(child, stage = "read"))
  expect_s3_class(read_handle, "fault_source_handle")
  expect_error(source_read(read_handle, 1, 1), class = "fmridataset_error_backend_io")
  source_close(read_handle)

  native_handle <- source_open(fault_source(child, stage = "native_read"))
  expect_error(
    source_read_native(native_handle, 1),
    "Injected native_read failure",
    class = "fmridataset_error_backend_io"
  )
  expect_equal(source_read(native_handle, 1, 1), matrix(1))
  source_close(native_handle)

  reset_source_counts(child)
  close_handle <- source_open(fault_source(child, stage = "close"))
  expect_equal(source_read(close_handle, 2, 2), matrix(4))
  expect_error(source_close(close_handle), class = "fmridataset_error_backend_io")
  # The child handle is still released when the close stage is faulted.
  expect_equal(source_counts(child)$opens, 1)
  expect_equal(source_counts(child)$closes, 1)
})

test_that("a counting source missing from the registry reports every counter", {
  x <- counting_source(memory_source(matrix(as.double(1:6), 2)))
  rm(list = x$counter_id, envir = fmridataset:::.source_counter_registry)
  expect_named(source_counts(x), names(fmridataset:::.source_counter_zero()))
  source_read(x)
  counts <- source_counts(x)
  expect_equal(counts$reads, 1)
  expect_equal(counts$storage_bytes, 48)
  expect_equal(counts$output_bytes, 48)
})

test_that("reads realize the declared dtype's storage mode on every path", {
  integers <- memory_source(matrix(1:6, 2))
  expect_identical(source_dtype(integers), "float64")
  expect_identical(storage.mode(source_read(integers)), "double")
  expect_identical(storage.mode(source_read(source_open(integers), 1, 1)), "double")
  expect_identical(storage.mode(source_read(source_view(integers, 2:1))), "double")
  expect_identical(
    source_read(integers),
    source_read(row_sharded_source(list(integers)))
  )

  bits <- memory_source(matrix(c(0, 1, 1, NA), 2), dtype = "logical")
  expect_identical(source_read(bits), matrix(c(FALSE, TRUE, TRUE, NA), 2))
  expect_identical(storage.mode(source_read(bits, integer(), 1)), "logical")

  bytes <- memory_source(matrix(as.raw(1:4), 2), dtype = "uint8")
  expect_identical(source_read(bytes), matrix(as.double(1:4), 2))
})

test_that("memory_source rejects data that cannot hold the declared dtype", {
  expect_error(
    memory_source(matrix(c(1.5, 300), 1), dtype = "logical"),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    memory_source(matrix(c(1, 300), 1), dtype = "uint8"),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    memory_source(matrix(c(-1, 2), 1), dtype = "uint16"),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    memory_source(matrix(c(0.5, 2), 1), dtype = "int32"),
    class = "fmridataset_error_source_contract"
  )
  expect_error(
    memory_source(matrix(letters[1:2], 1), dtype = "float64"),
    class = "fmridataset_error_source_contract"
  )
  expect_s3_class(
    memory_source(matrix(c(0, 255, NA), 1), dtype = "uint8"),
    "memory_source"
  )
  expect_s3_class(
    memory_source(matrix(c(1.5, 300), 1), dtype = "float32"),
    "memory_source"
  )
})

test_that("realization costs accept open handles", {
  src <- memory_source(matrix(as.double(1:6), 2))
  handle <- source_open(src)
  on.exit(source_close(handle), add = TRUE)
  expect_identical(
    unclass(source_realization_cost(handle, 1, 2:3)),
    unclass(source_realization_cost(src, 1, 2:3))
  )
})

test_that("row-sharded realization costs combine their shards' traits", {
  memory <- memory_source(matrix(as.double(1:6), 2))
  all_memory <- fmridataset:::.source_cost_traits(row_sharded_source(list(memory, memory)))
  expect_true(all_memory$already_realized)
  expect_false(all_memory$compressed)

  zarr <- zarr_array_source(
    "fixture.zarr",
    shape = c(2L, 3L), dtype = "float64", chunks = c(1L, 3L)
  )
  mixed <- row_sharded_source(list(memory, zarr))
  traits <- fmridataset:::.source_cost_traits(mixed)
  expect_false(traits$already_realized)
  expect_true(traits$compressed)
  expect_equal(
    source_realization_cost(mixed)$estimated_peak_bytes,
    source_realization_cost(zarr, 1:2)$estimated_peak_bytes * 2
  )
})

test_that("row_sharded_source accepts extension sources reporting double shapes", {
  source_shape.double_shape_source <- function(x, ...) c(2, 3)
  source_dtype.double_shape_source <- function(x, ...) "float64"
  source_chunks.double_shape_source <- function(x, ...) c(1, 3)
  source_capabilities.double_shape_source <- function(x, ...) {
    c("block_slice", "serializable")
  }
  source_fingerprint.double_shape_source <- function(x, ...) "double-shape"
  source_open.double_shape_source <- function(x, ...) x
  source_read.double_shape_source <- function(x, observations = NULL,
                                              features = NULL, ...) {
    matrix(as.double(1:6), 2, 3)[observations %||% 1:2, features %||% 1:3, drop = FALSE]
  }
  source_close.double_shape_source <- function(x, ...) invisible(TRUE)
  # Register for dispatch from inside the package, and unregister afterwards
  # so the conformance inventory never sees this throwaway class.
  table <- get(".__S3MethodsTable__.", envir = asNamespace("fmridataset"))
  generics <- c(
    "source_shape", "source_dtype", "source_chunks", "source_capabilities",
    "source_fingerprint", "source_open", "source_read", "source_close"
  )
  for (generic in generics) {
    name <- paste0(generic, ".double_shape_source")
    registerS3method(generic, "double_shape_source", get(name), envir = asNamespace("fmridataset"))
  }
  withr::defer(rm(
    list = paste0(generics, ".double_shape_source"),
    envir = table
  ))

  ext <- structure(list(), class = c("double_shape_source", "array_source"))
  sharded <- row_sharded_source(list(ext, ext))
  expect_identical(source_shape(sharded), c(4L, 3L))
  expect_identical(source_chunks(sharded), c(1L, 3L))
  expect_equal(source_read(sharded, c(4L, 1L), 2L), matrix(c(4, 3), 2))
})

test_that("array sources construct runtime delarr plans", {
  m <- matrix(seq_len(12), 3, 4)
  x <- as_delarr(memory_source(m))
  expect_s3_class(x$seed, "delarr_provider_seed")
  expect_false(contains_runtime_state(x$seed))
  expect_equal(delarr::collect(x[3:1, c(4, 2)]), m[3:1, c(4, 2), drop = FALSE])
  restored <- unserialize(serialize(x, NULL))
  expect_equal(
    delarr::collect(restored[3:1, c(4, 2)]),
    m[3:1, c(4, 2), drop = FALSE]
  )
})
