# Budgets exist to bound what a read will occupy in memory, but they were
# computed from the STORAGE width of the assay's dtype. Every read realizes an
# R vector, so a float32 assay was budgeted at half its true size and a uint8
# assay at an eighth: collect_assay(fr, memory_budget = 48000) on a float32
# source returned an object of 64216 bytes, 34% over the stated ceiling.

test_that("realized width reflects what R actually allocates", {
  # Every numeric dtype is realized as an R double.
  for (dtype in c(
    "uint8", "int8", "uint16", "int16", "float16", "bfloat16",
    "uint32", "int32", "float32", "uint64", "int64", "float64"
  )) {
    expect_equal(
      fmridataset:::.realized_dtype_bytes(dtype), 8,
      info = dtype
    )
  }
  expect_equal(fmridataset:::.realized_dtype_bytes("logical"), 4)
  expect_equal(fmridataset:::.realized_dtype_bytes("complex64"), 16)
  expect_equal(fmridataset:::.realized_dtype_bytes("complex128"), 16)
})

test_that("realized width is never below storage width for numeric dtypes", {
  for (dtype in c("float32", "float64", "int32", "uint8", "complex128")) {
    expect_gte(
      fmridataset:::.realized_dtype_bytes(dtype),
      fmridataset:::.dtype_bytes(dtype)
    )
  }
})

test_that("realization cost separates storage, output, conversion, and peak", {
  float32 <- fmridataset:::.realization_cost_from_shape(
    c(10, 20), "float32",
    already_realized = FALSE
  )
  float16 <- fmridataset:::.realization_cost_from_shape(
    c(10, 20), "float16",
    already_realized = FALSE
  )

  expect_equal(float32$storage_dtype, "float32")
  expect_equal(float32$realized_dtype, "double")
  expect_equal(float32$storage_bytes, 10 * 20 * 4)
  expect_equal(float32$estimated_output_bytes, 10 * 20 * 8)
  expect_equal(float32$selection_buffer_bytes, float32$estimated_output_bytes)
  expect_equal(float32$conversion_buffer_bytes, float32$estimated_output_bytes)
  expect_equal(
    float32$estimated_peak_bytes,
    float32$estimated_output_bytes + float32$selection_buffer_bytes +
      float32$conversion_buffer_bytes
  )
  expect_lt(float16$storage_bytes, float32$storage_bytes)
  expect_equal(float16$estimated_output_bytes, float32$estimated_output_bytes)
})

test_that("compressed sources include a decompression buffer", {
  source <- zarr_array_source(
    "memory://budget-test",
    shape = c(5, 7),
    dtype = "float32",
    chunks = c(2, 3)
  )
  cost <- source_realization_cost(source)

  expect_equal(cost$decompression_buffer_bytes, cost$storage_bytes)
  expect_equal(
    cost$estimated_peak_bytes,
    cost$estimated_output_bytes + cost$selection_buffer_bytes +
      cost$conversion_buffer_bytes + cost$decompression_buffer_bytes
  )
})

test_that("memory sources do not invent conversion buffers", {
  source <- memory_source(matrix(seq_len(20), 4, 5), dtype = "float32")
  cost <- source_realization_cost(source, observations = 1:2, features = 1:3)

  expect_equal(cost$shape, c(2L, 3L))
  expect_equal(cost$storage_bytes, 2 * 3 * 4)
  expect_equal(cost$estimated_output_bytes, 2 * 3 * 8)
  expect_equal(cost$selection_buffer_bytes, 0)
  expect_equal(cost$conversion_buffer_bytes, 0)
  expect_equal(cost$estimated_peak_bytes, cost$estimated_output_bytes)
})

test_that("an unsupported dtype is still rejected with the source-contract error", {
  expect_error(
    fmridataset:::.realized_dtype_bytes("float128"),
    class = "fmridataset_error_source_contract"
  )
})

test_that("collect_assay never returns an object larger than its budget", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")

  source <- nifti_array_source(path, path)
  expect_identical(source_dtype(source), "float32")

  shape <- source_shape(source)
  frame <- fmri_frame(
    assays = list(bold = source),
    observations = data.frame(.obs_id = sprintf("t%03d", seq_len(shape[[1L]]))),
    space = nifti_source_space(source)
  )

  n_values <- prod(as.double(shape))
  storage_bytes <- n_values * 4 # float32 on disk
  cost <- source_realization_cost(source)
  realized_bytes <- cost$estimated_output_bytes # R doubles in memory

  # A budget between the two must now be refused. Before the fix this was
  # allowed, and the returned object exceeded it.
  midpoint <- (storage_bytes + realized_bytes) / 2
  expect_error(
    collect_assay(frame, memory_budget = midpoint),
    class = "fmridataset_error_budget"
  )

  # The peak estimate includes conversion buffers as well as the retained
  # output. A budget at that conservative peak is honoured.
  collected <- collect_assay(frame, memory_budget = cost$estimated_peak_bytes)
  expect_equal(dim(collected), as.integer(shape))
  expect_lte(as.numeric(utils::object.size(collected)) - 216, realized_bytes)
})

test_that("the budget error names the shortfall, the dtype and the ceiling", {
  frame <- make_frame_fixture()$frame
  err <- tryCatch(
    collect_assay(frame, memory_budget = 1),
    error = function(e) e
  )
  expect_s3_class(err, "fmridataset_error_budget")
  expect_match(conditionMessage(err), "memory_budget of 1 bytes")
  expect_match(conditionMessage(err), "estimated to retain")
  expect_match(conditionMessage(err), "peak at")
  expect_equal(err$storage_dtype, "float64")
  expect_equal(err$realized_dtype, "double")
  expect_gt(err$estimated_output_bytes, 1)
  expect_gte(err$estimated_peak_bytes, err$estimated_output_bytes)
  expect_equal(err$memory_budget, 1)
})

test_that("force bypasses the budget", {
  frame <- make_frame_fixture()$frame
  expect_silent(collect_assay(frame, memory_budget = 1, force = TRUE))
})

test_that("block plans size their blocks by realized width", {
  frame <- make_frame_fixture()$frame
  dtype <- assay(frame)$dtype
  realized <- fmridataset:::.realized_dtype_bytes(dtype)

  # A budget that holds fewer than one realized value must be refused, even
  # when it would have held one value at storage width.
  expect_error(
    plan_blocks(frame, memory_budget = realized - 1, target_block_bytes = realized - 1),
    class = "fmridataset_error_budget"
  )

  budget <- realized * 4
  plan <- plan_blocks(frame, memory_budget = budget, target_block_bytes = budget)

  # The plan's own accounting must be on the realized basis, and no block may
  # exceed the ceiling once realized.
  expect_equal(plan$dtype_bytes, realized)
  expect_equal(plan$output_dtype_bytes, realized)
  expect_gte(plan$peak_dtype_bytes, plan$output_dtype_bytes)
  expect_lte(plan$max_peak_bytes, budget)
  expect_lte(plan$max_output_bytes, plan$max_peak_bytes)
  expect_equal(plan$max_block_bytes, plan$max_peak_bytes)
  expect_equal(plan$total_bytes, prod(as.double(plan$shape)) * realized)
  expect_equal(plan$total_output_bytes, plan$total_bytes)
})

test_that("counting sources report storage and realized traffic separately", {
  source <- counting_source(
    memory_source(matrix(seq_len(12), 3, 4), dtype = "float32")
  )
  source_read(source, observations = 1:2, features = 1:3)
  counts <- source_counts(source)

  expect_equal(counts$values, 6)
  expect_equal(counts$storage_bytes, 6 * 4)
  expect_equal(counts$output_bytes, 6 * 8)
  expect_equal(counts$bytes, counts$output_bytes)
})

test_that("as_delarr refuses a realization above its explicit ceiling", {
  source <- memory_source(matrix(seq_len(100), 10, 10), dtype = "float32")
  cost <- source_realization_cost(source)

  expect_error(
    as_delarr(source, memory_budget = cost$estimated_peak_bytes - 1),
    class = "fmridataset_error_budget"
  )
  lazy <- as_delarr(source, memory_budget = cost$estimated_peak_bytes)
  expect_equal(delarr::collect(lazy), source_read(source))
})

test_that("measured vector-heap peak stays within the documented tolerance", {
  source <- memory_source(matrix(0, nrow = 1500, ncol = 1000))
  feature_ids <- sprintf("f-%04d", seq_len(1000))
  frame <- fmri_frame(
    assays = list(signal = source),
    observations = data.frame(.obs_id = sprintf("o-%04d", seq_len(1500))),
    features = data.frame(.feature_id = feature_ids),
    space = index_space(1000, ids = feature_ids)
  )
  cost <- source_realization_cost(source)

  baseline <- gc(reset = TRUE)["Vcells", "used"]
  value <- collect_assay(frame, memory_budget = cost$estimated_peak_bytes)
  peak <- (gc()["Vcells", "max used"] - baseline) * 8

  # R object headers, selectors, and profiler granularity are outside the
  # numerical-payload model. Allow 25% plus 2 MiB for those fixed costs.
  tolerance <- cost$estimated_peak_bytes * 0.25 + 2 * 1024^2
  expect_equal(dim(value), c(1500L, 1000L))
  expect_lte(peak, cost$estimated_peak_bytes + tolerance)
})
