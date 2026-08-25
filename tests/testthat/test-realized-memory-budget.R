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
  realized_bytes <- n_values * 8 # R doubles in memory

  # A budget between the two must now be refused. Before the fix this was
  # allowed, and the returned object exceeded it.
  midpoint <- (storage_bytes + realized_bytes) / 2
  expect_error(
    collect_assay(frame, memory_budget = midpoint),
    class = "fmridataset_error_budget"
  )

  # A budget at the realized size is honoured, and the result fits inside it.
  collected <- collect_assay(frame, memory_budget = realized_bytes)
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
  expect_match(conditionMessage(err), "realized as R doubles")
  expect_match(conditionMessage(err), "force = TRUE")
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
  expect_lte(plan$max_block_bytes, budget)
  expect_equal(plan$total_bytes, prod(as.double(plan$shape)) * realized)
})
