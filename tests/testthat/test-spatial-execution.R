test_that("execution dispatch distinguishes matrix, native, and reconstruction paths", {
  x <- make_frame_fixture()$frame

  expect_identical(execution_path(x, operation = "matrix"), "matrix")
  expect_identical(execution_path(x, operation = "spatial"), "reconstruct")
  expect_error(
    execution_path(x, operation = "spatial", path = "native"),
    class = "fmridataset_error_backend_io"
  )

  maps <- collect_spatial_maps(x, observations = c(3L, 1L))
  expect_identical(names(maps), observation_ids(x)[c(3L, 1L)])
  expect_equal(
    unname(t(vapply(maps, function(map) vectorize_space(space(x), map), numeric(ncol(x))))),
    unname(collect_assay(x[c(3L, 1L), ]))
  )
})

test_that("NIfTI spatial execution uses the native fast path in observation order", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")
  source <- nifti_array_source(c(path, path), path)
  counted <- counting_source(source)
  spatial <- nifti_source_space(source, template = "fixture")
  x <- fmri_frame(
    assays = list(signal = counted),
    observations = data.frame(.obs_id = sprintf("volume-%02d", seq_len(source_shape(source)[[1L]]))),
    space = spatial,
    active_assay = "signal"
  )
  selected <- c(8L, 1L, 5L)

  expect_identical(execution_path(x, operation = "spatial"), "native")
  maps <- collect_spatial_maps(x, selected)
  expect_length(maps, length(selected))
  expect_true(all(vapply(maps, methods::is, logical(1), "NeuroVol")))
  expect_identical(names(maps), observation_ids(x)[selected])
  expect_equal(
    unname(t(vapply(maps, function(map) vectorize_space(spatial, map), numeric(ncol(x))))),
    unname(source_read(source, selected, seq_len(ncol(x)))),
    tolerance = 0
  )
  reference_map <- reconstruct_space(
    spatial,
    as.numeric(source_read(source, selected[[1L]], seq_len(ncol(x))))
  )
  expect_equal(as.numeric(maps[[1L]]), as.numeric(reference_map), tolerance = 0)
  expect_identical(source_counts(counted)$reads, 3)

  reset_source_counts(counted)
  seen <- execute_spatial(x, selected, function(map, observation_id) {
    c(id = observation_id, class = class(map)[[1L]])
  })
  expect_identical(vapply(seen, `[[`, character(1), "id"), observation_ids(x)[selected])
  expect_identical(source_counts(counted)$reads, 3)
})

test_that("feature-restricted views reconstruct instead of widening native reads", {
  path <- system.file("extdata", "global_mask_v4.nii", package = "neuroim2")
  skip_if(!file.exists(path), "neuroim2 NIfTI fixture is unavailable")
  source <- nifti_array_source(path, path)
  counted <- counting_source(source)
  x <- fmri_frame(
    assays = list(signal = counted),
    observations = data.frame(.obs_id = sprintf("volume-%02d", seq_len(source_shape(source)[[1L]]))),
    space = nifti_source_space(source),
    active_assay = "signal"
  )
  restricted <- x[, c(3L, 1L)]

  expect_identical(execution_path(restricted, operation = "spatial"), "reconstruct")
  map <- collect_spatial_maps(restricted, 2L)[[1L]]
  expect_equal(
    as.numeric(vectorize_space(space(restricted), map)),
    as.numeric(collect_assay(restricted[2L, ]))
  )
  expect_error(
    collect_spatial_maps(restricted, 2L, path = "native"),
    class = "fmridataset_error_backend_io"
  )
})

test_that("spatial collection and streaming enforce different memory totals", {
  x <- make_frame_fixture()$frame
  collection_cost <- fmridataset:::.spatial_realization_cost(
    x, 2L, active_assay(x), "reconstruct"
  )
  streaming_cost <- fmridataset:::.spatial_realization_cost(
    x, 1L, active_assay(x), "reconstruct"
  )

  expect_gt(collection_cost$estimated_peak_bytes, collection_cost$estimated_output_bytes)
  expect_gt(collection_cost$estimated_peak_bytes, streaming_cost$estimated_peak_bytes)

  expect_error(
    collect_spatial_maps(
      x, 1:2,
      memory_budget = collection_cost$estimated_peak_bytes - 1
    ),
    class = "fmridataset_error_budget"
  )
  streamed <- execute_spatial(
    x,
    1:2,
    function(map, observation_id) observation_id,
    memory_budget = streaming_cost$estimated_peak_bytes
  )
  expect_identical(unlist(streamed, use.names = FALSE), observation_ids(x)[1:2])
})

test_that("composite spatial budgets sum native part realizations", {
  spatial <- make_composite_space_fixture()
  x <- fmri_frame(
    assays = list(signal = matrix(seq_len(8L), nrow = 1L)),
    observations = data.frame(.obs_id = "map-1"),
    space = spatial
  )

  cost <- fmridataset:::.spatial_realization_cost(
    x, 1L, active_assay(x), "reconstruct"
  )
  expect_equal(cost$estimated_output_bytes, 64)
  expect_gt(cost$estimated_peak_bytes, cost$estimated_output_bytes)
  expect_error(collect_spatial_maps(
    x,
    memory_budget = cost$estimated_peak_bytes - 1
  ), class = "fmridataset_error_budget")
  map <- collect_spatial_maps(x, memory_budget = cost$estimated_peak_bytes)[[1L]]
  expect_s3_class(map, "composite_map")
  expect_identical(names(map$parts), composite_part_names(spatial))
})
