# Snapshot tests for print methods and summaries

test_that("fmri_dataset print snapshots", {
  ref_data <- generate_reference_data()

  # Basic dataset
  dset_basic <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  expect_snapshot({
    print(dset_basic)
  })

  # Multi-run dataset
  dset_multi <- matrix_dataset(
    ref_data$multirun_data,
    TR = ref_data$metadata$TR
  )

  expect_snapshot({
    print(dset_multi)
  })

  # Masked dataset
  dset_masked <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data),
    mask = ref_data$mask
  )

  expect_snapshot({
    print(dset_masked)
  })
})

test_that("FmriSeries show method snapshots", {
  ref_data <- generate_reference_data()

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  fmri_series <- as_delayed_array(dset)

  expect_snapshot({
    show(fmri_series)
  })

  # Subset version
  subset_series <- fmri_series[1:10, 1:20]

  expect_snapshot({
    show(subset_series)
  })
})

test_that("backend print snapshots", {
  ref_data <- generate_reference_data()

  # Matrix backend
  backend_mat <- matrix_backend(
    data = ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  expect_snapshot({
    print(backend_mat)
  })

  # Multi-run backend
  backend_multi <- matrix_backend(
    data = ref_data$multirun_data,
    TR = ref_data$metadata$TR
  )

  expect_snapshot({
    print(backend_multi)
  })
})

test_that("sampling_frame print snapshots", {
  # Various configurations
  sframe_single <- fmrihrf::sampling_frame(TR = 2, blocklens = 100)

  expect_snapshot({
    print(sframe_single)
  })

  sframe_multi <- fmrihrf::sampling_frame(TR = 2.5, blocklens = c(100, 150, 120))

  expect_snapshot({
    print(sframe_multi)
  })

  # With event table
  events <- data.frame(
    onset = seq(0, 90, by = 10),
    duration = rep(5, 10),
    condition = rep(c("A", "B"), 5)
  )

  sframe_events <- fmrihrf::sampling_frame(
    TR = 2,
    blocklens = 100,
    event_table = events
  )

  expect_snapshot({
    print(sframe_events)
  })
})

test_that("error message snapshots", {
  # Invalid dataset creation
  expect_snapshot_error({
    matrix_dataset(data = "not a matrix", TR = 2)
  })

  expect_snapshot_error({
    matrix_dataset(data = matrix(1:10, 5, 2), TR = -1)
  })

  # Invalid sampling frame
  expect_snapshot_error({
    fmrihrf::sampling_frame(TR = 0, blocklens = 100)
  })

  expect_snapshot_error({
    fmrihrf::sampling_frame(TR = 2, blocklens = numeric(0))
  })
})

test_that("summary output snapshots", {
  ref_data <- generate_reference_data()

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  # If summary method exists
  if (exists("summary.fmri_dataset")) {
    expect_snapshot({
      summary(dset)
    })
  }
})

test_that("data chunk iterator snapshots", {
  ref_data <- generate_reference_data()

  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ncol(ref_data$matrix_data)
  )

  # Capture chunk information
  chunk_info <- list()
  chunks <- data_chunks(dset, chunk_spec = 25)

  i <- 1
  for (chunk in chunks) {
    chunk_info[[i]] <- list(
      chunk_number = i,
      dimensions = dim(chunk),
      first_value = chunk[1, 1],
      last_value = chunk[nrow(chunk), ncol(chunk)]
    )
    i <- i + 1
    if (i > 3) break # Just first 3 chunks for snapshot
  }

  expect_snapshot({
    print(chunk_info)
  })
})
