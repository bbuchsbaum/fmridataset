# Snapshot tests for print methods and summaries

test_that("fmri_dataset print snapshots", {
  testthat::local_edition(3)

  ref_data <- generate_reference_data()
  
  # Basic dataset
  dset_basic <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths
  )
  
  expect_snapshot({
    print(dset_basic)
  })
  
  # Multi-run dataset
  dset_multi <- matrix_dataset(
    do.call(rbind, ref_data$multirun_data),
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths
  )
  
  expect_snapshot({
    print(dset_multi)
  })
  
  # Masked dataset
  dset_masked <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths,
  )

  # Inject custom mask for snapshot coverage
  dset_masked$mask <- as.numeric(ref_data$mask)
  
  expect_snapshot({
    print(dset_masked)
  })
})

test_that("FmriSeries show method snapshots", {
  testthat::local_edition(3)

  ref_data <- generate_reference_data()
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths
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
  testthat::local_edition(3)

  ref_data <- generate_reference_data()
  
  # Matrix backend
  backend_mat <- matrix_backend(
    data_matrix = ref_data$matrix_data,
    metadata = list(
      TR = ref_data$metadata$TR,
      run_length = ref_data$metadata$run_lengths
    )
  )
  
  expect_snapshot({
    print(backend_mat)
  })
  
  # Multi-run backend
  backend_multi <- matrix_backend(
    data_matrix = do.call(rbind, ref_data$multirun_data),
    metadata = list(
      TR = ref_data$metadata$TR,
      run_length = ref_data$metadata$run_lengths
    )
  )
  
  expect_snapshot({
    print(backend_multi)
  })
})

test_that("sampling_frame print snapshots", {
  testthat::local_edition(3)

  # Various configurations
  sframe_single <- fmrihrf::sampling_frame(TR = 2, blocklens = 100)
  
  expect_snapshot({
    print(sframe_single)
  })
  
  sframe_multi <- fmrihrf::sampling_frame(TR = 2.5, blocklens = c(100, 150, 120))
  
  expect_snapshot({
    print(sframe_multi)
  })
  
  # Event schedule alongside sampling frame
  events <- data.frame(
    onset = seq(0, 90, by = 10),
    duration = rep(5, 10),
    condition = rep(c("A", "B"), 5)
  )

  expect_snapshot({
    list(
      frame = sframe_single,
      events = events
    )
  })
})

test_that("error message snapshots", {
  testthat::local_edition(3)

  # Invalid dataset creation
  expect_snapshot_error({
    matrix_dataset(
      datamat = matrix(1:10, nrow = 5, ncol = 2),
      TR = 2,
      run_length = 10
    )
  })
  
  expect_snapshot_error({
    matrix_dataset(
      datamat = matrix(1:10, nrow = 5, ncol = 2),
      TR = -1,
      run_length = 5
    )
  })
  
  # Invalid sampling frame
  expect_snapshot_error({
    fmrihrf::sampling_frame(TR = 0, blocklens = 100)
  })
  
  expect_snapshot_error({
    fmrihrf::sampling_frame(TR = 2, blocklens = c(100, -50))
  })

  expect_snapshot_error({
    fmrihrf::sampling_frame(TR = c(2, 2), blocklens = c(50, 50), precision = 2)
  })
})

test_that("summary output snapshots", {
  testthat::local_edition(3)

  ref_data <- generate_reference_data()
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths
  )
  
  # If summary method exists
  if (exists("summary.fmri_dataset")) {
    expect_snapshot({
      summary(dset)
    })
  }
})

test_that("data chunk iterator snapshots", {
  testthat::local_edition(3)

  ref_data <- generate_reference_data()
  
  dset <- matrix_dataset(
    ref_data$matrix_data,
    TR = ref_data$metadata$TR,
    run_length = ref_data$metadata$run_lengths
  )
  
  # Capture chunk information
  chunk_info <- list()
  chunks <- collect_chunks(data_chunks(dset, nchunks = 4))
  
  i <- 1
  for (chunk in chunks) {
    chunk_info[[i]] <- list(
      chunk_number = i,
      dimensions = dim(chunk$data),
      first_value = chunk$data[1, 1],
      last_value = chunk$data[nrow(chunk$data), ncol(chunk$data)]
    )
    i <- i + 1
    if (i > 3) break  # Just first 3 chunks for snapshot
  }
  
  expect_snapshot({
    print(chunk_info)
  })
})
