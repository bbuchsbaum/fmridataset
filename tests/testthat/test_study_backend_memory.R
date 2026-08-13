test_that("study_backend works with data_chunks", {
  # Create small study
  backends <- lapply(1:3, function(id) {
    matrix_backend(
      matrix(id, nrow = 50, ncol = 100),
      mask = rep(TRUE, 100)
    )
  })

  study_backend_obj <- study_backend(backends, paste0("sub-", 1:3))

  # Create proper sampling frame
  sf <- list(
    blocklens = rep(50, 3),
    TR = 2,
    nruns = 3
  )
  class(sf) <- "sampling_frame"

  # Add blockids method for sampling_frame if not available
  if (!exists("blockids.sampling_frame")) {
    blockids.sampling_frame <- function(x) {
      rep(seq_along(x$blocklens), times = x$blocklens)
    }
  }

  # Create dataset
  dataset <- structure(
    list(
      backend = study_backend_obj,
      sampling_frame = sf,
      nruns = 3
    ),
    class = c("fmri_file_dataset", "fmri_dataset")
  )

  # Get chunks - use runwise to get one chunk per subject/run
  chunks <- data_chunks(dataset, runwise = TRUE)

  # Should have 3 chunks (one per subject)
  chunk_list <- list()
  i <- 1
  tryCatch(
    {
      while (TRUE) {
        chunk_list[[i]] <- chunks$nextElem()
        i <- i + 1
      }
    },
    error = function(e) {
      if (!grepl("StopIteration", e$message)) stop(e)
    }
  )

  expect_equal(length(chunk_list), 3)

  # Each chunk should have correct data
  expect_equal(unique(as.vector(chunk_list[[1]]$data[, 1])), 1)
  expect_equal(unique(as.vector(chunk_list[[2]]$data[, 1])), 2)
  expect_equal(unique(as.vector(chunk_list[[3]]$data[, 1])), 3)
})
