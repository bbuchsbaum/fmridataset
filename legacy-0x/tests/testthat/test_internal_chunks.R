library(fmridataset)

# Tests for exec_strategy and collect_chunks

test_that("exec_strategy and collect_chunks work", {
  set.seed(1)
  Y <- matrix(rnorm(50 * 10), 50, 10)
  dset <- matrix_dataset(Y, TR = 1, run_length = 50)

  strat <- fmridataset:::exec_strategy("chunkwise", nchunks = 3)
  iter <- strat(dset)
  chunks <- fmridataset:::collect_chunks(iter)

  expect_equal(length(chunks), 3)
  expect_true(all(sapply(chunks, inherits, "data_chunk")))

  voxel_inds <- sort(unlist(lapply(chunks, function(ch) ch$voxel_ind)))
  expect_equal(voxel_inds, 1:10)
})
