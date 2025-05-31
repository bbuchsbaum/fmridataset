test_that("workflow creation and application works", {
  wf <- create_workflow("elegant_preproc")
  wf <- describe(wf, "test workflow")
  wf <- add_step(wf, function(ds) { ds$metadata$test_step <- TRUE; ds })
  wf <- finish_with_flourish(wf)

  dataset <- fmri_dataset_create(
    images = matrix(rnorm(200), nrow = 20, ncol = 10),
    TR = 2.0,
    run_lengths = 20
  )

  result <- apply_workflow(dataset, wf)
  expect_true(isTRUE(get_metadata(result, "test_step")))
})
