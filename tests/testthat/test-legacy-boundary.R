test_that("the 1.0 namespace does not expose the legacy architecture", {
  removed <- c(
    "fmri_dataset", "fmri_dataset_legacy", "matrix_dataset",
    "fmri_mem_dataset", "fmri_study_dataset", "latent_dataset",
    "fmri_latent_dataset", "fmri_series", "new_fmri_series",
    "fmri_group", "as_fmri_group", "group_map", "group_reduce",
    "backend_open", "backend_close", "backend_get_data",
    "backend_get_dims", "backend_get_mask", "backend_get_metadata",
    "register_backend", "create_backend", "get_backend_registry",
    "matrix_backend", "nifti_backend", "h5_backend", "latent_backend",
    "study_backend", "bids_h5_dataset", "compress_bids_study",
    "data_chunks", "as.matrix_dataset", "as_delayed_array",
    "get_TR", "get_run_lengths", "n_runs", "n_timepoints"
  )
  exports <- getNamespaceExports("fmridataset")
  expect_length(intersect(removed, exports), 0L)
  expect_false(any(vapply(
    removed,
    exists,
    logical(1),
    envir = asNamespace("fmridataset"),
    inherits = FALSE
  )))
})

test_that("the canonical core has no fmrihrf hard dependency", {
  imports <- packageDescription("fmridataset", fields = "Imports")
  expect_false(grepl("(^|[,[:space:]])fmrihrf([,[:space:]]|$)", imports))
})

test_that("legacy serialization adapters remain internal", {
  exports <- getNamespaceExports("fmridataset")
  expect_true(all(c("as_fmri_frame", "upgrade_dataset") %in% exports))
  expect_true(exists(
    "as_fmri_frame.matrix_dataset",
    envir = asNamespace("fmridataset"),
    inherits = FALSE
  ))
  expect_false("as_fmri_frame.matrix_dataset" %in% exports)
})
