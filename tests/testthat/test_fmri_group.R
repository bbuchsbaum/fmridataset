test_that("fmri_group constructs valid objects", {
  subjects <- make_dummy_subjects()
  gd <- fmri_group(subjects, id = "sub", dataset_col = "dataset", space = "MNI152", mask_strategy = "union")

  expect_s3_class(gd, "fmri_group")
  expect_identical(subjects(gd), subjects)
  expect_identical(gd$id, "sub")
  expect_identical(gd$dataset_col, "dataset")
  expect_identical(gd$space, "MNI152")
  expect_identical(gd$mask_strategy, "union")
})

test_that("fmri_group validates dataset column", {
  bad_subjects <- data.frame(sub = "sub-01", dataset = "not-a-list", stringsAsFactors = FALSE)
  expect_error(fmri_group(bad_subjects, id = "sub", dataset_col = "dataset"), "must be a list-column")

  subjects <- make_dummy_subjects()
  subjects$dataset[[1]] <- list(structure(list(), class = "fmri_dataset"), structure(list(), class = "fmri_dataset"))
  expect_error(fmri_group(subjects, id = "sub", dataset_col = "dataset"), "length-1 entry")
})

test_that("subjects<- replacement re-validates", {
  gd <- fmri_group(make_dummy_subjects(), id = "sub", dataset_col = "dataset")
  new_subjects <- make_dummy_subjects(2L)
  subjects(gd) <- new_subjects
  expect_identical(subjects(gd), new_subjects)

  bad_subjects <- data.frame(sub = "sub-01", stringsAsFactors = FALSE)
  expect_error({
    subjects(gd) <- bad_subjects
  }, "Replacement `subjects`")
})

test_that("print.fmri_group emits a compact summary", {
  gd <- fmri_group(make_dummy_subjects(), id = "sub", dataset_col = "dataset", mask_strategy = "subject_specific")
  expect_output(print(gd), "<fmri_group>")
  expect_output(print(gd), "subjects       : 3")
})

test_that("n_subjects returns the correct count", {
  gd <- fmri_group(make_dummy_subjects(5L), id = "sub", dataset_col = "dataset")
  expect_identical(n_subjects(gd), 5L)
})

test_that("as_fmri_group wraps a data frame", {
  subjects <- make_dummy_subjects(2L)
  gd <- as_fmri_group(subjects, id = "sub", dataset_col = "dataset")
  expect_s3_class(gd, "fmri_group")
  expect_identical(subjects(gd), subjects)
})
