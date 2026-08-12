test_that("iter_subjects yields one row per invocation", {
  gd <- make_dummy_group(2L)
  it <- iter_subjects(gd)

  first <- it[["next"]]()
  expect_s3_class(first$dataset, "fmri_dataset")
  expect_identical(first$sub, "sub-01")

  second <- it[["next"]]()
  expect_identical(second$sub, "sub-02")
  expect_null(it[["next"]]())
})

test_that("iter_subjects respects order_by", {
  subjects <- make_dummy_subjects(3L)
  subjects <- subjects[c(3, 1, 2), ]
  gd <- fmri_group(subjects, id = "sub", dataset_col = "dataset")

  it <- iter_subjects(gd, order_by = "age")
  out <- vapply(rep(list(NULL), 3L), function(...) it[["next"]]()$sub, character(1))
  expect_identical(out, c("sub-01", "sub-02", "sub-03"))
})

test_that("iter_subjects handles empty groups", {
  subjects <- make_dummy_subjects(0L)
  gd <- fmri_group(subjects, id = "sub", dataset_col = "dataset")

  it <- iter_subjects(gd)
  expect_null(it[["next"]]())
})

test_that("group_map returns a named list by default", {
  gd <- make_dummy_group(3L)
  res <- group_map(gd, function(row) row$age)

  expect_equal(names(res), c("sub-01", "sub-02", "sub-03"))
  expect_identical(res[["sub-02"]], 32L)
})

test_that("group_map can bind rows", {
  gd <- make_dummy_group(3L)
  res <- group_map(
    gd,
    function(row) data.frame(sub = row$sub, age = row$age, stringsAsFactors = FALSE),
    out = "bind_rows"
  )

  expect_s3_class(res, "data.frame")
  expect_identical(res$sub, c("sub-01", "sub-02", "sub-03"))
})

test_that("group_map supports error policies", {
  gd <- make_dummy_group(3L)

  skip_res <- group_map(
    gd,
    function(row) {
      if (row$sub == "sub-02") stop("boom")
      data.frame(sub = row$sub)
    },
    on_error = "skip",
    out = "bind_rows"
  )
  expect_identical(skip_res$sub, c("sub-01", "sub-03"))

  warn_res <- expect_warning(
    group_map(
      gd,
      function(row) {
        if (row$sub == "sub-02") stop("boom")
        data.frame(sub = row$sub)
      },
      on_error = "warn"
    ),
    "group_map(): boom",
    fixed = TRUE
  )
  expect_length(warn_res, 2L)

  expect_error(
    group_map(
      gd,
      function(row) {
        if (row$sub == "sub-02") stop("boom")
        data.frame(sub = row$sub)
      },
      on_error = "stop"
    ),
    "boom"
  )
})
