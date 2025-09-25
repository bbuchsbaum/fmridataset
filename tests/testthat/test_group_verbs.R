test_that("filter_subjects reduces the group", {
  gd <- make_dummy_group(space = "MNI152")
  filtered <- filter_subjects(gd, age > 31)

  expect_s3_class(filtered, "fmri_group")
  expect_identical(filtered$space, "MNI152")
  expect_identical(n_subjects(filtered), 2L)
  expect_true(all(subjects(filtered)$age > 31))
})

test_that("mutate_subjects adds new columns", {
  gd <- make_dummy_group()
  mutated <- mutate_subjects(gd, cohort = ifelse(age > 31, "older", "younger"))

  expect_true("cohort" %in% names(subjects(mutated)))
  expect_identical(subjects(mutated)$cohort, c("younger", "older", "older"))
})

test_that("left_join_subjects merges metadata", {
  gd <- make_dummy_group()
  meta <- data.frame(
    sub = c("sub-01", "sub-02", "sub-03"),
    scanner = c("A", "B", "C"),
    stringsAsFactors = FALSE
  )

  joined <- left_join_subjects(gd, meta, by = "sub")
  expect_identical(subjects(joined)$scanner, meta$scanner)
})

test_that("left_join_subjects falls back without dplyr", {
  gd <- make_dummy_group()
  meta <- data.frame(
    sub = c("sub-01", "sub-02", "sub-03"),
    site = c("01", "01", "02"),
    stringsAsFactors = FALSE
  )

  joined <- testthat::with_mocked_bindings(
    left_join_subjects(gd, meta, by = "sub"),
    requireNamespace = function(pkg, quietly = TRUE) FALSE,
    .package = "base"
  )
  expect_identical(subjects(joined)$site, meta$site)
})

test_that("sample_subjects performs unstratified sampling", {
  gd <- make_dummy_group()
  set.seed(123)
  sampled <- sample_subjects(gd, n = 2L)
  expect_identical(n_subjects(sampled), 2L)
  expect_true(all(sampled$subjects$sub %in% gd$subjects$sub))
})

test_that("sample_subjects supports stratified sampling", {
  gd <- make_dummy_group()
  gd <- mutate_subjects(gd, cohort = c("A", "A", "B"))
  set.seed(123)
  sampled <- sample_subjects(gd, n = 1L, strata = "cohort")

  expect_identical(n_subjects(sampled), 2L)
  expect_setequal(sampled$subjects$cohort, c("A", "B"))
})

test_that("stream_subjects warns on unsupported prefetch", {
  gd <- make_dummy_group()
  expect_warning(stream_subjects(gd, prefetch = 2L), "Prefetch values")

  stream <- stream_subjects(gd, prefetch = 1L)
  first <- stream[["next"]]()
  expect_identical(first$sub, "sub-01")
})

test_that("group_reduce aggregates over subjects", {
  gd <- make_dummy_group()
  total_age <- group_reduce(
    gd,
    .map = function(row) row$age,
    .reduce = function(acc, value) acc + value,
    .init = 0L
  )
  expect_identical(total_age, sum(subjects(gd)$age))
})

test_that("group_reduce handles errors", {
  gd <- make_dummy_group()
  expect_warning(
    group_reduce(
      gd,
      .map = function(row) {
        if (row$sub == "sub-02") stop("boom")
        row$age
      },
      .reduce = `+`,
      .init = 0L,
      on_error = "warn"
    ),
    "group_reduce(): boom",
    fixed = TRUE
  )

  skip_res <- group_reduce(
    gd,
    .map = function(row) {
      if (row$sub == "sub-02") stop("boom")
      row$age
    },
    .reduce = `+`,
    .init = 0L,
    on_error = "skip"
  )
  expect_identical(skip_res, sum(subjects(gd)$age[c(1, 3)]))
})
