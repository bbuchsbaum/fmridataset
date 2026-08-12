# Tests for R/group_map.R, R/group_verbs.R, R/group_iter.R - coverage improvement

# Helper to create a test fmri_group
# Uses simple length-1 list objects as dummy datasets since validate_fmri_group
# requires length(dataset_entry) == 1
make_test_group <- function(n = 3) {
  datasets <- lapply(seq_len(n), function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })

  subjects_df <- data.frame(
    subject_id = paste0("sub-", sprintf("%03d", seq_len(n))),
    age = c(25, 30, 35)[seq_len(n)],
    group = c("A", "B", "A")[seq_len(n)],
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )

  fmri_group(subjects_df, id = "subject_id", dataset_col = "dataset")
}

# --- iter_subjects ---

test_that("iter_subjects iterates through all subjects", {
  gd <- make_test_group(3)
  iter <- iter_subjects(gd)

  results <- list()
  i <- 0
  repeat {
    row <- iter[["next"]]()
    if (is.null(row)) break
    i <- i + 1
    results[[i]] <- row
  }

  expect_equal(i, 3)
  expect_equal(results[[1]]$subject_id, "sub-001")
  expect_equal(results[[2]]$subject_id, "sub-002")
  expect_equal(results[[3]]$subject_id, "sub-003")
})

test_that("iter_subjects respects order_by", {
  gd <- make_test_group(3)
  iter <- iter_subjects(gd, order_by = "age")

  r1 <- iter[["next"]]()
  r2 <- iter[["next"]]()
  r3 <- iter[["next"]]()

  expect_equal(r1$age, 25)
  expect_equal(r2$age, 30)
  expect_equal(r3$age, 35)
})

test_that("iter_subjects errors on invalid order_by column", {
  gd <- make_test_group(3)
  expect_error(iter_subjects(gd, order_by = "nonexistent"), "order_by")
})

# --- group_map ---

test_that("group_map applies function to all subjects", {
  gd <- make_test_group(3)
  results <- group_map(gd, function(row) row$subject_id)

  expect_type(results, "list")
  expect_length(results, 3)
  expect_equal(results[["sub-001"]], "sub-001")
})

test_that("group_map with bind_rows output", {
  gd <- make_test_group(3)
  results <- group_map(gd, function(row) {
    data.frame(id = row$subject_id, age = row$age)
  }, out = "bind_rows")

  expect_true(is.data.frame(results))
  expect_equal(nrow(results), 3)
})

test_that("group_map with order_by", {
  gd <- make_test_group(3)
  ages <- c()
  group_map(gd, function(row) {
    ages <<- c(ages, row$age)
    NULL
  }, order_by = "age")

  expect_equal(ages, c(25, 30, 35))
})

test_that("group_map with on_error='warn' warns but continues", {
  gd <- make_test_group(3)
  expect_warning(
    results <- group_map(gd, function(row) {
      if (row$age == 30) stop("test error")
      row$subject_id
    }, on_error = "warn"),
    "test error"
  )
  # Should have 2 results (skipped the erroring one)
  expect_length(results, 2)
})

test_that("group_map with on_error='skip' skips silently", {
  gd <- make_test_group(3)
  results <- group_map(gd, function(row) {
    if (row$age == 30) stop("test error")
    row$subject_id
  }, on_error = "skip")

  expect_length(results, 2)
})

test_that("group_map with on_error='stop' stops on error", {
  gd <- make_test_group(3)
  expect_error(
    group_map(gd, function(row) {
      if (row$age == 25) stop("test error")
    }, on_error = "stop"),
    "test error"
  )
})

test_that("group_map with NULL returns are skipped", {
  gd <- make_test_group(3)
  results <- group_map(gd, function(row) {
    if (row$age == 30) return(NULL)
    row$subject_id
  })
  expect_length(results, 2)
})

test_that("group_map bind_rows with empty results", {
  gd <- make_test_group(3)
  results <- group_map(gd, function(row) NULL, out = "bind_rows")
  expect_true(is.data.frame(results))
  expect_equal(nrow(results), 0)
})

# --- filter_subjects ---

test_that("filter_subjects filters by predicate", {
  gd <- make_test_group(3)
  filtered <- filter_subjects(gd, age > 25)

  expect_equal(nrow(subjects(filtered)), 2)
  expect_true(all(subjects(filtered)$age > 25))
})

test_that("filter_subjects with no expressions returns unchanged", {
  gd <- make_test_group(3)
  result <- filter_subjects(gd)
  expect_equal(nrow(subjects(result)), 3)
})

test_that("filter_subjects with multiple predicates (AND)", {
  gd <- make_test_group(3)
  filtered <- filter_subjects(gd, age > 25, group == "A")

  expect_equal(nrow(subjects(filtered)), 1)
  expect_equal(subjects(filtered)$subject_id, "sub-003")
})

test_that("filter_subjects errors on non-logical expression", {
  gd <- make_test_group(3)
  expect_error(filter_subjects(gd, age), "logical")
})

# --- mutate_subjects ---

test_that("mutate_subjects adds new column", {
  gd <- make_test_group(3)
  result <- mutate_subjects(gd, age_group = ifelse(age > 27, "old", "young"))

  expect_true("age_group" %in% names(subjects(result)))
  expect_equal(subjects(result)$age_group, c("young", "old", "old"))
})

test_that("mutate_subjects with no expressions returns unchanged", {
  gd <- make_test_group(3)
  result <- mutate_subjects(gd)
  expect_equal(subjects(result), subjects(gd))
})

test_that("mutate_subjects with scalar value", {
  gd <- make_test_group(3)
  result <- mutate_subjects(gd, site = "site_A")
  expect_equal(subjects(result)$site, rep("site_A", 3))
})

test_that("mutate_subjects errors on unnamed expression", {
  gd <- make_test_group(3)
  expect_error(mutate_subjects(gd, age + 1), "named")
})

test_that("mutate_subjects errors on wrong length", {
  gd <- make_test_group(3)
  expect_error(mutate_subjects(gd, bad = c(1, 2)), "length")
})

# --- left_join_subjects ---

test_that("left_join_subjects joins metadata", {
  gd <- make_test_group(3)
  meta <- data.frame(
    subject_id = c("sub-001", "sub-002", "sub-003"),
    score = c(90, 85, 95),
    stringsAsFactors = FALSE
  )

  result <- left_join_subjects(gd, meta)
  expect_true("score" %in% names(subjects(result)))
  expect_equal(subjects(result)$score, c(90, 85, 95))
})

test_that("left_join_subjects with explicit by", {
  gd <- make_test_group(3)
  meta <- data.frame(
    subject_id = c("sub-001", "sub-002", "sub-003"),
    score = c(90, 85, 95),
    stringsAsFactors = FALSE
  )

  result <- left_join_subjects(gd, meta, by = "subject_id")
  expect_true("score" %in% names(subjects(result)))
})

# --- sample_subjects ---

test_that("sample_subjects samples correct number", {
  gd <- make_test_group(3)
  set.seed(42)
  result <- sample_subjects(gd, n = 2)
  expect_equal(nrow(subjects(result)), 2)
})

test_that("sample_subjects with replacement", {
  gd <- make_test_group(3)
  set.seed(42)
  result <- sample_subjects(gd, n = 5, replace = TRUE)
  expect_equal(nrow(subjects(result)), 5)
})

test_that("sample_subjects errors when n > available without replacement", {
  gd <- make_test_group(3)
  expect_error(sample_subjects(gd, n = 5), "Cannot sample more")
})

test_that("sample_subjects with strata", {
  gd <- make_test_group(3)
  set.seed(42)
  result <- sample_subjects(gd, n = 1, strata = "group")
  # Should sample 1 from each stratum
  expect_equal(nrow(subjects(result)), 2) # 2 strata, 1 from each
})

test_that("sample_subjects with empty group returns unchanged", {
  datasets <- list()
  subjects_df <- data.frame(
    subject_id = character(0),
    stringsAsFactors = FALSE
  )
  subjects_df$dataset <- list()
  gd <- structure(
    list(
      subjects = subjects_df,
      id = "subject_id",
      dataset_col = "dataset",
      space = NULL,
      mask_strategy = "subject_specific"
    ),
    class = c("fmri_group", "fmridataset_group")
  )
  result <- sample_subjects(gd, n = 0)
  expect_equal(nrow(subjects(result)), 0)
})

test_that("sample_subjects errors with wrong n length (unstratified)", {
  gd <- make_test_group(3)
  expect_error(sample_subjects(gd, n = c(1, 2)), "single integer")
})

# --- print.fmri_group ---

test_that("print.fmri_group produces output", {
  gd <- make_test_group(3)
  out <- capture.output(print(gd))
  expect_true(any(grepl("fmri_group", out)))
  expect_true(any(grepl("subjects", out)))
  expect_true(any(grepl("subject_id", out)))
})

test_that("print.fmri_group shows space when set", {
  datasets <- lapply(1:2, function(i) {
    structure(list(id = i), class = "fmri_dataset")
  })

  subjects_df <- data.frame(
    subject_id = c("s1", "s2"),
    dataset = I(datasets),
    stringsAsFactors = FALSE
  )

  gd <- fmri_group(subjects_df, id = "subject_id", space = "MNI152")
  out <- capture.output(print(gd))
  expect_true(any(grepl("MNI152", out)))
})

# --- validate_fmri_group ---
# Note: validate_fmri_group's NULL warning path is unreachable because
# the length check fires first (length(NULL) == 0 != 1). Not testing it.
