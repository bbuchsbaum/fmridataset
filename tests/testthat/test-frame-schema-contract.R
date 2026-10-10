.order_schema_frame <- function(observations, assay_order = c("beta", "var")) {
  space <- index_space(3, id_policy = "deterministic", namespace = "demo")
  values <- list(beta = matrix(1:6 + 0, 2, 3), var = matrix(1:6 + 0, 2, 3))
  fmri_frame(values[assay_order], observations, space = space, active_assay = "beta")
}

test_that("schema comparison ignores column and assay registry order like the digest", {
  observations <- data.frame(.obs_id = c("o1", "o2"), cond = c("a", "b"), rt = c(1, 2))
  reference <- .order_schema_frame(observations)
  reordered_columns <- .order_schema_frame(observations[, c(".obs_id", "rt", "cond")])
  reordered_assays <- .order_schema_frame(observations, c("var", "beta"))

  for (candidate in list(reordered_columns, reordered_assays)) {
    expect_identical(frame_schema_digest(candidate), frame_schema_digest(reference))
    expect_true(compare_frame_schema(candidate, reference)$compatible)
    expect_invisible(validate_against_schema(candidate, reference))
  }

  changed <- observations
  changed$rt <- c(1L, 2L)
  report <- compare_frame_schema(.order_schema_frame(changed), reference)
  expect_false(report$compatible)
  expect_identical(report$path, "schema.observation.columns.rt.class")

  renamed <- observations
  names(renamed)[[3L]] <- "reaction_time"
  report <- compare_frame_schema(.order_schema_frame(renamed), reference)
  expect_false(report$compatible)
  expect_identical(report$path, "schema.observation.columns")
})

test_that("frame schema validation checks every section's shape", {
  schema <- frame_schema(make_frame_fixture()$frame)
  reject <- function(x, pattern = NULL) {
    expect_no_warning(expect_error(
      validate_frame_schema(x), pattern,
      class = "fmridataset_error_schema"
    ))
  }
  edit <- function(...) {
    value <- unclass(schema)
    updates <- list(...)
    for (name in names(updates)) value[name] <- list(updates[[name]])
    class(value) <- class(schema)
    value
  }
  reject(edit(observation = "garbage"), "observation")
  reject(edit(feature = 1), "feature")
  reject(edit(entities = "x"), "entities")
  reject(edit(relations = 1), "relations")
  reject(edit(tables = "x"), "tables")
  reject(edit(assays = list(beta = "float64")), "assay")

  observation <- schema$observation
  observation$count <- 1e10
  reject(edit(observation = observation), "observation")
  observation <- schema$observation
  observation$columns <- list(cond = "character")
  reject(edit(observation = observation), "column")
  feature <- schema$feature
  feature$space$feature_ids <- feature$space$feature_ids[-1L]
  reject(edit(feature = feature), "feature space")

  for (name in list(NULL, c("beta", "variance"), NA_character_, 1)) {
    reject(edit(active_assay = list(policy = "named", name = name)), "active assay")
  }

  extra <- unclass(schema)
  extra$surprise <- TRUE
  class(extra) <- class(schema)
  reject(extra, "unknown: surprise")
  reordered <- unclass(schema)[rev(names(schema))]
  class(reordered) <- class(schema)
  expect_invisible(validate_frame_schema(reordered))
})

test_that("frame schemas record POSIXct time zones and difftime units", {
  space <- index_space(3, id_policy = "deterministic", namespace = "demo")
  make <- function(time, duration) {
    fmri_frame(
      list(beta = matrix(1:6 + 0, 2, 3)),
      data.frame(.obs_id = c("o1", "o2"), time = time, duration = duration),
      space = space
    )
  }
  instants <- c("2020-01-01 10:00:00", "2021-06-01 00:00:00")
  reference <- make(
    as.POSIXct(instants, tz = "America/New_York"),
    as.difftime(c(1, 2), units = "secs")
  )
  columns <- frame_schema(reference)$observation$columns
  expect_identical(columns$time$tzone, "America/New_York")
  expect_identical(columns$duration$units, "secs")
  expect_false(any(c("tzone", "units") %in% names(columns$.obs_id)))

  other_zone <- make(
    as.POSIXct(instants, tz = "UTC"),
    as.difftime(c(1, 2), units = "secs")
  )
  other_units <- make(
    as.POSIXct(instants, tz = "America/New_York"),
    as.difftime(c(1, 2), units = "mins")
  )
  for (candidate in list(other_zone, other_units)) {
    expect_false(compare_frame_schema(candidate, reference)$compatible)
    expect_false(identical(frame_schema_digest(candidate), frame_schema_digest(reference)))
  }
})
