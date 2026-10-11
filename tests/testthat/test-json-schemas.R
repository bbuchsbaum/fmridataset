.installed_json_schemas <- function() {
  dir <- system.file("schema", package = "fmridataset")
  paths <- list.files(dir, pattern = "[.]schema[.]json$", full.names = TRUE)
  stats::setNames(
    lapply(paths, jsonlite::read_json, simplifyVector = FALSE),
    basename(paths)
  )
}

# Collect every JSON object (named list) in a parsed schema with its path.
.json_schema_objects <- function(x, path = "#") {
  out <- list()
  if (is.list(x) && length(x) && !is.null(names(x))) {
    out[[path]] <- x
  }
  if (is.list(x)) {
    keys <- if (is.null(names(x))) as.character(seq_along(x)) else names(x)
    for (i in seq_along(x)) {
      out <- c(out, .json_schema_objects(x[[i]], paste0(path, "/", keys[[i]])))
    }
  }
  out
}

test_that("installed schemas are the current FDS frame and study schemas only", {
  skip_if_not_installed("jsonlite")
  expect_setequal(
    names(.installed_json_schemas()),
    c("fds-v1.schema.json", "fds-study-v2.schema.json")
  )
})

test_that("closed schema objects declare every required property", {
  skip_if_not_installed("jsonlite")
  checked <- 0L
  schemas <- .installed_json_schemas()
  for (file in names(schemas)) {
    objects <- .json_schema_objects(schemas[[file]])
    for (path in names(objects)) {
      value <- objects[[path]]
      if (!identical(value$additionalProperties, FALSE) || is.null(value$required)) {
        next
      }
      checked <- checked + 1L
      undeclared <- setdiff(unlist(value$required), names(value$properties))
      expect_identical(undeclared, character(), info = paste0(file, " ", path))
    }
  }
  expect_gt(checked, 0L)
})

test_that("schema array dtypes enumerate the supported source dtypes", {
  skip_if_not_installed("jsonlite")
  schemas <- .installed_json_schemas()
  frame_dtype <- schemas[["fds-v1.schema.json"]][["$defs"]]$dtype$enum
  expect_setequal(unlist(frame_dtype), .supported_source_dtypes)
  expect_identical(
    schemas[["fds-v1.schema.json"]][["$defs"]]$array$properties$dtype[["$ref"]],
    "#/$defs/dtype"
  )
  study_dtype <- schemas[["fds-study-v2.schema.json"]][["$defs"]]$array$properties$dtype$enum
  expect_setequal(unlist(study_dtype), .supported_source_dtypes)
})

test_that("schema registries accept the empty array jsonlite writes for list()", {
  skip_if_not_installed("jsonlite")
  expect_identical(as.character(jsonlite::toJSON(list())), "[]")
  empty_array <- list(type = "array", maxItems = 0L)
  schemas <- .installed_json_schemas()
  for (file in names(schemas)) {
    schema <- schemas[[file]]
    expect_true(
      any(vapply(schema[["$defs"]]$record$anyOf, identical, logical(1), empty_array)),
      info = file
    )
    for (field in c("entities", "tables", "metadata", "extensions")) {
      expect_identical(
        schema$properties[[field]][["$ref"]], "#/$defs/record",
        info = paste(file, field)
      )
    }
  }
  frame <- .installed_json_schemas()[["fds-v1.schema.json"]]
  expect_identical(frame$properties$relations[["$ref"]], "#/$defs/record")
  study <- .installed_json_schemas()[["fds-study-v2.schema.json"]]
  for (field in c("arrays", "links")) {
    expect_true(
      any(vapply(study$properties[[field]]$anyOf, identical, logical(1), empty_array)),
      info = field
    )
  }
})

test_that("the study link schema declares every serialized link field", {
  skip_if_not_installed("jsonlite")
  link <- .installed_json_schemas()[["fds-study-v2.schema.json"]][["$defs"]]$link
  serialized <- c(
    "source", "target", "type", "map", "source_axis", "target_axis",
    "operator", "metadata", "schema_version"
  )
  expect_setequal(names(link$properties), serialized)
  expect_setequal(unlist(link$required), serialized)
})
