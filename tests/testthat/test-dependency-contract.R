.description_fields <- function() {
  as.list(utils::packageDescription("fmridataset"))
}

.dependency_names <- function(field) {
  trimws(sub("\\s*\\(.*$", "", unlist(strsplit(field, ",", fixed = TRUE))))
}

.dcf_one_line <- function(field) gsub("\\s+", " ", field)

test_that("the hard dependency surface is minimal and versioned", {
  fields <- .description_fields()
  expected <- c(
    "delarr", "digest", "Matrix", "methods", "neuroim2",
    "rlang", "tibble", "utils", "uuid"
  )

  expect_setequal(.dependency_names(fields$Imports), expected)
  imports <- .dcf_one_line(fields$Imports)
  expect_match(imports, "delarr \\(>= 0\\.1\\.0\\)")
  expect_match(imports, "neuroim2 \\(>= 0\\.19\\.0\\)")
})

test_that("consumer packages cannot form a development dependency cycle", {
  fields <- .description_fields()
  consumers <- c("fmristore", "multidesign", "fmrigds")

  expect_length(intersect(.dependency_names(fields$Suggests), consumers), 0L)
  expect_false(any(vapply(consumers, grepl, logical(1), x = fields$Remotes,
                          fixed = TRUE)))
})

test_that("custom dependency remotes are immutable", {
  fields <- .description_fields()
  remotes <- trimws(unlist(strsplit(fields$Remotes, ",", fixed = TRUE)))
  custom <- c("delarr", "neuroim2", "neuroatlas", "neurosurf")

  expect_setequal(sub("^bbuchsbaum/([^@]+)@.*$", "\\1", remotes), custom)
  expect_true(all(grepl("@[0-9a-f]{40}$", remotes)))
})
