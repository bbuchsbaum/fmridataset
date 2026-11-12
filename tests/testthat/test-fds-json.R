.json_round_trip <- function(x) {
  path <- withr::local_tempfile(fileext = ".json", .local_envir = parent.frame())
  fmridataset:::.fds_json_write(x, path)
  fmridataset:::.fds_json_read(path)$value
}

test_that("FDS JSON round-trips every manifest value type exactly", {
  values <- list(
    scalars = list(a = "x", b = TRUE, c = 2L, d = 2, e = 1 / 3, f = -0, g = NA, h = NA_integer_, i = NaN, j = Inf),
    vectors = list(c("a", NA), c(TRUE, NA), c(1L, NA, -5L), c(1.5, NA, NaN, -Inf, -0, 1e-300)),
    empty = list(character(), integer(), double(), logical(), list(), stats::setNames(list(), character()), NULL),
    named_vector = c(a = 1.5, b = 2),
    factor = factor(c("b", "a", NA), levels = c("b", "a")),
    ordered = factor(c("lo", "hi"), levels = c("lo", "hi"), ordered = TRUE),
    date = as.Date(c("2024-01-02", NA)),
    datetime = as.POSIXct(c(0, 1.5), origin = "1970-01-01", tz = "America/Toronto"),
    difftime = as.difftime(c(1, 2.5), units = "mins"),
    matrix = matrix(c(1.5, 2, NA, 4), 2, dimnames = list(c("r1", "r2"), NULL)),
    tibble = tibble::tibble(
      id = c("a", "b"), n = 1:2, x = c(0.5, NA),
      when = as.Date(c("2020-01-01", "2021-01-01")), f = factor(c("u", "v")),
      v = list(1:2, "z")
    ),
    data_frame = data.frame(a = 1:3, b = c("x", "y", "z")),
    record = list(site = "A", scanner = list(vendor = "X", field = 3), tags = list("a", "b")),
    awkward_names = list(
      list(`$weird` = 1, ok = 2),
      stats::setNames(list(1, 2), c("a", "a")),
      list(a = NULL)
    ),
    other = list(complex(real = 1, imaginary = -2), as.raw(c(0, 255, 16)), raw()),
    classed = structure(list(policy = "require", namespace = NULL), class = c("fmri_id_policy", "list"))
  )
  for (name in names(values)) {
    expect_identical(.json_round_trip(values[[name]]), values[[name]], info = name)
  }
})

test_that("doubles survive with all 17 significant digits", {
  set.seed(7)
  x <- c(
    rnorm(500) * 10^stats::runif(500, -300, 300),
    .Machine$double.eps, .Machine$double.xmax, .Machine$double.xmin, 5e-324, 0.1 + 0.2
  )
  expect_identical(.json_round_trip(x), x)
})

test_that("sparse matrices keep their canonical bytes", {
  sparse <- Matrix::sparseMatrix(i = c(1, 3), j = c(2, 1), x = c(0.5, -2), dims = c(3, 2))
  expect_identical(canonical_bytes(.json_round_trip(sparse)), canonical_bytes(sparse))
})

test_that("FDS JSON is plain where it can be and refuses runtime state", {
  path <- withr::local_tempfile(fileext = ".json")
  fmridataset:::.fds_json_write(list(id = "s", n = 2L, x = 2), path)
  text <- paste(readLines(path), collapse = "\n")
  expect_match(text, '"id": "s"', fixed = TRUE)
  expect_match(text, '"n": 2,', fixed = TRUE)
  expect_match(text, '"x": 2.0', fixed = TRUE)
  expect_error(fmridataset:::.fds_json_write(list(f = function() 1), path), class = "fmridataset_error_schema")
  writeLines('{"format": "something-else", "value": 1}', path)
  expect_error(fmridataset:::.fds_json_read(path), "not an FDS JSON document", class = "fmridataset_error_schema")
})
