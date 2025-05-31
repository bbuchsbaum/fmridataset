library(testthat)

# minimal custom backend for testing
mock_backend <- bids_backend(
  "custom",
  backend_config = list(
    find_scans = function(...) NULL,
    read_metadata = function(...) NULL,
    get_run_info = function(...) NULL
  )
)


test_that("bids_backend validates backend_config type", {
  expect_error(bids_backend("custom", backend_config = "oops"),
               "backend_config must be a list")
})

create_query <- function() {
  bids_query("/tmp", backend = mock_backend)
}

test_that("subject.bids_query coerces and warns", {
  q <- create_query()
  expect_warning(q2 <- subject(q, 1), "character")
  expect_equal(q2$filters$subjects, "1")
  expect_warning(q3 <- subject(q, NA_character_), "NA")
  expect_null(q3$filters$subjects)
})

test_that("task/session/run/derivatives/space validate inputs", {
  q <- create_query()
  expect_warning(q <- task(q, 2), "character")
  expect_equal(q$filters$tasks, "2")

  q <- create_query()
  expect_warning(q <- session(q, NA), "NA")
  expect_null(q$filters$sessions)

  q <- create_query()
  expect_warning(q <- run(q, 3), "character")
  expect_equal(q$filters$runs, "3")

  q <- create_query()
  expect_warning(q <- derivatives(q, 4), "character")
  expect_equal(q$filters$derivatives, "4")

  q <- create_query()
  expect_warning(q <- space(q, 5), "character")
  expect_equal(q$filters$spaces, "5")
})

