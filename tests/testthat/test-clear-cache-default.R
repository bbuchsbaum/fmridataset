test_that("clear_cache() errors on unsupported classes", {
  expect_error(clear_cache(42),
               "No clear_cache method for class: numeric")
})
