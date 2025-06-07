library(fmridataset)

test_that("error constructors create structured errors", {
  err <- fmridataset:::fmridataset_error_backend_io("oops", file = "f.h5", operation = "read")
  expect_s3_class(err, "fmridataset_error_backend_io")
  expect_match(err$message, "oops")
  expect_equal(err$file, "f.h5")
  expect_equal(err$operation, "read")

  err2 <- fmridataset:::fmridataset_error_config("bad", parameter = "x", value = 1)
  expect_s3_class(err2, "fmridataset_error_config")
  expect_equal(err2$parameter, "x")
  expect_equal(err2$value, 1)
})


test_that("stop_fmridataset throws the constructed error", {
  expect_error(
    fmridataset:::stop_fmridataset(fmridataset:::fmridataset_error_config, "bad", parameter = "y"),
    class = "fmridataset_error_config"
  )
})
