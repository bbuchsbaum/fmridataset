library(testthat)
library(fmridataset)

test_that("is_absolute_path correctly identifies absolute paths on Unix", {
  # Unix absolute paths
  expect_true(fmridataset:::is_absolute_path("/home/user/file.txt"))
  expect_true(fmridataset:::is_absolute_path("/"))
  expect_true(fmridataset:::is_absolute_path("/var/log/test.log"))
  
  # Unix relative paths
  expect_false(fmridataset:::is_absolute_path("relative/path.txt"))
  expect_false(fmridataset:::is_absolute_path("./file.txt"))
  expect_false(fmridataset:::is_absolute_path("../parent/file.txt"))
  expect_false(fmridataset:::is_absolute_path("file.txt"))
  expect_false(fmridataset:::is_absolute_path(""))
})

test_that("is_absolute_path correctly identifies absolute paths on Windows", {
  # Windows absolute paths
  expect_true(fmridataset:::is_absolute_path("C:\\Program Files\\app.exe"))
  expect_true(fmridataset:::is_absolute_path("D:\\data\\file.txt"))
  expect_true(fmridataset:::is_absolute_path("Z:\\network\\share.dat"))
  expect_true(fmridataset:::is_absolute_path("c:\\lowercase.txt"))
  
  # Windows relative paths
  expect_false(fmridataset:::is_absolute_path("relative\\path.txt"))
  expect_false(fmridataset:::is_absolute_path(".\\file.txt"))
  expect_false(fmridataset:::is_absolute_path("..\\parent\\file.txt"))
})

test_that("is_absolute_path handles edge cases", {
  # Empty and special characters
  expect_false(fmridataset:::is_absolute_path(""))
  expect_false(fmridataset:::is_absolute_path(" "))
  expect_false(fmridataset:::is_absolute_path("~"))
  expect_false(fmridataset:::is_absolute_path("~/home"))
  
  # Mixed separators (common on Windows with cross-platform code)
  expect_true(fmridataset:::is_absolute_path("C:/mixed/separators.txt"))
  expect_false(fmridataset:::is_absolute_path("relative/mixed\\separators.txt"))
  
  # UNC paths on Windows
  expect_false(fmridataset:::is_absolute_path("\\\\server\\share"))  # UNC not handled by simple regex
  
  # Special cases
  expect_false(fmridataset:::is_absolute_path("file:/path"))  # URI scheme
  expect_false(fmridataset:::is_absolute_path("http://example.com"))
})

test_that("is_absolute_path works with vectors", {
  paths <- c(
    "/absolute/unix",
    "relative/unix", 
    "C:\\absolute\\windows",
    "relative\\windows",
    ""
  )
  
  expected <- c(TRUE, FALSE, TRUE, FALSE, FALSE)
  expect_equal(fmridataset:::is_absolute_path(paths), expected)
})

test_that("is_absolute_path handles NULL and NA values", {
  # These should not occur in normal usage but test defensive programming
  expect_error(fmridataset:::is_absolute_path(NULL))
  expect_equal(fmridataset:::is_absolute_path(c("valid", NA)), c(FALSE, NA))
})