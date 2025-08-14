library(testthat)
library(fmridataset)

test_that("fs::is_absolute_path correctly identifies absolute paths on Unix", {
  # Unix absolute paths
  expect_true(fs::is_absolute_path("/home/user/file.txt"))
  expect_true(fs::is_absolute_path("/"))
  expect_true(fs::is_absolute_path("/var/log/test.log"))
  
  # Unix relative paths
  expect_false(fs::is_absolute_path("relative/path.txt"))
  expect_false(fs::is_absolute_path("./file.txt"))
  expect_false(fs::is_absolute_path("../parent/file.txt"))
  expect_false(fs::is_absolute_path("file.txt"))
  expect_false(fs::is_absolute_path(""))
})

test_that("fs::is_absolute_path correctly identifies absolute paths on Windows", {
  # Windows absolute paths
  expect_true(fs::is_absolute_path("C:\\Program Files\\app.exe"))
  expect_true(fs::is_absolute_path("D:\\data\\file.txt"))
  expect_true(fs::is_absolute_path("Z:\\network\\share.dat"))
  expect_true(fs::is_absolute_path("c:\\lowercase.txt"))
  
  # Windows relative paths
  expect_false(fs::is_absolute_path("relative\\path.txt"))
  expect_false(fs::is_absolute_path(".\\file.txt"))
  expect_false(fs::is_absolute_path("..\\parent\\file.txt"))
})

test_that("fs::is_absolute_path handles edge cases", {
  # Empty and special characters
  expect_false(fs::is_absolute_path(""))
  expect_false(fs::is_absolute_path(" "))
  # Note: fs treats "~" as absolute path on Unix systems
  expect_true(fs::is_absolute_path("~"))
  
  # Mixed separators (common on Windows with cross-platform code)
  expect_true(fs::is_absolute_path("C:/mixed/separators.txt"))
  expect_false(fs::is_absolute_path("relative/mixed\\separators.txt"))
  
  # UNC paths on Windows - fs package should handle UNC paths properly
  if (.Platform$OS.type == "windows") {
    expect_true(fs::is_absolute_path("\\\\server\\share"))
  }
  
  # Special cases
  expect_false(fs::is_absolute_path("file:/path"))  # URI scheme
  expect_false(fs::is_absolute_path("http://example.com"))
})

test_that("fs::is_absolute_path works with vectors", {
  paths <- c(
    "/absolute/unix",
    "relative/unix", 
    "C:\\absolute\\windows",
    "relative\\windows",
    ""
  )
  
  expected <- c(TRUE, FALSE, TRUE, FALSE, FALSE)
  expect_equal(fs::is_absolute_path(paths), expected)
})

test_that("fs::is_absolute_path handles NULL and NA values", {
  # fs package returns logical(0) for NULL rather than throwing error
  expect_length(fs::is_absolute_path(NULL), 0)
  expect_type(fs::is_absolute_path(NULL), "logical")
  
  # NA handling - fs preserves NA values but returns FALSE for NA_character_
  result <- fs::is_absolute_path(c("valid", NA_character_))
  expect_length(result, 2)
  expect_false(result[2])  # fs returns FALSE for NA_character_
})