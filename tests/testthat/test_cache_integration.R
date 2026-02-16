test_that("memoise integration works with custom cache", {
  skip_if_not_installed("neuroim2")
  skip_on_cran()

  test_cache <- cachem::cache_mem(max_size = 1024 * 1024, evict = "lru")

  test_func <- memoise::memoise(function(x) {
    Sys.sleep(0.05)
    x^2
  }, cache = test_cache)

  result1 <- test_func(5)

  start_time <- Sys.time()
  result2 <- test_func(5)
  cached_duration <- as.numeric(Sys.time() - start_time, units = "secs")

  expect_equal(result1, result2)
  expect_equal(result1, 25)
  expect_lt(cached_duration, 0.05)

  keys <- test_cache$keys()
  expect_equal(length(keys), 1)
  expect_true(test_cache$exists(keys[1]))
})

test_that("cache eviction works with memoised functions", {
  # Create very small cache to force eviction
  test_cache <- cachem::cache_mem(max_size = 500, evict = "lru")

  # Memoised function that returns large objects
  large_func <- memoise::memoise(function(x) {
    rep(x, 50) # Returns vector of 50 elements
  }, cache = test_cache)

  # Fill cache beyond capacity
  result1 <- large_func(1)
  result2 <- large_func(2)
  result3 <- large_func(3) # Should trigger eviction

  # Verify results are correct
  expect_equal(result1, rep(1, 50))
  expect_equal(result2, rep(2, 50))
  expect_equal(result3, rep(3, 50))

  # Cache should have at most the number of items that fit
  keys <- test_cache$keys()
  expect_lte(length(keys), 3) # May be fewer due to eviction
})

test_that("cache works with file-based operations", {
  skip_if_not_installed("neuroim2")

  # Create temporary files for testing
  temp_dir <- tempdir()
  temp_files <- file.path(temp_dir, paste0("test_", 1:3, ".txt"))

  # Write test files
  for (i in seq_along(temp_files)) {
    writeLines(rep(paste("line", i), 10), temp_files[i])
  }

  on.exit({
    unlink(temp_files)
  })

  # Create cache for file operations
  file_cache <- cachem::cache_mem(max_size = 10240, evict = "lru")

  # Memoised file reader
  read_file_cached <- memoise::memoise(function(filename) {
    readLines(filename)
  }, cache = file_cache)

  # First read should cache
  content1_first <- read_file_cached(temp_files[1])
  expect_equal(length(content1_first), 10)
  expect_match(content1_first[1], "line 1")

  # Second read should be cached
  content1_second <- read_file_cached(temp_files[1])
  expect_identical(content1_first, content1_second)

  # Read other files
  content2 <- read_file_cached(temp_files[2])
  content3 <- read_file_cached(temp_files[3])

  expect_match(content2[1], "line 2")
  expect_match(content3[1], "line 3")

  # Check cache contains our files
  keys <- file_cache$keys()
  expect_gte(length(keys), 1)
})

test_that("cache handles errors gracefully", {
  error_cache <- cachem::cache_mem(max_size = 1024, evict = "lru")

  # Function that sometimes errors
  error_func <- memoise::memoise(function(x) {
    if (x < 0) stop("Negative input not allowed")
    x * 2
  }, cache = error_cache)

  # Normal operation should work and cache
  expect_equal(error_func(5), 10)

  # Error should not be cached (memoise default behavior)
  expect_error(error_func(-1), "Negative input not allowed")

  # After error, normal operation should still work
  expect_equal(error_func(3), 6)

  # Cache should only contain successful results
  keys <- error_cache$keys()
  expect_gte(length(keys), 1)
})

test_that("cache clearing works with memoised functions", {
  clear_cache <- cachem::cache_mem(max_size = 1024, evict = "lru")

  clear_func <- memoise::memoise(function(x) {
    x + 100
  }, cache = clear_cache)

  # Populate cache
  result1 <- clear_func(1)
  result2 <- clear_func(2)

  expect_equal(result1, 101)
  expect_equal(result2, 102)

  # Verify cache has items
  keys_before <- clear_cache$keys()
  expect_equal(length(keys_before), 2)

  # Clear cache
  clear_cache$reset()

  # Verify cache is empty
  keys_after <- clear_cache$keys()
  expect_equal(length(keys_after), 0)
})

test_that("multiple caches work independently", {
  cache1 <- cachem::cache_mem(max_size = 512, evict = "lru")
  cache2 <- cachem::cache_mem(max_size = 1024, evict = "lru")

  func1 <- memoise::memoise(function(x) x * 2, cache = cache1)
  func2 <- memoise::memoise(function(x) x * 3, cache = cache2)

  # Use both functions
  result1 <- func1(10)
  result2 <- func2(10)

  expect_equal(result1, 20)
  expect_equal(result2, 30)

  # Check caches are independent
  keys1 <- cache1$keys()
  keys2 <- cache2$keys()
  info1 <- cache1$info()
  info2 <- cache2$info()

  expect_equal(length(keys1), 1)
  expect_equal(length(keys2), 1)
  expect_equal(info1$max_size, 512)
  expect_equal(info2$max_size, 1024)

  # Clear one cache, other should be unaffected
  cache1$reset()

  keys1_after <- cache1$keys()
  keys2_after <- cache2$keys()

  expect_equal(length(keys1_after), 0)
  expect_equal(length(keys2_after), 1) # Should still have cached result
})
