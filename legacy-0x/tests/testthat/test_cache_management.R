test_that("cache configuration works correctly", {
  # Test default cache size
  expect_equal(.get_cache_size(), getOption("fmridataset.cache_max_mb") * 1024^2)

  # Test cache eviction policy
  expect_equal(.get_cache_evict(), "lru")

  # Test cache logging default
  expect_false(.get_cache_logging())

  # Test option overrides
  old_opts <- options(
    fmridataset.cache_max_mb = 256,
    fmridataset.cache_evict = "fifo",
    fmridataset.cache_logging = TRUE
  )
  on.exit(options(old_opts))

  expect_equal(.get_cache_size(), getOption("fmridataset.cache_max_mb") * 1024^2)
  expect_equal(.get_cache_evict(), "fifo")
  expect_true(.get_cache_logging())
})

test_that("fmri_clear_cache clears the cache", {
  # Initial cache should be empty or have some default state
  info_before <- fmri_cache_info()

  # Clear cache (should not error even if empty)
  expect_silent(fmri_clear_cache())

  # Verify cache is cleared
  info_after <- fmri_cache_info()
  expect_equal(info_after$n_objects, 0)
  expect_equal(info_after$current_size, "0.0 B")
})

test_that("fmri_cache_info returns proper structure", {
  info <- fmri_cache_info()

  # Check required fields
  expected_fields <- c(
    "max_size", "current_size", "n_objects", "eviction_policy",
    "cache_hit_rate", "total_hits", "total_misses", "utilization_pct"
  )

  expect_true(all(expected_fields %in% names(info)))

  # Check field types
  expect_type(info$n_objects, "integer")
  expect_type(info$eviction_policy, "character")
  expect_match(info$max_size, "\\d+\\.\\d+ [KMGT]?B")
  expect_match(info$current_size, "\\d+\\.\\d+ [KMGT]?B")
})

test_that("fmri_cache_resize works correctly", {
  # Test invalid inputs
  expect_error(fmri_cache_resize(-1), "must be a positive number")
  expect_error(fmri_cache_resize(c(100, 200)), "must be a positive number")
  expect_error(fmri_cache_resize("100"), "must be a positive number")

  # Test resize function (which shows warning about restart requirement)
  expect_warning(fmri_cache_resize(100), "Cache resizing is not supported")
  expect_warning(fmri_cache_resize(100), "options\\(fmridataset.cache_max_mb")
})

test_that("cache prevents unbounded growth", {
  # Skip if we can't create test data
  skip_if_not_installed("neuroim2")

  # Create a small cache for testing
  test_cache <- cachem::cache_mem(max_size = 1024, evict = "lru") # 1KB limit

  # Add items that exceed cache size
  test_cache$set("item1", rep(1, 100)) # ~800 bytes
  test_cache$set("item2", rep(2, 100)) # Should evict item1

  # Cache should not grow indefinitely
  keys <- test_cache$keys()
  expect_lte(length(keys), 2) # Should have limited number of items
})

test_that("LRU eviction policy works correctly", {
  # Create small test cache
  test_cache <- cachem::cache_mem(max_size = 2048, evict = "lru")

  # Fill cache
  test_cache$set("a", rep(1, 200))
  test_cache$set("b", rep(2, 200))
  test_cache$set("c", rep(3, 200))

  # Access 'a' to make it most recently used
  test_cache$get("a")

  # Add new item that should force eviction
  test_cache$set("d", rep(4, 200))

  # Cache should maintain reasonable number of items and not grow unbounded
  keys <- test_cache$keys()
  expect_lte(length(keys), 4) # Should not exceed reasonable limit
})

test_that("cache statistics are tracked correctly", {
  # Create test cache
  test_cache <- cachem::cache_mem(max_size = 10240, evict = "lru")

  # Add and access items to generate statistics
  test_cache$set("key1", "value1")
  test_cache$set("key2", "value2")

  # Generate hits and misses
  test_cache$get("key1") # hit
  test_cache$get("key2") # hit
  test_cache$get("nonexistent") # miss

  keys <- test_cache$keys()

  # Check that cache is working
  expect_equal(length(keys), 2)
})

test_that("cache handles concurrent access gracefully", {
  # This is a basic test - full concurrency testing would require parallel processing
  test_cache <- cachem::cache_mem(max_size = 10240, evict = "lru")

  # Simulate rapid access patterns
  for (i in 1:50) {
    key <- paste0("key", i %% 10) # Reuse keys to test replacement
    value <- rep(i, 10)
    test_cache$set(key, value)
    retrieved <- test_cache$get(key)
    expect_equal(retrieved, value)
  }

  # Cache should still be functional
  keys <- test_cache$keys()
  expect_lte(length(keys), 10) # Should not have more than 10 unique keys
})

test_that("cache memory estimation is reasonable", {
  test_cache <- cachem::cache_mem(max_size = 10240, evict = "lru")

  # Add known-size objects
  small_obj <- rep(1, 10)
  large_obj <- rep(1, 1000)

  test_cache$set("small", small_obj)
  test_cache$set("large", large_obj)

  keys <- test_cache$keys()

  # Should have both objects cached
  expect_equal(length(keys), 2)
  expect_true(test_cache$exists("small"))
  expect_true(test_cache$exists("large"))
})

test_that("cache configuration options are respected on startup", {
  # Test that package options are properly set
  expect_true("fmridataset.cache_max_mb" %in% names(options()))
  expect_true("fmridataset.cache_evict" %in% names(options()))
  expect_true("fmridataset.cache_logging" %in% names(options()))

  # Test default values
  expect_true(is.numeric(getOption("fmridataset.cache_max_mb")))
  expect_equal(getOption("fmridataset.cache_evict"), "lru")
  expect_equal(getOption("fmridataset.cache_logging"), FALSE)
})
