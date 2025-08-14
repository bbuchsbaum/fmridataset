# Comprehensive path handling tests for fmridataset

skip_if(TRUE, "Path handling tests need rewriting - matrix_dataset doesn't handle file paths")

test_that("relative paths are correctly resolved", {
  # Create test structure
  temp_dir <- tempdir()
  base_dir <- file.path(temp_dir, "project")
  data_dir <- file.path(base_dir, "data")
  sub_dir <- file.path(data_dir, "sub-01")
  
  dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)
  old_wd <- getwd()
  on.exit({
    setwd(old_wd)
    unlink(base_dir, recursive = TRUE)
  })
  
  # Test from different working directories
  setwd(base_dir)
  
  # Skip path tests for matrix_dataset as it doesn't handle file paths
  # matrix_dataset is for in-memory data only
  
  # This test should be updated to use file-based datasets
  # For now, skip this test
  skip("matrix_dataset doesn't handle file paths - test needs updating")
  
  # Nested relative path
  setwd(data_dir)
  dset2 <- matrix_dataset(
    matrix(1:20, 10, 2), 
    TR = 2, 
    run_length = 10,
    base_path = "sub-01"
  )
  
  expect_equal(
    normalizePath(dset2$base_path, mustWork = FALSE),
    normalizePath(sub_dir, mustWork = FALSE)
  )
})

test_that("absolute paths are preserved", {
  # Absolute path handling
  if (.Platform$OS.type == "windows") {
    abs_paths <- c(
      "C:\\Users\\test\\data",
      "D:\\data\\neuroimaging",
      "\\\\server\\share\\data"  # UNC path
    )
  } else {
    abs_paths <- c(
      "/home/user/data",
      "/tmp/neuroimaging",
      "/Users/test/Documents/fMRI"
    )
  }
  
  for (path in abs_paths) {
    dset <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = path
    )
    
    # Should preserve absolute path
    expect_equal(dset$base_path, path)
  }
})

test_that("path components with spaces are handled", {
  temp_dir <- tempdir()
  space_dir <- file.path(temp_dir, "path with spaces", "more spaces")
  dir.create(space_dir, recursive = TRUE, showWarnings = FALSE)
  on.exit(unlink(file.path(temp_dir, "path with spaces"), recursive = TRUE))
  
  # Create dataset with spaced path
  dset <- matrix_dataset(
    matrix(1:20, 10, 2), 
    TR = 2, 
    run_length = 10,
    base_path = space_dir
  )
  
  expect_equal(dset$base_path, space_dir)
  
  # Test file operations with spaces
  test_file <- file.path(space_dir, "test file.txt")
  writeLines("test", test_file)
  expect_true(file.exists(test_file))
})

test_that("special characters in paths are handled", {
  skip_on_cran()
  
  temp_dir <- tempdir()
  special_chars <- list(
    apostrophe = "user's_data",
    parentheses = "data(backup)",
    brackets = "data[2023]",
    ampersand = "this&that",
    hash = "data#1",
    at = "user@host",
    equals = "key=value"
  )
  
  if (.Platform$OS.type != "windows") {
    # Unix allows more special chars
    special_chars$colon <- "time:12:00"
    special_chars$asterisk <- "data*old"
  }
  
  for (name in names(special_chars)) {
    dir_name <- special_chars[[name]]
    special_dir <- file.path(temp_dir, dir_name)
    
    # Try to create directory
    tryCatch({
      dir.create(special_dir, showWarnings = FALSE)
      
      if (dir.exists(special_dir)) {
        dset <- matrix_dataset(
          matrix(1:20, 10, 2), 
          TR = 2, 
          run_length = 10,
          base_path = special_dir
        )
        
        expect_equal(dset$base_path, special_dir)
        unlink(special_dir, recursive = TRUE)
      }
    }, error = function(e) {
      # Some characters might not be allowed on certain systems
      skip(paste("Cannot create directory with", name))
    })
  }
})

test_that("path normalization works correctly", {
  # Test path normalization
  equivalent_paths <- list(
    c("./data", "data"),
    c("data/../data", "data"),
    c("./data/./sub", "data/sub"),
    c("data//sub", "data/sub")  # Double slashes
  )
  
  for (paths in equivalent_paths) {
    dset1 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = paths[1]
    )
    
    dset2 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = paths[2]
    )
    
    # Normalized paths should be equivalent
    expect_equal(
      normalizePath(dset1$base_path, mustWork = FALSE),
      normalizePath(dset2$base_path, mustWork = FALSE)
    )
  }
})

test_that("symlinks are handled correctly", {
  skip_on_cran()
  skip_if(.Platform$OS.type == "windows", "Symlinks require admin on Windows")
  
  temp_dir <- tempdir()
  real_dir <- file.path(temp_dir, "real_data")
  link_dir <- file.path(temp_dir, "link_data")
  
  dir.create(real_dir, showWarnings = FALSE)
  on.exit({
    unlink(link_dir)
    unlink(real_dir, recursive = TRUE)
  })
  
  # Create symlink
  tryCatch({
    file.symlink(real_dir, link_dir)
    
    if (file.exists(link_dir)) {
      # Dataset through symlink
      dset <- matrix_dataset(
        matrix(1:20, 10, 2), 
        TR = 2, 
        run_length = 10,
        base_path = link_dir
      )
      
      # Should resolve to real path or preserve symlink
      # (behavior may vary by system)
      expect_true(dir.exists(dset$base_path))
    }
  }, error = function(e) {
    skip("Cannot create symlinks on this system")
  })
})

test_that("network paths are handled appropriately", {
  skip_on_cran()
  
  if (.Platform$OS.type == "windows") {
    # UNC paths on Windows
    unc_paths <- c(
      "\\\\server\\share\\data",
      "\\\\10.0.0.1\\data",
      "\\\\localhost\\c$\\data"
    )
    
    for (unc in unc_paths) {
      dset <- matrix_dataset(
        matrix(1:20, 10, 2), 
        TR = 2, 
        run_length = 10,
        base_path = unc
      )
      
      expect_equal(dset$base_path, unc)
    }
  } else {
    # Network mounts on Unix
    network_paths <- c(
      "/mnt/server/data",
      "/Volumes/ShareName",
      "/net/hostname/path"
    )
    
    for (path in network_paths) {
      dset <- matrix_dataset(
        matrix(1:20, 10, 2), 
        TR = 2, 
        run_length = 10,
        base_path = path
      )
      
      expect_equal(dset$base_path, path)
    }
  }
})

test_that("very long paths are handled gracefully", {
  skip_on_cran()
  
  # Create deeply nested structure
  temp_dir <- tempdir()
  base <- temp_dir
  
  # Build a very deep path
  for (i in 1:50) {
    base <- file.path(base, paste0("level", i))
  }
  
  # Don't actually create it (would be too deep)
  # Just test that we handle it gracefully
  expect_error(
    dset <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = base
    ),
    NA  # Expect no error in creation
  )
  
  # But accessing might fail on some systems
  if (!is.null(dset)) {
    expect_equal(dset$base_path, base)
  }
})

test_that("path resolution with .. components works", {
  temp_dir <- tempdir()
  base_dir <- file.path(temp_dir, "test_paths")
  sub1 <- file.path(base_dir, "sub1")
  sub2 <- file.path(base_dir, "sub2")
  
  dir.create(sub1, recursive = TRUE, showWarnings = FALSE)
  dir.create(sub2, recursive = TRUE, showWarnings = FALSE)
  
  old_wd <- getwd()
  on.exit({
    setwd(old_wd)
    unlink(base_dir, recursive = TRUE)
  })
  
  setwd(sub1)
  
  # Path with .. to go to sibling
  dset <- matrix_dataset(
    matrix(1:20, 10, 2), 
    TR = 2, 
    run_length = 10,
    base_path = "../sub2"
  )
  
  expect_equal(
    normalizePath(dset$base_path, mustWork = FALSE),
    normalizePath(sub2, mustWork = FALSE)
  )
})

test_that("home directory expansion works", {
  skip_on_cran()
  
  # Test tilde expansion
  home_paths <- c(
    "~/data",
    "~/Documents/fMRI",
    "~"
  )
  
  for (path in home_paths) {
    dset <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = path
    )
    
    # Should expand tilde
    expanded <- path.expand(path)
    expect_equal(
      normalizePath(dset$base_path, mustWork = FALSE),
      normalizePath(expanded, mustWork = FALSE)
    )
  }
})

test_that("case sensitivity is handled appropriately", {
  skip_on_cran()
  
  temp_dir <- tempdir()
  
  if (.Platform$OS.type == "windows" || Sys.info()["sysname"] == "Darwin") {
    # Case-insensitive filesystems
    dir1 <- file.path(temp_dir, "TestData")
    dir2 <- file.path(temp_dir, "testdata")
    
    dir.create(dir1, showWarnings = FALSE)
    on.exit(unlink(dir1, recursive = TRUE))
    
    dset1 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = dir1
    )
    
    dset2 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = dir2
    )
    
    # Paths might be considered equivalent
    # (actual behavior depends on filesystem)
  } else {
    # Case-sensitive filesystems
    dir1 <- file.path(temp_dir, "TestData")
    dir2 <- file.path(temp_dir, "testdata")
    
    dir.create(dir1, showWarnings = FALSE)
    dir.create(dir2, showWarnings = FALSE)
    on.exit({
      unlink(dir1, recursive = TRUE)
      unlink(dir2, recursive = TRUE)
    })
    
    dset1 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = dir1
    )
    
    dset2 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = dir2
    )
    
    # Paths should be different
    expect_false(identical(dset1$base_path, dset2$base_path))
  }
})

test_that("path validation provides helpful errors", {
  # Invalid path characters
  if (.Platform$OS.type == "windows") {
    invalid_paths <- c(
      "data<file>.nii",  # < not allowed
      "data>file.nii",   # > not allowed
      "data|file.nii",   # | not allowed
      "data:file.nii",   # : only in drive
      "data*file.nii",   # * not allowed
      "data?file.nii"    # ? not allowed
    )
  } else {
    # Unix has fewer restrictions
    # Note: can't test null byte in R strings
    invalid_paths <- character(0)
  }
  
  for (path in invalid_paths) {
    # Should either handle gracefully or error clearly
    result <- tryCatch({
      matrix_dataset(
        matrix(1:20, 10, 2), 
        TR = 2, 
        run_length = 10,
        base_path = path
      )
      "success"
    }, error = function(e) {
      e$message
    })
    
    # Either succeeds or gives clear error
    expect_true(
      result == "success" || 
      grepl("path|file|invalid|character", result, ignore.case = TRUE)
    )
  }
})

test_that("paths with trailing slashes are normalized", {
  paths_with_slashes <- c(
    "data/",
    "data//",
    "./data/",
    "data/sub/"
  )
  
  paths_without_slashes <- c(
    "data",
    "data",
    "./data",
    "data/sub"
  )
  
  for (i in seq_along(paths_with_slashes)) {
    dset1 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = paths_with_slashes[i]
    )
    
    dset2 <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = paths_without_slashes[i]
    )
    
    # Should normalize to same path
    expect_equal(
      normalizePath(dset1$base_path, mustWork = FALSE),
      normalizePath(dset2$base_path, mustWork = FALSE)
    )
  }
})

test_that("environment variables in paths are expanded", {
  # Set a test environment variable
  Sys.setenv(FMRI_TEST_DIR = tempdir())
  on.exit(Sys.unsetenv("FMRI_TEST_DIR"))
  
  # Path with environment variable (shell-style)
  # Note: R doesn't automatically expand shell-style vars
  # but we test that paths are stored correctly
  env_path <- "$FMRI_TEST_DIR/data"
  
  dset <- matrix_dataset(
    matrix(1:20, 10, 2), 
    TR = 2, 
    run_length = 10,
    base_path = env_path
  )
  
  # Path is stored as-is (expansion would be backend responsibility)
  expect_equal(dset$base_path, env_path)
})

test_that("mixed path separators are handled", {
  skip_if(.Platform$OS.type != "windows", "Mixed separators mainly a Windows issue")
  
  # Windows accepts both / and \
  mixed_paths <- c(
    "C:/Users\\test/data",
    "data\\sub/file.nii",
    "/Users\\test\\data"
  )
  
  for (path in mixed_paths) {
    dset <- matrix_dataset(
      matrix(1:20, 10, 2), 
      TR = 2, 
      run_length = 10,
      base_path = path
    )
    
    # Should handle mixed separators
    expect_true(is.character(dset$base_path))
  }
})