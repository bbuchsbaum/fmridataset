library(testthat)
library(fmridataset)

# Mock LatentNeuroVec for testing - using S3 for simpler mocking
if (!methods::isClass("MockLatentNeuroVec")) {
  setClass("MockLatentNeuroVec", 
    slots = c(
      basis = "matrix", 
      loadings = "matrix", 
      mask = "logical"
    )
  )
}

# Helper to create mock LatentNeuroVec
create_mock_latent_neurovec <- function(n_time = 10, n_vox = 100, n_comp = 5, 
                                       spatial_dims = c(10, 10, 1)) {
  basis <- matrix(rnorm(n_time * n_comp), nrow = n_time, ncol = n_comp)
  loadings <- matrix(rnorm(n_vox * n_comp), nrow = n_vox, ncol = n_comp)
  mask <- rep(TRUE, n_vox)
  
  # Create the mock object without space_obj slot
  obj <- new("MockLatentNeuroVec", 
             basis = basis, 
             loadings = loadings, 
             mask = mask)
  
  # Add space info as an attribute that we can use in the mocked space function
  attr(obj, "space_dims") <- c(spatial_dims, n_time)
  attr(obj, "space_spacing") <- c(1, 1, 1)
  attr(obj, "space_origin") <- c(0, 0, 0)
  
  obj
}

# Mock dim method for MockNeuroSpace using S3
dim.MockNeuroSpace <- function(x) {
  x$dims
}

test_that("latent_backend constructor validates inputs", {
  # Valid character vector input
  expect_error(latent_backend(c("file1.lv.h5", "file2.lv.h5")), "All source files must exist")
  
  # Valid list input with mock objects
  lvec1 <- create_mock_latent_neurovec()
  lvec2 <- create_mock_latent_neurovec()
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec1, lvec2))
      
      expect_s3_class(backend, "latent_backend")
      expect_s3_class(backend, "storage_backend")
      expect_equal(length(backend$source), 2)
      expect_false(backend$preload)
    }
  )
})

test_that("latent_backend handles single object input", {
  lvec <- create_mock_latent_neurovec()
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec), preload = TRUE)
      
      expect_s3_class(backend, "latent_backend")
      expect_true(backend$preload)
      expect_equal(length(backend$source), 1)
    }
  )
})

test_that("latent_backend validates object types in list", {
  invalid_obj <- list(basis = matrix(1:10, 2, 5))  # Wrong class
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      expect_error(
        latent_backend(list(invalid_obj)),
        "Source item 1 must be a LatentNeuroVec object or file path"
      )
    }
  )
})

test_that("latent_backend validates mixed input types", {
  # Test that all items in list must be valid
  lvec <- create_mock_latent_neurovec()
  
  # This should work for existing file
  temp_file <- tempfile(fileext = ".lv.h5")
  file.create(temp_file)
  on.exit(unlink(temp_file))
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      mixed_list <- list(lvec, temp_file)
      backend <- latent_backend(mixed_list)
      expect_s3_class(backend, "latent_backend")
    }
  )
})

test_that("latent_backend rejects invalid source types", {
  expect_error(
    latent_backend(123),
    "source must be a character vector of file paths or a list of LatentNeuroVec objects"
  )
  
  expect_error(
    latent_backend(data.frame(a = 1)),
    "Source item 1 must be a LatentNeuroVec object or file path"
  )
})

test_that("backend_open works with mock objects", {
  lvec1 <- create_mock_latent_neurovec(n_time = 10)
  lvec2 <- create_mock_latent_neurovec(n_time = 15)
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec1, lvec2))
      
      # Mock the fmristore package check
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) pkg == "fmristore",
        .package = "base",
        {
          opened_backend <- backend_open(backend)
          expect_true(opened_backend$is_open)
          expect_equal(length(opened_backend$data), 2)
        }
      )
    }
  )
})

test_that("backend_open fails without fmristore", {
  lvec <- create_mock_latent_neurovec()
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec))
      
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) FALSE,
        .package = "base",
        {
          expect_error(
            backend_open(backend),
            "The fmristore package is required"
          )
        }
      )
    }
  )
})

test_that("backend_close works correctly", {
  lvec <- create_mock_latent_neurovec()
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec))
      
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) TRUE,
        .package = "base",
        {
          opened_backend <- backend_open(backend)
          closed_backend <- backend_close(opened_backend)
          expect_false(closed_backend$is_open)
          expect_null(closed_backend$data)
        }
      )
    }
  )
})

test_that("backend_get_dims returns correct dimensions", {
  lvec1 <- create_mock_latent_neurovec(n_time = 10, n_comp = 5)
  lvec2 <- create_mock_latent_neurovec(n_time = 15, n_comp = 5)
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec1, lvec2))
      
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) TRUE,
        .package = "base",
        {
          opened_backend <- backend_open(backend)
          dims <- backend_get_dims(opened_backend)
          
          expect_equal(dims$space, c(10, 10, 1))
          expect_equal(dims$time, 25)  # 10 + 15
          expect_equal(dims$n_runs, 2)
          expect_equal(dims$n_components, 5)
          expect_equal(dims$data_dims, c(25, 5))
        }
      )
    }
  )
})

test_that("backend_get_mask returns component mask", {
  lvec <- create_mock_latent_neurovec(n_comp = 8)
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec))
      
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) TRUE,
        .package = "base",
        {
          opened_backend <- backend_open(backend)
          mask <- backend_get_mask(opened_backend)
          
          expect_true(is.logical(mask))
          expect_equal(length(mask), 8)  # n_components
          expect_true(all(mask))  # All components active
        }
      )
    }
  )
})

test_that("backend_get_data returns latent scores", {
  lvec1 <- create_mock_latent_neurovec(n_time = 8, n_comp = 3)
  lvec2 <- create_mock_latent_neurovec(n_time = 7, n_comp = 3)
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec1, lvec2))
      
      with_mocked_bindings(
        requireNamespace = function(pkg, quietly = TRUE) TRUE,
        .package = "base",
        {
          opened_backend <- backend_open(backend)
          
          # Test getting all data
          data <- backend_get_data(opened_backend)
          expect_equal(dim(data), c(15, 3))  # 15 total timepoints, 3 components
          
          # Test row subsetting
          data_subset <- backend_get_data(opened_backend, rows = 1:5)
          expect_equal(dim(data_subset), c(5, 3))
          
          # Test column subsetting
          data_cols <- backend_get_data(opened_backend, cols = 1:2)
          expect_equal(dim(data_cols), c(15, 2))
          
          # Test both row and column subsetting
          data_both <- backend_get_data(opened_backend, rows = 3:7, cols = 2:3)
          expect_equal(dim(data_both), c(5, 2))
        }
      )
    }
  )
})

test_that("backend operations require open backend", {
  lvec <- create_mock_latent_neurovec()
  
  with_mocked_bindings(
    space = function(x) {
      space_obj <- list(
        dims = attr(x, "space_dims"),
        spacing = attr(x, "space_spacing"),
        origin = attr(x, "space_origin")
      )
      class(space_obj) <- c("MockNeuroSpace", "list")
      space_obj
    },
    .package = "neuroim2",
    {
      backend <- latent_backend(list(lvec))
      
      # Backend not opened yet
      expect_error(
        backend_get_dims(backend),
        "Backend must be opened before getting dimensions"
      )
      
      expect_error(
        backend_get_mask(backend), 
        "Backend must be opened before getting mask"
      )
      
      expect_error(
        backend_get_data(backend),
        "Backend must be opened before getting data"
      )
      
      expect_error(
        backend_get_metadata(backend),
        "Backend must be opened before getting metadata"
      )
    }
  )
})