# Get Registered Backend Information

Retrieves information about a registered backend or lists all registered
backends.

## Usage

``` r
get_backend_registry(name = NULL)
```

## Arguments

- name:

  Character string, name of backend to query. If NULL, returns all
  registrations.

## Value

For a specific backend: a list with registration details. For all
backends: a named list where each element contains registration details.

## Examples

``` r
# List all registered backends
get_backend_registry()
#> $h5
#> $h5$name
#> [1] "h5"
#> 
#> $h5$factory
#> function (source, mask_source, mask_dataset = "data/elements", 
#>     data_dataset = "data", preload = FALSE) 
#> {
#>     if (!requireNamespace("fmristore", quietly = TRUE)) {
#>         stop_fmridataset(fmridataset_error_config, message = "Package 'fmristore' is required for H5 backend but is not available", 
#>             parameter = "backend_type")
#>     }
#>     if (is.character(source)) {
#>         if (!all(file.exists(source))) {
#>             missing_files <- source[!file.exists(source)]
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("H5 source files not found: %s", 
#>                 paste(missing_files, collapse = ", ")), file = missing_files, 
#>                 operation = "open")
#>         }
#>     }
#>     else if (is.list(source)) {
#>         valid_types <- vapply(source, function(x) {
#>             inherits(x, "H5NeuroVec")
#>         }, logical(1))
#>         if (!all(valid_types)) {
#>             stop_fmridataset(fmridataset_error_config, message = "All source objects must be H5NeuroVec objects", 
#>                 parameter = "source")
#>         }
#>     }
#>     else {
#>         stop_fmridataset(fmridataset_error_config, message = "source must be character vector (H5 file paths) or list (H5NeuroVec objects)", 
#>             parameter = "source", value = class(source))
#>     }
#>     if (is.character(mask_source)) {
#>         if (!file.exists(mask_source)) {
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("H5 mask file not found: %s", 
#>                 mask_source), file = mask_source, operation = "open")
#>         }
#>     }
#>     else if (!inherits(mask_source, "NeuroVol") && !inherits(mask_source, 
#>         "H5NeuroVol")) {
#>         stop_fmridataset(fmridataset_error_config, message = "mask_source must be file path, NeuroVol, or H5NeuroVol object", 
#>             parameter = "mask_source", value = class(mask_source))
#>     }
#>     backend <- new.env(parent = emptyenv())
#>     backend$source <- source
#>     backend$mask_source <- mask_source
#>     backend$mask_dataset <- mask_dataset
#>     backend$data_dataset <- data_dataset
#>     backend$preload <- preload
#>     backend$h5_objects <- NULL
#>     backend$mask <- NULL
#>     backend$mask_vec <- NULL
#>     backend$dims <- NULL
#>     backend$metadata <- NULL
#>     class(backend) <- c("h5_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce37de1b50>
#> <environment: namespace:fmridataset>
#> 
#> $h5$description
#> [1] "HDF5 format backend using fmristore"
#> 
#> $h5$validate_function
#> NULL
#> 
#> $h5$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 
#> $latent
#> $latent$name
#> [1] "latent"
#> 
#> $latent$factory
#> function (source, preload = FALSE) 
#> {
#>     if (is.character(source)) {
#>         if (!all(file.exists(source))) {
#>             missing <- source[!file.exists(source)]
#>             stop_fmridataset(fmridataset_error_backend_io, sprintf("Source files not found: %s", 
#>                 paste(missing, collapse = ", ")), file = missing[1], 
#>                 operation = "create")
#>         }
#>         if (!all(grepl("\\.(lv\\.h5|h5)$", source, ignore.case = TRUE))) {
#>             stop_fmridataset(fmridataset_error_config, "All source files must be HDF5 files (.h5 or .lv.h5)", 
#>                 parameter = "source")
#>         }
#>     }
#>     else if (is.list(source)) {
#>         for (i in seq_along(source)) {
#>             item <- source[[i]]
#>             if (is.character(item)) {
#>                 if (length(item) != 1 || !file.exists(item)) {
#>                   stop_fmridataset(fmridataset_error_config, 
#>                     sprintf("Source item %d must be an existing file path", 
#>                       i), parameter = "source")
#>                 }
#>             }
#>             else if (!inherits(item, "LatentNeuroVec")) {
#>                 has_basis <- isS4(item) && "basis" %in% methods::slotNames(item)
#>                 if (!has_basis) {
#>                   stop_fmridataset(fmridataset_error_config, 
#>                     sprintf("Source item %d must be a LatentNeuroVec object or file path", 
#>                       i), parameter = "source")
#>                 }
#>             }
#>         }
#>     }
#>     else {
#>         stop_fmridataset(fmridataset_error_config, "source must be character vector or list", 
#>             parameter = "source", value = class(source))
#>     }
#>     backend <- list(source = source, preload = preload, data = NULL, 
#>         dims = NULL, is_open = FALSE)
#>     class(backend) <- c("latent_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce379e13e0>
#> <environment: namespace:fmridataset>
#> 
#> $latent$description
#> [1] "Latent space backend for dimension-reduced data"
#> 
#> $latent$validate_function
#> NULL
#> 
#> $latent$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 
#> $matrix
#> $matrix$name
#> [1] "matrix"
#> 
#> $matrix$factory
#> function (data_matrix, mask = NULL, spatial_dims = NULL, metadata = NULL) 
#> {
#>     if (!is.matrix(data_matrix)) {
#>         stop_fmridataset(fmridataset_error_config, message = "data_matrix must be a matrix", 
#>             parameter = "data_matrix", value = class(data_matrix))
#>     }
#>     n_timepoints <- nrow(data_matrix)
#>     n_voxels <- ncol(data_matrix)
#>     if (is.null(mask)) {
#>         mask <- rep(TRUE, n_voxels)
#>     }
#>     if (!is.logical(mask)) {
#>         stop_fmridataset(fmridataset_error_config, message = "mask must be a logical vector", 
#>             parameter = "mask", value = class(mask))
#>     }
#>     if (length(mask) != n_voxels) {
#>         stop_fmridataset(fmridataset_error_config, message = sprintf("mask length (%d) must equal number of columns (%d)", 
#>             length(mask), n_voxels), parameter = "mask")
#>     }
#>     if (is.null(spatial_dims)) {
#>         spatial_dims <- c(n_voxels, 1, 1)
#>     }
#>     if (length(spatial_dims) != 3 || !is.numeric(spatial_dims)) {
#>         stop_fmridataset(fmridataset_error_config, message = "spatial_dims must be a numeric vector of length 3", 
#>             parameter = "spatial_dims", value = spatial_dims)
#>     }
#>     if (prod(spatial_dims) != n_voxels) {
#>         stop_fmridataset(fmridataset_error_config, message = sprintf("Product of spatial_dims (%d) must equal number of voxels (%d)", 
#>             prod(spatial_dims), n_voxels), parameter = "spatial_dims")
#>     }
#>     backend <- list(data_matrix = data_matrix, mask = mask, spatial_dims = spatial_dims, 
#>         metadata = metadata %||% list())
#>     class(backend) <- c("matrix_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce37d08000>
#> <environment: namespace:fmridataset>
#> 
#> $matrix$description
#> [1] "In-memory matrix backend"
#> 
#> $matrix$validate_function
#> NULL
#> 
#> $matrix$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 
#> $nifti
#> $nifti$name
#> [1] "nifti"
#> 
#> $nifti$factory
#> function (source, mask_source, preload = FALSE, mode = c("normal", 
#>     "bigvec", "mmap", "filebacked"), dummy_mode = FALSE) 
#> {
#>     mode <- match.arg(mode)
#>     if (is.character(source)) {
#>         if (!dummy_mode && !all(file.exists(source))) {
#>             missing_files <- source[!file.exists(source)]
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("Source files not found: %s", 
#>                 paste(missing_files, collapse = ", ")), file = missing_files, 
#>                 operation = "open")
#>         }
#>     }
#>     else if (is.list(source)) {
#>         valid_types <- vapply(source, function(x) {
#>             inherits(x, "NeuroVec")
#>         }, logical(1))
#>         if (!all(valid_types)) {
#>             stop_fmridataset(fmridataset_error_config, message = "All source objects must be NeuroVec objects", 
#>                 parameter = "source")
#>         }
#>     }
#>     else {
#>         stop_fmridataset(fmridataset_error_config, message = "source must be character vector (file paths) or list (in-memory objects)", 
#>             parameter = "source", value = class(source))
#>     }
#>     if (is.character(mask_source)) {
#>         if (!dummy_mode && !file.exists(mask_source)) {
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("Mask file not found: %s", 
#>                 mask_source), file = mask_source, operation = "open")
#>         }
#>     }
#>     else if (!inherits(mask_source, "NeuroVol")) {
#>         stop_fmridataset(fmridataset_error_config, message = "mask_source must be file path or NeuroVol object", 
#>             parameter = "mask_source", value = class(mask_source))
#>     }
#>     backend <- new.env(parent = emptyenv())
#>     backend$source <- source
#>     backend$mask_source <- mask_source
#>     backend$preload <- preload
#>     backend$mode <- mode
#>     backend$dummy_mode <- dummy_mode
#>     backend$data <- NULL
#>     backend$mask <- NULL
#>     backend$mask_vec <- NULL
#>     backend$dims <- NULL
#>     backend$metadata <- NULL
#>     backend$run_length <- NULL
#>     backend$cache <- cachem::cache_mem(max_size = 64 * 1024^2, 
#>         evict = "lru")
#>     class(backend) <- c("nifti_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce37dd9218>
#> <environment: namespace:fmridataset>
#> 
#> $nifti$description
#> [1] "NIfTI format backend using neuroim2"
#> 
#> $nifti$validate_function
#> NULL
#> 
#> $nifti$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 
#> $study
#> $study$name
#> [1] "study"
#> 
#> $study$factory
#> function (backends, subject_ids = NULL, strict = getOption("fmridataset.mask_check", 
#>     "identical")) 
#> {
#>     if (!is.list(backends) || length(backends) == 0) {
#>         stop_fmridataset(fmridataset_error_config, message = "backends must be a non-empty list")
#>     }
#>     backends <- lapply(backends, function(b) {
#>         if (!inherits(b, "storage_backend")) {
#>             if (inherits(b, "matrix_dataset") && !is.null(b$datamat)) {
#>                 mask_logical <- as.logical(b$mask)
#>                 matrix_backend(b$datamat, mask = mask_logical)
#>             }
#>             else if (!is.null(b$backend)) {
#>                 b$backend
#>             }
#>             else {
#>                 b
#>             }
#>         }
#>         else {
#>             b
#>         }
#>     })
#>     lapply(backends, function(b) {
#>         if (!inherits(b, "storage_backend")) {
#>             stop_fmridataset(fmridataset_error_config, message = "all elements of backends must inherit from 'storage_backend'")
#>         }
#>     })
#>     if (is.null(subject_ids)) {
#>         subject_ids <- seq_along(backends)
#>     }
#>     if (length(subject_ids) != length(backends)) {
#>         stop_fmridataset(fmridataset_error_config, message = "subject_ids must match length of backends")
#>     }
#>     dims_list <- lapply(backends, backend_get_dims)
#>     spatial_dims <- lapply(dims_list, function(x) as.numeric(x$spatial))
#>     time_dims <- vapply(dims_list, function(x) x$time, numeric(1))
#>     ref_spatial <- spatial_dims[[1]]
#>     for (i in seq_along(spatial_dims[-1])) {
#>         sd <- spatial_dims[[i + 1]]
#>         if (!identical(sd, ref_spatial)) {
#>             stop_fmridataset(fmridataset_error_config, message = "spatial dimensions must match across backends")
#>         }
#>     }
#>     masks <- lapply(backends, backend_get_mask)
#>     ref_mask <- masks[[1]]
#>     if (strict == "identical") {
#>         for (m in masks[-1]) {
#>             if (!identical(m, ref_mask)) {
#>                 stop_fmridataset(fmridataset_error_config, message = "masks differ across backends")
#>             }
#>         }
#>         combined_mask <- ref_mask
#>     }
#>     else if (strict == "intersect") {
#>         for (m in masks[-1]) {
#>             overlap <- sum(m & ref_mask)/length(ref_mask)
#>             if (overlap < 0.95) {
#>                 stop_fmridataset(fmridataset_error_config, message = "mask overlap <95%")
#>             }
#>         }
#>         combined_mask <- Reduce("&", masks)
#>     }
#>     else {
#>         stop_fmridataset(fmridataset_error_config, message = "unknown strict setting")
#>     }
#>     subject_boundaries <- c(0L, cumsum(as.integer(time_dims)))
#>     backend <- list(backends = backends, subject_ids = subject_ids, 
#>         strict = strict, `_dims` = list(spatial = ref_spatial, 
#>             time = sum(time_dims)), `_mask` = combined_mask, 
#>         time_dims = as.integer(time_dims), subject_boundaries = as.integer(subject_boundaries))
#>     class(backend) <- c("study_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce379ee118>
#> <environment: namespace:fmridataset>
#> 
#> $study$description
#> [1] "Multi-subject study backend"
#> 
#> $study$validate_function
#> NULL
#> 
#> $study$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 
#> $zarr
#> $zarr$name
#> [1] "zarr"
#> 
#> $zarr$factory
#> function (source, data_key = "data", mask_key = "mask", preload = FALSE, 
#>     cache_size = 100) 
#> {
#>     if (!is.character(source) || length(source) != 1) {
#>         stop_fmridataset(fmridataset_error_config, "source must be a single character string", 
#>             parameter = "source", value = class(source))
#>     }
#>     if (!requireNamespace("Rarr", quietly = TRUE)) {
#>         stop_fmridataset(fmridataset_error_config, "The Rarr package is required for zarr_backend but is not installed.", 
#>             details = "Install with: BiocManager::install('Rarr')")
#>     }
#>     backend <- list(source = source, data_key = data_key, mask_key = mask_key, 
#>         preload = preload, cache_size = cache_size, store = NULL, 
#>         data_array = NULL, mask_array = NULL, dims = NULL, is_open = FALSE)
#>     class(backend) <- c("zarr_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce361c0508>
#> <environment: namespace:fmridataset>
#> 
#> $zarr$description
#> [1] "Zarr format backend"
#> 
#> $zarr$validate_function
#> NULL
#> 
#> $zarr$registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
#> 

# Get specific backend info
get_backend_registry("nifti")
#> $name
#> [1] "nifti"
#> 
#> $factory
#> function (source, mask_source, preload = FALSE, mode = c("normal", 
#>     "bigvec", "mmap", "filebacked"), dummy_mode = FALSE) 
#> {
#>     mode <- match.arg(mode)
#>     if (is.character(source)) {
#>         if (!dummy_mode && !all(file.exists(source))) {
#>             missing_files <- source[!file.exists(source)]
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("Source files not found: %s", 
#>                 paste(missing_files, collapse = ", ")), file = missing_files, 
#>                 operation = "open")
#>         }
#>     }
#>     else if (is.list(source)) {
#>         valid_types <- vapply(source, function(x) {
#>             inherits(x, "NeuroVec")
#>         }, logical(1))
#>         if (!all(valid_types)) {
#>             stop_fmridataset(fmridataset_error_config, message = "All source objects must be NeuroVec objects", 
#>                 parameter = "source")
#>         }
#>     }
#>     else {
#>         stop_fmridataset(fmridataset_error_config, message = "source must be character vector (file paths) or list (in-memory objects)", 
#>             parameter = "source", value = class(source))
#>     }
#>     if (is.character(mask_source)) {
#>         if (!dummy_mode && !file.exists(mask_source)) {
#>             stop_fmridataset(fmridataset_error_backend_io, message = sprintf("Mask file not found: %s", 
#>                 mask_source), file = mask_source, operation = "open")
#>         }
#>     }
#>     else if (!inherits(mask_source, "NeuroVol")) {
#>         stop_fmridataset(fmridataset_error_config, message = "mask_source must be file path or NeuroVol object", 
#>             parameter = "mask_source", value = class(mask_source))
#>     }
#>     backend <- new.env(parent = emptyenv())
#>     backend$source <- source
#>     backend$mask_source <- mask_source
#>     backend$preload <- preload
#>     backend$mode <- mode
#>     backend$dummy_mode <- dummy_mode
#>     backend$data <- NULL
#>     backend$mask <- NULL
#>     backend$mask_vec <- NULL
#>     backend$dims <- NULL
#>     backend$metadata <- NULL
#>     backend$run_length <- NULL
#>     backend$cache <- cachem::cache_mem(max_size = 64 * 1024^2, 
#>         evict = "lru")
#>     class(backend) <- c("nifti_backend", "storage_backend")
#>     backend
#> }
#> <bytecode: 0x55ce37dd9218>
#> <environment: namespace:fmridataset>
#> 
#> $description
#> [1] "NIfTI format backend using neuroim2"
#> 
#> $validate_function
#> NULL
#> 
#> $registered_at
#> [1] "2026-01-22 12:18:50 UTC"
#> 
```
