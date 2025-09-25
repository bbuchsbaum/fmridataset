.onLoad <- function(libname, pkgname) {
  # Register built-in backends
  register_builtin_backends()
  
  # Set default options if not already set
  op <- options()
  op.fmridataset <- list(
    fmridataset.cache_max_mb = 512,         # Main cache size (512MB default)
    fmridataset.cache_evict = "lru",        # LRU eviction policy
    fmridataset.cache_logging = FALSE,      # Cache logging disabled by default
    fmridataset.study_cache_mb = 1024,      # Study backend cache size
    fmridataset.block_size_mb = 64,         # Block size for chunked operations
    fmridataset.cache_threshold = 0.1       # 10% of cache size threshold
  )
  toset <- !(names(op.fmridataset) %in% names(op))
  if (any(toset)) options(op.fmridataset[toset])
  
  invisible()
}
