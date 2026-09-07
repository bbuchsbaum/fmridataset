# Extracted from test-conformance-coverage.R:92

# prequel ----------------------------------------------------------------------
fmristore_has_h5_array_source <- function() {
  requireNamespace("fmristore", quietly = TRUE) &&
    requireNamespace("hdf5r", quietly = TRUE) &&
    "h5_array_source" %in% getNamespaceExports("fmristore")
}

# test -------------------------------------------------------------------------
expect_feature_space_conformance(index_space(6))
