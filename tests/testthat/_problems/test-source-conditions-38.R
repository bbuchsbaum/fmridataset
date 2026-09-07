# Extracted from test-source-conditions.R:38

# test -------------------------------------------------------------------------
bogus <- structure(list(shape = c(2L, 2L)), class = c("bogus_source", "array_source"))
err <- expect_error(
    validate_array_source(bogus),
    class = "fmridataset_error_source_contract"
  )
