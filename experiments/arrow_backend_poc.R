# Proof-of-concept Arrow backend seed ------------------------------------
# This lightweight example validates the design for an Arrow-backed seed.

if (requireNamespace("arrow", quietly = TRUE) &&
    requireNamespace("DelayedArray", quietly = TRUE)) {

  setClass("ArrowBackendSeed",
           slots = list(table = "Table"),
           contains = "Array")

  setMethod("dim", "ArrowBackendSeed", function(x) {
    c(x@table$num_rows, x@table$num_columns)
  })

  setMethod("extract_array", "ArrowBackendSeed", function(x, index) {
    rows <- if (length(index) >= 1) index[[1]] else NULL
    cols <- if (length(index) >= 2) index[[2]] else NULL
    as.matrix(x@table[rows, cols])
  })

  arrow_backend_seed_poc <- function() {
    tbl <- arrow::arrow_table(matrix(1:9, nrow = 3))
    DelayedArray::DelayedArray(new("ArrowBackendSeed", table = tbl))
  }

}
