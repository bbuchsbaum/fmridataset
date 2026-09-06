#' Null-coalescing operator
#'
#' If x is NULL, return y; otherwise return x
#' @name grapes-or-or-grapes
#' @keywords internal
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
