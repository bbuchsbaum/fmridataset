#' Null-coalescing operator
#'
#' If x is NULL, return y; otherwise return x
#' @name grapes-or-or-grapes
#' @param x A value to test for `NULL`.
#' @param y The fallback value returned when `x` is `NULL`.
#' @return `y` if `x` is `NULL`; otherwise `x`.
#' @keywords internal
#' @examples
#' fmridataset:::`%||%`(NULL, 1)
#' fmridataset:::`%||%`(2, 1)
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
