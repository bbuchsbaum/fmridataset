#' Null-coalescing operator
#'
#' If x is NULL, return y; otherwise return x. Internal; not part of the
#' public namespace.
#'
#' @param x A value to test for `NULL`.
#' @param y The fallback value returned when `x` is `NULL`.
#' @return `y` if `x` is `NULL`; otherwise `x`.
#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
