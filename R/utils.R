#' Null-coalescing operator
#'
#' If x is NULL, return y; otherwise return x
#' @name grapes-or-or-grapes
#' @keywords internal
#' @export
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

#' Internal wrapper for requireNamespace
#'
#' Centralises optional dependency checks so tests can mock behaviour.
#' @keywords internal
.require_namespace <- function(package, ...) {
  base::requireNamespace(package, ...)
}

