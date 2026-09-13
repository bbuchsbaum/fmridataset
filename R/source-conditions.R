#' Build the structured conditions an array source may signal
#'
#' Storage packages implement the [array source][array-source] protocol and
#' must fail the way built-in sources fail, so that callers can handle every
#' source alike. `source_error()` builds the condition object for the three
#' failure kinds the protocol defines; signal it with `stop()`.
#'
#' - `"stale"` (`fmridataset_error_source_stale`): the physical store no longer
#'   matches the descriptor's revision evidence (a file was rewritten, a store
#'   was replaced). Supply `source`, `expected`, and `actual` so the caller can
#'   report what changed.
#' - `"io"` (`fmridataset_error_backend_io`): a genuine read, open, or close
#'   failure. Supply `file` and `operation` where known.
#' - `"contract"` (`fmridataset_error_source_contract`): the descriptor or a
#'   method violates the protocol. Supply `field`.
#'
#' Every condition also carries the `fmridataset_error` class, so a single
#' handler can catch all package errors.
#'
#' @param message One-sentence description of the failure.
#' @param type One of `"stale"`, `"io"`, or `"contract"`.
#' @param ... Named fields stored on the condition, such as `source`,
#'   `expected`, `actual`, `file`, `operation`, or `field`.
#' @return A condition object inheriting from the type-specific class,
#'   `fmridataset_error`, `error`, and `condition`.
#' @examples
#' cond <- source_error(
#'   "Store was rewritten after the descriptor was built.",
#'   type = "stale", source = "example.h5",
#'   expected = "abc", actual = "def"
#' )
#' inherits(cond, "fmridataset_error_source_stale")
#' tryCatch(stop(cond), fmridataset_error_source_stale = function(e) e$actual)
#' @export
source_error <- function(message, type = c("stale", "io", "contract"), ...) {
  type <- match.arg(type)
  if (!is.character(message) || length(message) != 1L || is.na(message) ||
    !nzchar(message)) {
    .frame_abort(
      "message must be one non-empty string.",
      "fmridataset_error_source_contract",
      field = "message"
    )
  }
  fields <- list(...)
  if (length(fields) && (is.null(names(fields)) || any(!nzchar(names(fields))))) {
    .frame_abort(
      "Condition fields must be named.",
      "fmridataset_error_source_contract",
      field = "..."
    )
  }
  class <- switch(type,
    stale = "fmridataset_error_source_stale",
    io = "fmridataset_error_backend_io",
    contract = "fmridataset_error_source_contract"
  )
  do.call(fmridataset_error, c(list(message = message, class = class), fields))
}

.source_generics <- c(
  "source_shape", "source_dtype", "source_chunks", "source_capabilities",
  "source_fingerprint", "source_open", "source_read", "source_close"
)

# Resolve one protocol method the way S3 dispatch from `envir` would: a
# function visible from the caller's scope first, then the registry of the
# generic's namespace. Methods defined inside a local() or test_that() block
# are visible to the caller but not to this namespace, so resolving from here
# reported a complete class as implementing nothing.
.resolve_source_method <- function(generic, cls, envir) {
  method <- utils::getS3method(generic, cls, optional = TRUE, envir = envir)
  if (!is.null(method)) {
    return(method)
  }
  name <- paste(generic, cls, sep = ".")
  if (exists(name, envir = envir, mode = "function")) {
    return(get(name, envir = envir, mode = "function"))
  }
  utils::getS3method(generic, cls, optional = TRUE)
}

# A descriptor that claims to be an array source but implements none of the
# protocol used to fail deep inside source_descriptor() with a bare
# "no applicable method" error. Name the missing methods up front instead.
.assert_source_methods <- function(x, envir = parent.frame()) {
  if (!inherits(x, "array_source")) {
    .frame_abort(
      "Object does not inherit from array_source.",
      "fmridataset_error_source_contract",
      field = "class"
    )
  }
  classes <- class(x)
  missing <- vapply(.source_generics, function(generic) {
    !any(vapply(classes, function(cls) {
      !is.null(.resolve_source_method(generic, cls, envir))
    }, logical(1)))
  }, logical(1))
  if (any(missing)) {
    .frame_abort(
      sprintf(
        "Class '%s' implements no method for: %s.",
        classes[[1L]], paste(.source_generics[missing], collapse = ", ")
      ),
      "fmridataset_error_source_contract",
      field = "methods",
      missing = .source_generics[missing],
      classes = classes
    )
  }
  invisible(TRUE)
}
