# Lossless JSON encoding of manifest values (FDS JSON, version 1).
#
# FDS manifests hold R values whose types matter to their digests: integer and
# double are different, NA is not NaN, a factor is not a character vector, and
# a tibble carries its class and row names. Plain JSON cannot tell these apart,
# so this encoding is plain where that is unambiguous and tagged elsewhere:
#
#   length-1 character, logical, integer, or double without attributes and
#     not NA      -> a JSON string, boolean, or number. Doubles are always
#                    written with a decimal point or exponent, integers never,
#                    so a reader can tell them apart.
#   a list with unique, non-empty names that do not start with "$" and no
#     other attributes -> a JSON object, in name order.
#   an unnamed list without attributes -> a JSON array.
#   anything else -> an object with one value tag and optional "$names" and
#     "$attr" entries:
#       {"$chr": [...]}, {"$int": [...]}, {"$dbl": [...]}, {"$lgl": [...]},
#       {"$cplx": {"re": [...], "im": [...]}}, {"$raw": "<hex>"},
#       {"$list": [...]}, {"$null": true}, {"$sparse": {...}}.
#     NA is null. In "$dbl", NaN, Inf, -Inf, and negative zero are the strings
#     "NaN", "Inf", "-Inf", and "-0".
#
# Doubles are written with 17 significant digits, which round-trips every
# double exactly. Decoding an encoded value gives back an identical() value
# (sparse matrices: equal canonical bytes), so digests survive the round trip.

.fds_json_format <- "org.fmridataset.fds-json/v1"

.fds_json_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_schema", ...)
}

.fds_json_plain_names <- function(names) {
  !is.null(names) && !anyNA(names) && all(nzchar(names)) &&
    !anyDuplicated(names) && !any(startsWith(names, "$"))
}

.fds_json_encode <- function(x) {
  if (is.null(x)) {
    return(list(`$null` = TRUE))
  }
  if (isS4(x)) {
    if (methods::is(x, "sparseMatrix")) {
      return(.fds_json_encode_sparse(x))
    }
    .fds_json_abort(
      sprintf("FDS JSON cannot encode an S4 object of class '%s'.", class(x)[[1L]]),
      field = "value"
    )
  }
  if (!is.list(x) && !is.atomic(x)) {
    .fds_json_abort(
      sprintf("FDS JSON cannot encode a value of type '%s'.", typeof(x)),
      field = "value"
    )
  }
  attrs <- attributes(x)
  value_names <- attrs$names
  other <- attrs[setdiff(names(attrs), "names")]

  if (is.list(x)) {
    values <- lapply(unname(unclass(x)), .fds_json_encode)
    if (!length(other) && length(x) && .fds_json_plain_names(value_names)) {
      return(stats::setNames(values, value_names))
    }
    if (!length(other) && is.null(value_names)) {
      return(values)
    }
    out <- list(`$list` = values)
  } else {
    if (!length(attrs) && length(x) == 1L && !is.na(x) &&
      typeof(x) %in% c("character", "logical", "integer", "double") &&
      !(is.double(x) && (is.infinite(x) || identical(x, 0) && 1 / x < 0))) {
      return(x)
    }
    out <- .fds_json_encode_atomic(unname(unclass(x)))
  }
  if (!is.null(value_names)) out$`$names` <- .fds_json_encode_atomic(value_names)
  if (length(other)) out$`$attr` <- lapply(other, .fds_json_encode)
  out
}

.fds_json_encode_atomic <- function(x) {
  attributes(x) <- NULL
  switch(typeof(x),
    character = list(`$chr` = I(x)),
    logical = list(`$lgl` = I(x)),
    integer = list(`$int` = I(x)),
    double = list(`$dbl` = .fds_json_doubles(x)),
    complex = list(`$cplx` = list(re = .fds_json_doubles(Re(x)), im = .fds_json_doubles(Im(x)))),
    raw = list(`$raw` = paste(as.character(x), collapse = "")),
    .fds_json_abort(
      sprintf("FDS JSON cannot encode an atomic vector of type '%s'.", typeof(x)),
      field = "value"
    )
  )
}

# Plain numbers where JSON can carry them; strings for the IEEE values it
# cannot, and null for NA.
.fds_json_doubles <- function(x) {
  special <- is.nan(x) | is.infinite(x) | (!is.na(x) & x == 0 & 1 / x < 0) |
    (is.na(x) & !is.nan(x))
  if (!any(special)) {
    return(I(x))
  }
  lapply(seq_along(x), function(i) {
    value <- x[[i]]
    if (is.nan(value)) {
      "NaN"
    } else if (is.na(value)) {
      NULL
    } else if (is.infinite(value)) {
      if (value > 0) "Inf" else "-Inf"
    } else if (value == 0 && 1 / value < 0) {
      "-0"
    } else {
      value
    }
  })
}

.fds_json_encode_sparse <- function(x) {
  general <- methods::as(methods::as(x, "CsparseMatrix"), "generalMatrix")
  entries <- Matrix::summary(general)
  values <- if ("x" %in% names(entries)) as.double(entries$x) else rep(1, nrow(entries))
  out <- list(
    class = class(x)[[1L]],
    dim = I(as.integer(dim(x))),
    i = I(as.integer(entries$i)),
    j = I(as.integer(entries$j)),
    x = .fds_json_doubles(values)
  )
  if (!is.null(dimnames(x))) out$dimnames <- .fds_json_encode(dimnames(x))
  list(`$sparse` = out)
}

.fds_json_decode <- function(j) {
  if (is.null(j)) {
    return(NULL)
  }
  if (!is.list(j)) {
    # A plain scalar: jsonlite already gives the R type (integers have no
    # decimal point in FDS JSON).
    return(j)
  }
  keys <- names(j)
  if (is.null(keys)) {
    return(lapply(j, .fds_json_decode))
  }
  if (!length(keys) || !any(startsWith(keys, "$"))) {
    return(stats::setNames(lapply(j, .fds_json_decode), keys))
  }
  tag <- intersect(
    c("$chr", "$lgl", "$int", "$dbl", "$cplx", "$raw", "$list", "$null", "$sparse"),
    keys
  )
  if (length(tag) != 1L) {
    .fds_json_abort(
      sprintf("FDS JSON object has keys %s but no single value tag.", paste(keys, collapse = ", ")),
      field = "value"
    )
  }
  if (identical(tag, "$null")) {
    return(NULL)
  }
  if (identical(tag, "$sparse")) {
    return(.fds_json_decode_sparse(j[["$sparse"]]))
  }
  value <- switch(tag,
    `$chr` = .fds_json_vector(j[[tag]], NA_character_, as.character),
    `$lgl` = .fds_json_vector(j[[tag]], NA, as.logical),
    `$int` = .fds_json_vector(j[[tag]], NA_integer_, as.integer),
    `$dbl` = .fds_json_decode_doubles(j[[tag]]),
    `$cplx` = complex(
      real = .fds_json_decode_doubles(j[[tag]]$re),
      imaginary = .fds_json_decode_doubles(j[[tag]]$im)
    ),
    `$raw` = {
      hex <- j[[tag]]
      if (!nzchar(hex)) {
        raw()
      } else {
        starts <- seq.int(1L, nchar(hex), by = 2L)
        as.raw(strtoi(substring(hex, starts, starts + 1L), 16L))
      }
    },
    `$list` = lapply(j[[tag]], .fds_json_decode)
  )
  attrs <- list()
  if ("$names" %in% keys) attrs$names <- .fds_json_decode(j[["$names"]])
  if ("$attr" %in% keys) attrs <- c(attrs, lapply(j[["$attr"]], .fds_json_decode))
  if (length(attrs)) attributes(value) <- attrs
  value
}

.fds_json_vector <- function(values, missing, as_type) {
  if (!length(values)) {
    return(as_type(vector(mode = typeof(missing), length = 0L)))
  }
  as_type(vapply(values, function(value) if (is.null(value)) missing else as_type(value), missing))
}

.fds_json_decode_doubles <- function(values) {
  if (!length(values)) {
    return(double())
  }
  vapply(values, function(value) {
    if (is.null(value)) {
      return(NA_real_)
    }
    if (is.character(value)) {
      return(switch(value,
        `NaN` = NaN, `Inf` = Inf, `-Inf` = -Inf, `-0` = -0,
        .fds_json_abort(sprintf("FDS JSON double has unknown value '%s'.", value), field = "value")
      ))
    }
    as.double(value)
  }, double(1))
}

.fds_json_decode_sparse <- function(j) {
  dims <- as.integer(unlist(j$dim))
  out <- Matrix::sparseMatrix(
    i = as.integer(unlist(j$i)), j = as.integer(unlist(j$j)),
    x = .fds_json_decode_doubles(j$x), dims = dims,
    dimnames = if (is.null(j$dimnames)) NULL else .fds_json_decode(j$dimnames)
  )
  if (!identical(class(out)[[1L]], j$class)) {
    out <- tryCatch(methods::as(out, j$class), error = function(e) out)
  }
  out
}

# Write and read whole documents. `value` is encoded under "value" next to a
# format marker, so a reader knows which encoding rules apply.
.fds_json_write <- function(value, path, extra = list()) {
  document <- c(list(format = .fds_json_format), extra, list(value = .fds_json_encode(value)))
  text <- jsonlite::toJSON(
    document,
    auto_unbox = TRUE, digits = I(17), always_decimal = TRUE,
    na = "null", null = "null", pretty = TRUE
  )
  writeLines(text, path, useBytes = TRUE)
  invisible(path)
}

.fds_json_read <- function(path) {
  document <- jsonlite::read_json(path, simplifyVector = FALSE)
  if (!identical(document$format, .fds_json_format)) {
    .fds_json_abort(
      sprintf(
        "%s is not an FDS JSON document (format %s, expected %s).",
        basename(path), document$format %||% "missing", .fds_json_format
      ),
      field = "format"
    )
  }
  document$value <- .fds_json_decode(document$value)
  document
}
