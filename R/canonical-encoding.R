.identity_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_identity", ...)
}

.canonical_utf8 <- function(x) {
  present <- !is.na(x)
  x[present] <- stringi::stri_trans_nfc(enc2utf8(x[present]))
  Encoding(x[present]) <- "UTF-8"
  x
}

.canonical_names <- function(x) {
  if (is.null(x)) return(NULL)
  .canonical_utf8(as.character(x))
}

.canonical_record_order <- function(x) {
  names_value <- names(x)
  if (is.null(names_value) || length(names_value) != length(x) ||
      anyNA(names_value) || any(!nzchar(names_value)) ||
      anyDuplicated(names_value)) {
    return(seq_along(x))
  }
  order(.canonical_names(names_value), method = "radix")
}

.canonical_attributes <- function(x) {
  values <- attributes(x)
  if (is.null(values)) return(NULL)
  names_value <- names(values)
  if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
      anyDuplicated(names_value)) {
    .identity_abort("Object attributes require unique non-empty names.",
                    field = "attributes")
  }
  names(values) <- .canonical_names(names_value)
  values <- values[order(names(values), method = "radix")]
  lapply(values, .canonical_normalize)
}

.canonical_sparse_matrix <- function(x) {
  compressed <- methods::as(x, "CsparseMatrix")
  entries <- Matrix::summary(compressed)
  if (nrow(entries)) entries <- entries[order(entries$j, entries$i), , drop = FALSE]
  list(
    .canonical_type = "sparse_matrix",
    value_type = typeof(entries$x),
    dimensions = as.integer(dim(compressed)),
    dimnames = lapply(dimnames(compressed), .canonical_names),
    i = as.integer(entries$i),
    j = as.integer(entries$j),
    x = .canonical_normalize(entries$x)
  )
}

.canonical_normalize <- function(x) {
  if (methods::is(x, "sparseMatrix")) {
    return(.canonical_normalize(.canonical_sparse_matrix(x)))
  }
  if (isS4(x)) {
    .identity_abort(
      "Canonical encoding supports sparse matrices but not arbitrary S4 objects.",
      field = "value"
    )
  }
  if (.source_contains_runtime_state(x)) {
    .identity_abort(
      "Canonical encoding does not accept functions, environments, external pointers, or runtime handles.",
      field = "value"
    )
  }
  if (is.null(x)) return(NULL)
  if (is.pairlist(x)) x <- as.list(x)
  if (is.list(x)) {
    attributes_value <- .canonical_attributes(x)
    # Traverse underlying list storage. Some list-based S3 classes implement
    # `[[` by returning another object of the same class (numeric_version is a
    # common example), which otherwise causes unbounded recursive dispatch.
    x <- unclass(x)
    order_value <- .canonical_record_order(x)
    x <- x[order_value]
    names_value <- .canonical_names(names(x))
    x <- lapply(x, .canonical_normalize)
    if (!is.null(attributes_value)) {
      if (!is.null(attributes_value$names)) attributes_value$names <- names_value
      attributes(x) <- attributes_value
    } else {
      names(x) <- names_value
    }
    return(x)
  }
  if (is.character(x)) x <- .canonical_utf8(x)
  if (is.double(x)) x[is.nan(x)] <- NaN
  attributes(x) <- .canonical_attributes(x)
  x
}

.canonical_int32 <- function(x) {
  writeBin(as.integer(x), raw(), size = 4L, endian = "big")
}

.canonical_length <- function(x) {
  if (length(x) != 1L || is.na(x) || x < 0 || x > .Machine$integer.max) {
    .identity_abort("Canonical values exceed the v1 32-bit length limit.",
                    field = "length")
  }
  .canonical_int32(x)
}

.canonical_tag <- function(x) charToRaw(x)

.canonical_string_bytes <- function(x) {
  if (is.na(x)) return(.canonical_tag("0"))
  value <- charToRaw(enc2utf8(x))
  c(.canonical_tag("1"), .canonical_length(length(value)), value)
}

.canonical_double_bytes <- function(x) {
  if (is.nan(x)) return(.canonical_tag("n"))
  if (is.na(x)) return(.canonical_tag("a"))
  if (is.infinite(x)) return(.canonical_tag(if (x > 0) "p" else "m"))
  c(.canonical_tag("f"), writeBin(x, raw(), size = 8L, endian = "big"))
}

.canonical_attribute_bytes <- function(x) {
  values <- attributes(x)
  if (is.null(values)) return(c(.canonical_tag("A"), .canonical_length(0L)))
  names_value <- names(values)
  if (is.null(names_value) || anyNA(names_value) || any(!nzchar(names_value)) ||
      anyDuplicated(names_value)) {
    .identity_abort("Object attributes require unique non-empty names.",
                    field = "attributes")
  }
  names_value <- .canonical_names(names_value)
  order_value <- order(names_value, method = "radix")
  values <- values[order_value]
  names_value <- names_value[order_value]
  c(
    .canonical_tag("A"), .canonical_length(length(values)),
    unlist(Map(function(name, value) {
      c(.canonical_string_bytes(name), .canonical_value_bytes(value))
    }, names_value, values), use.names = FALSE)
  )
}

.canonical_value_bytes <- function(x) {
  if (is.null(x)) return(.canonical_tag("N"))
  attrs <- .canonical_attribute_bytes(x)
  # Attributes are encoded independently above. Strip them before writing the
  # payload so classed atomic vectors such as POSIXct reach writeBin() as their
  # ordinary underlying vector storage.
  attributes(x) <- NULL
  value <- switch(
    typeof(x),
    logical = c(
      .canonical_tag("l"), .canonical_length(length(x)),
      as.raw(ifelse(is.na(x), 2L, as.integer(x)))
    ),
    integer = c(
      .canonical_tag("i"), .canonical_length(length(x)),
      unlist(lapply(x, .canonical_int32), use.names = FALSE)
    ),
    double = c(
      .canonical_tag("d"), .canonical_length(length(x)),
      unlist(lapply(x, .canonical_double_bytes), use.names = FALSE)
    ),
    complex = c(
      .canonical_tag("z"), .canonical_length(length(x)),
      unlist(lapply(x, function(value) {
        c(.canonical_double_bytes(Re(value)), .canonical_double_bytes(Im(value)))
      }), use.names = FALSE)
    ),
    character = c(
      .canonical_tag("c"), .canonical_length(length(x)),
      unlist(lapply(x, .canonical_string_bytes), use.names = FALSE)
    ),
    raw = c(.canonical_tag("r"), .canonical_length(length(x)), x),
    list = c(
      .canonical_tag("v"), .canonical_length(length(x)),
      unlist(lapply(x, .canonical_value_bytes), use.names = FALSE)
    ),
    .identity_abort(
      sprintf("Canonical encoding does not support R type '%s'.", typeof(x)),
      field = "value"
    )
  )
  c(value, attrs)
}

#' Encode and hash canonical R values
#'
#' Canonicalization version 1 normalizes semantic R values and writes a tagged,
#' length-prefixed package binary format. It is an explicitly R-only format.
#' Named lists and attributes use lexicographic field
#' order; unnamed list order is preserved. Strings are UTF-8 NFC, sparse matrix
#' storage layouts are normalized, and NaN payloads use one R value while
#' remaining distinct from `NA_real_`. Negative zero remains distinct from zero.
#'
#' @param x A serializable R value.
#' @return `canonical_bytes()` returns a raw vector. `canonical_sha256()` returns
#'   its lowercase SHA-256 hexadecimal digest.
#' @name canonical-encoding
NULL

#' @rdname canonical-encoding
#' @export
canonical_bytes <- function(x) {
  header <- charToRaw(paste0(.canonicalization_contract$id, "\n"))
  c(header, .canonical_value_bytes(.canonical_normalize(x)))
}

#' @rdname canonical-encoding
#' @export
canonical_sha256 <- function(x) {
  digest::digest(
    canonical_bytes(x), algo = .canonicalization_contract$algorithm,
    serialize = FALSE
  )
}
