# Compact printing for feature spaces, explanations, provenance, and records.
#
# Each method prints a `<class>` header line with the essentials, then a few
# indented fields, and returns its argument invisibly. Digests are shortened to
# their first 12 characters, as elsewhere in the package; the full values stay
# available from the accessors.

.short_digest <- function(x) substr(x, 1L, 12L)

# cat() with single spaces between parts and no trailing space.
.line <- function(...) cat(paste(...), "\n", sep = "")

.format_bytes <- function(bytes) {
  if (is.null(bytes) || !length(bytes) || is.na(bytes)) {
    return("unknown")
  }
  units <- c("B", "KB", "MB", "GB", "TB")
  power <- if (bytes < 1024) 0L else min(length(units) - 1L, floor(log(bytes, 1024)))
  value <- bytes / 1024^power
  if (power == 0L) {
    sprintf("%.0f B", value)
  } else {
    sprintf("%.1f %s", value, units[[power + 1L]])
  }
}

.format_id_sample <- function(ids, max = 4L) {
  if (!length(ids)) {
    return("(none)")
  }
  shown <- paste(utils::head(ids, max), collapse = ", ")
  if (length(ids) > max) paste0(shown, ", ... (", length(ids), " in all)") else shown
}

# Shorten labels for display. Long labels are usually paths or BIDS file
# names: keep the file name, then drop a suffix every label shares
# ("_space-..._bold"), then a shared prefix, cutting at separators, so what
# remains tells the labels apart: "sub-01_task-memory_run-01". A lone long
# label keeps its end.
.abbreviate_labels <- function(x, width = 40L) {
  if (!length(x) || all(nchar(x) <= width)) {
    return(x)
  }
  if (any(grepl("/", x, fixed = TRUE)) && !anyDuplicated(basename(x))) {
    x <- basename(x)
    if (all(nchar(x) <= width)) {
      return(x)
    }
  }
  if (length(unique(x)) < 2L) {
    long <- nchar(x) > width
    x[long] <- paste0("...", substring(x[long], nchar(x[long]) - width + 4L))
    return(x)
  }
  common_prefix <- function(values) {
    chars <- strsplit(values, "", fixed = TRUE)
    n <- min(lengths(chars))
    i <- 0L
    while (i < n && length(unique(vapply(chars, `[[`, "", i + 1L))) == 1L) i <- i + 1L
    substr(values[[1L]], 1L, i)
  }
  reverse <- function(values) {
    vapply(strsplit(values, "", fixed = TRUE), function(ch) paste(rev(ch), collapse = ""), "")
  }
  suffix <- reverse(common_prefix(reverse(x)))
  suffix <- sub("^[^/_]*", "", suffix)
  x <- substring(x, 1L, nchar(x) - nchar(suffix))
  if (all(nchar(x) <= width)) {
    return(x)
  }
  prefix <- common_prefix(x)
  prefix <- sub("[^/_]*$", "", prefix)
  middle <- substring(x, nchar(prefix) + 1L)
  out <- paste0(if (nzchar(prefix)) "...", middle)
  long <- nchar(out) > width
  out[long] <- paste0(substr(out[long], 1L, width - 3L), "...")
  out
}

# "a" or "an" before a number as it is read aloud: an 8, an 11, an 18,
# an 80, an 11000, but a 1, a 110, a 1800.
.article <- function(n) {
  digits <- format(n, scientific = FALSE, trim = TRUE)
  group <- nchar(digits) %% 3L
  if (group == 0L) group <- 3L
  leading <- substr(digits, 1L, group)
  if (startsWith(digits, "8") || leading %in% c("11", "18")) "an" else "a"
}

.plural <- function(n, singular, plural = paste0(singular, "s")) {
  paste(format(n, big.mark = ","), if (identical(as.numeric(n), 1)) singular else plural)
}

.space_parent_line <- function(parent) {
  sprintf("%s, %s", class(parent)[[1L]], .plural(n_features(parent), "feature"))
}

#' @export
print.volume_space <- function(x, ...) {
  .line(
    "<volume_space>", .plural(n_features(x), "voxel"), "in a",
    paste(x$dim, collapse = " x "), "grid"
  )
  .line("  template:", x$template %||% "(none)", "  units:", x$units %||% "(none)")
  .line("  digest:  ", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.surface_space <- function(x, ...) {
  hemispheres <- table(factor(x$hemisphere[x$support], levels = unique(x$hemisphere)))
  hemispheres <- hemispheres[hemispheres > 0L]
  counts <- if (length(hemispheres) == 1L) {
    paste(names(hemispheres), "hemisphere")
  } else {
    paste(names(hemispheres), as.integer(hemispheres), collapse = ", ")
  }
  .line(
    "<surface_space>", .plural(n_features(x), "vertex", "vertices"),
    paste0("(", counts, ")")
  )
  medial <- sum(x$medial_wall %||% FALSE)
  other <- length(x$vertex_ids) - n_features(x) - medial
  if (medial > 0L) {
    .line("  excluded:", .plural(medial, "medial-wall vertex", "medial-wall vertices"))
  }
  if (other > 0L) {
    .line("  excluded:", .plural(other, "vertex", "vertices"), "outside the support")
  }
  .line("  template:", x$template %||% "(none)")
  .line("  digest:  ", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.parcel_space <- function(x, ...) {
  atlas <- x$atlas$id
  .line(
    "<parcel_space>", .plural(n_features(x), "parcel"),
    if (!is.null(atlas)) sprintf("(atlas \"%s\")", atlas)
  )
  .line("  parent:     ", .space_parent_line(x$parent))
  .line("  aggregation:", x$aggregation %||% "mean")
  .line("  digest:     ", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.basis_space <- function(x, ...) {
  .line("<basis_space>", .plural(n_features(x), "component"), sprintf("(%s)", x$basis_type))
  .line("  parent: ", .space_parent_line(x$parent))
  encoder <- if (is.null(basis_analysis(x))) {
    "none (synthesis only)"
  } else if (isTRUE(x$projection$left_inverse_validated)) {
    "validated left inverse of the decoder"
  } else {
    "supplied"
  }
  .line("  encoder:", encoder)
  .line("  digest: ", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.index_space <- function(x, ...) {
  .line("<index_space>", .plural(n_features(x), "feature"))
  .line("  ids:   ", .format_id_sample(feature_ids(x)))
  .line("  digest:", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.composite_space <- function(x, ...) {
  parts <- composite_parts(x)
  .line(
    "<composite_space>", .plural(n_features(x), "feature"), "in",
    .plural(length(parts), "part")
  )
  width <- max(nchar(names(parts)))
  for (name in names(parts)) {
    .line(sprintf("  %-*s", width + 1L, paste0(name, ":")), .space_parent_line(parts[[name]]))
  }
  .line("  digest:", .short_digest(space_digest(x)))
  invisible(x)
}

#' @export
print.frame_explanation <- function(x, ...) {
  shape <- x$shape
  base <- vapply(x$selection, function(s) as.integer(s$axis_length), integer(1))
  is_view <- !identical(as.integer(shape), unname(base))
  .line(
    "<frame explanation>", shape[[1L]], "x", shape[[2L]],
    if (is_view) {
      sprintf(
        "view of %s %d x %d frame",
        .article(base[[1L]]), base[[1L]], base[[2L]]
      )
    } else {
      "frame"
    }
  )
  .line("  ids:    ", if (isTRUE(x$ids_durable)) "durable" else "not durable (ephemeral)")
  .line(
    "  space:  ", x$space$type, sprintf("(%s)", .plural(x$space$features, "feature")),
    .short_digest(x$space$digest)
  )
  .line("  assays:")
  width <- max(nchar(names(x$assays)))
  for (name in names(x$assays)) {
    a <- x$assays[[name]]
    .line(sprintf(
      "    %-*s  %s, %s, chunks %s, %s to read%s",
      width, name, a$source_type, a$dtype, paste(a$chunks, collapse = " x "),
      .format_bytes(a$realization_bytes),
      if (identical(name, x$active_assay)) "  (active)" else ""
    ))
  }
  counts <- x$counts
  annotations <- c(
    observation_blocks = counts$observation_blocks,
    feature_blocks = counts$feature_blocks,
    entities = counts$entities, relations = counts$relations, tables = counts$tables
  )
  present <- annotations[annotations > 0]
  if (length(present)) {
    .line("  annotations:", paste(present, gsub("_", " ", names(present)), collapse = ", "))
  }
  shown <- Filter(Negate(is.null), x$digests[c("schema", "semantic")])
  if (length(shown)) {
    .line("  digests:", paste(names(shown), vapply(shown, .short_digest, ""), collapse = "  "))
  }
  invisible(x)
}

#' @export
print.provenance_graph <- function(x, ...) {
  records <- provenance_records(x)
  tips <- provenance_tips(x)
  .line("<provenance_graph>", .plural(length(records), "record"))
  shown <- utils::tail(records, 10L)
  if (length(records) > length(shown)) {
    .line("  ...", length(records) - length(shown), "earlier records")
  }
  width <- max(c(0L, nchar(vapply(shown, function(r) r$operation, ""))))
  for (record in shown) {
    parents <- length(record$parents)
    notes <- c(
      if (parents) .plural(parents, "parent"),
      if (record$id %in% tips) "(tip)"
    )
    .line(sprintf(
      "  %s  %-*s  %s", .short_digest(record$id), width, record$operation,
      paste(notes, collapse = ", ")
    ))
  }
  invisible(x)
}

#' @export
print.surface_map <- function(x, ...) {
  present <- sum(!is.na(x$values))
  .line(
    "<surface_map>", .plural(length(x$values), "vertex", "vertices"),
    sprintf("(%d with values)", present)
  )
  .line("  template:", x$template %||% "(none)")
  .line("  range:   ", if (present) paste(format(range(x$values, na.rm = TRUE)), collapse = " to ") else "(no values)")
  invisible(x)
}

#' @export
print.composite_map <- function(x, ...) {
  .line("<composite_map>", .plural(length(x$parts), "part"))
  width <- max(nchar(x$part_names)) + 1L
  for (name in x$part_names) {
    .line(sprintf("  %-*s", width, paste0(name, ":")), class(x$parts[[name]])[[1L]])
  }
  invisible(x)
}

.strip_record_class <- function(x) {
  if (is.list(x)) {
    x <- lapply(unclass(x), .strip_record_class)
  }
  x
}

#' @export
print.unaligned_record <- function(x, ...) {
  if (!length(x)) {
    .line("<unaligned_record> (empty)")
    return(invisible(x))
  }
  .line("<unaligned_record>", .plural(length(x), "field"))
  utils::str(
    .strip_record_class(x),
    no.list = TRUE, give.attr = FALSE, comp.str = "$ ", indent.str = " "
  )
  invisible(x)
}

#' @export
print.fmri_identity <- function(x, ...) {
  .line("<fmri_identity>", x$domain, .short_digest(x$digest))
  .line("  canonicalization:", x$canonicalization$id %||% "(none)")
  invisible(x)
}

#' @export
print.frame_link <- function(x, ...) {
  .line("<frame_link>", x$source, "->", x$target, sprintf("(%s)", x$type))
  if (!is.null(x$map)) {
    .line(
      "  map:", .plural(nrow(x$map), "pair"),
      sprintf("(%s IDs -> %s IDs)", x$source_axis, x$target_axis)
    )
  }
  if (!is.null(x$operator)) .line("  operator:", class(x$operator)[[1L]])
  invisible(x)
}
