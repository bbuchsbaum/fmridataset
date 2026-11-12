#' Frame metadata, typed tables, and provenance
#'
#' Accessors for the parts of a frame that are not aligned to an axis: its
#' unaligned metadata record, its typed tables (such as BIDS events), and its
#' provenance graph. Views return the values of their base frame.
#'
#' @param x An `fmri_frame` or view. `provenance()` also accepts an
#'   `fmri_collection` or `fmri_study`.
#' @param name One table name.
#' @param ... Unused.
#' @return `frame_metadata()` returns an `unaligned_record`; `frame_tables()`
#'   a named list of typed tables; `frame_table()` one typed table, whose rows
#'   [table_data()] returns; `provenance()` a `provenance_graph`, or `NULL`
#'   when none was recorded.
#' @examples
#' frame <- fmri_frame(
#'   assays = list(bold = matrix(0, 2, 3)),
#'   observations = data.frame(.obs_id = c("a", "b")),
#'   tables = list(events = event_table(
#'     data.frame(event_id = "e1", onset = 0, trial_type = "go")
#'   )),
#'   metadata = list(site = "A")
#' )
#' frame_metadata(frame)
#' names(frame_tables(frame))
#' table_data(frame_table(frame, "events"))
#' @name frame-annotations
NULL

#' @rdname frame-annotations
#' @export
frame_metadata <- function(x) {
  .assert_frame_like(x, "frame_metadata")
  .frame_container_value(x, "metadata") %||% unaligned_record()
}

#' @rdname frame-annotations
#' @export
frame_tables <- function(x) {
  .assert_frame_like(x, "frame_tables")
  .frame_container_value(x, "tables") %||% list()
}

#' @rdname frame-annotations
#' @export
frame_table <- function(x, name) {
  values <- frame_tables(x)
  if (!is.character(name) || length(name) != 1L || is.na(name) ||
    !name %in% names(values)) {
    .frame_abort(
      sprintf(
        "Unknown frame table %s; the frame has %s.",
        encodeString(as.character(name)[1L], quote = "\""),
        if (length(values)) paste0("\"", names(values), "\"", collapse = ", ") else "no tables"
      ),
      "fmridataset_error_alignment",
      table = name, available = names(values)
    )
  }
  values[[name]]
}

#' @rdname frame-annotations
#' @export
provenance <- function(x, ...) UseMethod("provenance")

#' @export
provenance.fmri_frame <- function(x, ...) .frame_container_value(x, "provenance")

#' @export
provenance.fmri_collection <- function(x, ...) x$provenance

#' @export
provenance.fmri_study <- function(x, ...) x$provenance

.assert_frame_like <- function(x, what) {
  if (!inherits(x, "fmri_frame")) {
    .frame_abort(
      sprintf("%s() requires an fmri_frame or view, not a %s.", what, class(x)[[1L]]),
      "fmridataset_error_alignment",
      actual = class(x)
    )
  }
  invisible(TRUE)
}
