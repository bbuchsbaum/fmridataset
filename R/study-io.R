#' Save and reopen a study as a BIDS derivative dataset
#'
#' `write_study()` saves an [fmri_study()] to a directory that is a BIDS
#' derivative dataset, and `open_study()` reopens it. BIDS supplies the
#' dataset frame: `dataset_description.json` (with `DatasetType`
#' `"derivative"` and `GeneratedBy`), and `participants.tsv` when the study has
#' a subject entity. The package adds only what BIDS has no place for, and lists
#' it in `.bidsignore` so BIDS tools skip it:
#'
#' * `study.json`: the study manifest (entities, tables, links, metadata, and
#'   provenance), with each frame referenced by its semantic digest. It uses the
#'   package's lossless JSON encoding, so reading it back gives the same
#'   manifest and the same [fds_study_manifest_digest()].
#' * `storage.json`: where each frame and study-level array is stored, and the
#'   package versions that wrote them. It is not part of the study's identity.
#' * `frames/`: one HDF5 file per frame, written with [write_frame()];
#'   collection members get one file each.
#' * `arrays.h5`: study-level arrays such as entity blocks, when there are any.
#'
#' The directory is written under a temporary name next to `path` and renamed
#' into place only when complete, so a failed write leaves no partial study.
#' `open_study()` reads frames lazily and checks every frame against the digest
#' recorded in `study.json`; a file that was replaced or rewritten raises an
#' error of class `fmridataset_error_source_stale`.
#'
#' Frames are stored with the `fmristore` package, which is not on CRAN;
#' install it from <https://bbuchsbaum.r-universe.dev>.
#'
#' @param x An `fmri_study`.
#' @param path Directory to create (`write_study()`) or open (`open_study()`).
#' @param name Dataset name for `dataset_description.json`; defaults to the
#'   directory name.
#' @param source_datasets Optional list describing the datasets the study was
#'   derived from, written as BIDS `SourceDatasets`: each element a list with
#'   any of `URL`, `DOI`, and `Version`.
#' @param dataset_links Optional named list mapping the dataset names used in
#'   BIDS URIs to their locations, written as BIDS `DatasetLinks`. Frames
#'   imported with [read_bids_bold()] record their inputs as `bids:raw:...`,
#'   so `list(raw = "/data/my-study")` (or a relative path or URL) says where
#'   `raw` is.
#' @param overwrite Replace an existing directory at `path`.
#' @return `write_study()` returns the normalized `path` invisibly.
#'   `open_study()` returns an `fmri_study` whose frames read lazily from disk.
#' @seealso [write_frame()] for single frames, [fds_study_manifest()].
#' @examples
#' if (requireNamespace("fmristore", quietly = TRUE)) {
#'   subjects <- entity_frame(
#'     data.frame(subject_id = c("sub-01", "sub-02"), age = c(31, 27)),
#'     key = "subject_id", entity_type = "subject"
#'   )
#'   frame <- fmri_frame(
#'     assays = list(bold = matrix(rnorm(8), 4, 2)),
#'     observations = data.frame(
#'       .obs_id = sprintf("vol-%d", 1:4),
#'       subject_id = rep(c("sub-01", "sub-02"), each = 2)
#'     ),
#'     space = index_space(2, ids = c("roi-1", "roi-2")),
#'     entities = list(subject = subjects),
#'     relations = list(volume_subject = key_relation("subject_id", target = "subject"))
#'   )
#'   study <- fmri_study(list(main = frame), entities = list(subject = subjects))
#'
#'   path <- write_study(study, file.path(tempdir(), "example-study"))
#'   list.files(path)
#'   reopened <- open_study(path)
#'   reopened
#'   unlink(path, recursive = TRUE)
#' }
#' @name study-io
NULL

.study_io_format <- "org.fmridataset.fds-study-dataset/v1"
.study_bids_version <- "1.10.0"

.study_io_abort <- function(message, ...) {
  .frame_abort(message, "fmridataset_error_backend_io", ...)
}

# A file name for an arbitrary representation or member name: its position
# keeps names unique, and the slug keeps the file recognisable.
.study_file_stem <- function(index, name) {
  sprintf("%02d-%s", index, gsub("[^A-Za-z0-9._-]+", "-", name))
}

#' @rdname study-io
#' @export
write_study <- function(x, path, name = NULL, source_datasets = NULL,
                        dataset_links = NULL, overwrite = FALSE) {
  if (!inherits(x, "fmri_study")) {
    .study_io_abort(
      sprintf("write_study() saves an fmri_study, not a %s.", class(x)[[1L]]),
      operation = "write_study", actual = class(x)
    )
  }
  validate_fmri_study(x)
  .assert_one_string(path, "path", .alignment_abort)
  if (!is.logical(overwrite) || length(overwrite) != 1L || is.na(overwrite)) {
    .study_io_abort("overwrite must be TRUE or FALSE.", operation = "write_study")
  }
  if (file.exists(path) && !overwrite) {
    .study_io_abort(
      sprintf("%s already exists; pass overwrite = TRUE to replace it.", path),
      operation = "write_study", path = path
    )
  }
  # Everything that can refuse the study (ephemeral IDs, invalid manifests)
  # runs before anything is written.
  manifest <- fds_study_manifest(x)
  representations <- fds_study_representations(x)
  bindings <- fds_study_bindings(x)
  .require_frame_store("write_frame_h5")

  parent <- dirname(path)
  if (!dir.exists(parent)) {
    .study_io_abort(sprintf("The parent directory %s does not exist.", parent), operation = "write_study")
  }
  staging <- file.path(parent, paste0(".", basename(path), ".partial-", substr(uuid::UUIDgenerate(), 1L, 8L)))
  dir.create(staging)
  committed <- FALSE
  on.exit(if (!committed) unlink(staging, recursive = TRUE), add = TRUE)
  dir.create(file.path(staging, "frames"))

  study_value <- manifest
  storage <- list(representations = list(), arrays = list())
  ids <- names(representations)
  for (i in seq_along(ids)) {
    id <- ids[[i]]
    value <- representations[[id]]
    stem <- .study_file_stem(i, id)
    entry <- manifest$representations[[id]]
    if (inherits(value, "fmri_collection")) {
      dir.create(file.path(staging, "frames", stem))
      members <- collection_frames(value)
      member_digests <- list()
      member_files <- list()
      for (k in seq_along(members)) {
        member <- names(members)[[k]]
        file <- file.path("frames", stem, paste0(.study_file_stem(k, member), ".h5"))
        write_frame(members[[member]], file.path(staging, file))
        member_digests[[member]] <- list(
          semantic_digest = fds_frame_manifest_digest(entry$members[[member]])
        )
        member_files[[member]] <- file
      }
      entry$members <- member_digests
      storage$representations[[id]] <- list(type = "fmri_collection", members = member_files)
    } else {
      file <- file.path("frames", paste0(stem, ".h5"))
      write_frame(value, file.path(staging, file))
      entry <- list(
        type = "fmri_frame",
        semantic_digest = fds_frame_manifest_digest(entry$manifest)
      )
      storage$representations[[id]] <- list(type = "fmri_frame", file = file)
    }
    study_value$representations[[id]] <- entry
  }

  if (length(bindings)) {
    .study_write_arrays(bindings, manifest$arrays, file.path(staging, "arrays.h5"))
    storage$arrays <- lapply(seq_along(bindings), function(k) {
      list(file = "arrays.h5", dataset = sprintf("arrays/a%03d", k))
    })
    names(storage$arrays) <- names(bindings)
  }

  .fds_json_write(study_value, file.path(staging, "study.json"), extra = list(kind = "fmri_study"))
  storage_document <- c(
    list(
      dataset_format = .study_io_format,
      study_digest = fds_study_manifest_digest(manifest),
      fmridataset_version = as.character(utils::packageVersion("fmridataset")),
      fmristore_version = as.character(utils::packageVersion("fmristore"))
    ),
    storage
  )
  .fds_json_write(storage_document, file.path(staging, "storage.json"), extra = list(kind = "storage"))
  .study_write_bids_files(x, staging, name %||% basename(path), source_datasets, dataset_links)

  if (file.exists(path)) {
    backup <- paste0(staging, "-previous")
    if (!file.rename(path, backup)) {
      .study_io_abort(sprintf("Could not move the existing %s aside.", path), operation = "write_study")
    }
    if (!file.rename(staging, path)) {
      file.rename(backup, path)
      .study_io_abort(sprintf("Could not move the new study into %s.", path), operation = "write_study")
    }
    unlink(backup, recursive = TRUE)
  } else if (!file.rename(staging, path)) {
    .study_io_abort(sprintf("Could not move the new study into %s.", path), operation = "write_study")
  }
  committed <- TRUE
  invisible(normalizePath(path, winslash = "/", mustWork = TRUE))
}

.study_write_arrays <- function(bindings, descriptors, file) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    .study_io_abort("Saving study-level arrays requires the hdf5r package.", operation = "write_study")
  }
  h5 <- hdf5r::H5File$new(file, mode = "w")
  on.exit(h5$close_all(), add = TRUE)
  h5$create_group("arrays")
  for (k in seq_along(bindings)) {
    values <- bindings[[k]]
    values <- if (inherits(values, "array_source")) source_read(values) else as.matrix(values)
    # Stored as numbers; the declared dtype is restored when the study is
    # opened, so the file never decides a binding's type.
    storage.mode(values) <- if (is.logical(values)) "integer" else storage.mode(values)
    h5[[sprintf("arrays/a%03d", k)]] <- unname(values)
  }
  invisible(file)
}

.study_participant_labels <- function(ids) {
  labels <- sub("^sub-", "", ids)
  if (anyNA(labels) || !all(grepl("^[A-Za-z0-9]+$", labels)) || anyDuplicated(labels)) {
    return(NULL)
  }
  paste0("sub-", labels)
}

.study_write_bids_files <- function(x, dir, name, source_datasets, dataset_links) {
  description <- list(
    Name = name,
    BIDSVersion = .study_bids_version,
    DatasetType = "derivative",
    GeneratedBy = list(list(
      Name = "fmridataset",
      Version = as.character(utils::packageVersion("fmridataset")),
      CodeURL = "https://github.com/bbuchsbaum/fmridataset"
    ))
  )
  if (!is.null(source_datasets)) description$SourceDatasets <- source_datasets
  if (!is.null(dataset_links)) {
    if (!is.list(dataset_links) || !.fds_json_plain_names(names(dataset_links)) ||
      "" %in% names(dataset_links) ||
      !all(vapply(dataset_links, .is_one_string, logical(1)))) {
      .study_io_abort(
        "dataset_links must be a named list of single strings, as in list(raw = \"/data/study\").",
        operation = "write_study"
      )
    }
    description$DatasetLinks <- dataset_links
  }
  writeLines(
    jsonlite::toJSON(description, auto_unbox = TRUE, pretty = TRUE),
    file.path(dir, "dataset_description.json")
  )
  writeLines(
    c("study.json", "storage.json", "frames/", "arrays.h5"),
    file.path(dir, ".bidsignore")
  )
  # participants.tsv is a view for BIDS tools; study.json stays authoritative.
  # It is written when a subject entity has IDs that are valid BIDS labels.
  shared <- entities(x)
  subject_name <- NULL
  for (entity_name in entity_names(shared)) {
    if (identical(shared[[entity_name]]$entity_type, "subject") || identical(entity_name, "subject")) {
      subject_name <- entity_name
      break
    }
  }
  if (is.null(subject_name)) {
    return(invisible(dir))
  }
  subject <- shared[[subject_name]]
  labels <- .study_participant_labels(entity_ids(subject))
  if (is.null(labels)) {
    return(invisible(dir))
  }
  data <- as.data.frame(entity_data(subject), stringsAsFactors = FALSE)
  data[[entity_key(subject)]] <- NULL
  scalar <- vapply(data, function(column) is.atomic(column) && is.null(dim(column)), logical(1))
  table <- cbind(data.frame(participant_id = labels, stringsAsFactors = FALSE), data[scalar])
  utils::write.table(
    table, file.path(dir, "participants.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, na = "n/a"
  )
  invisible(dir)
}

#' @rdname study-io
#' @export
open_study <- function(path) {
  .assert_one_string(path, "path", .alignment_abort)
  study_file <- file.path(path, "study.json")
  storage_file <- file.path(path, "storage.json")
  if (!file.exists(study_file) || !file.exists(storage_file)) {
    .study_io_abort(
      sprintf("%s is not a saved study: study.json or storage.json is missing.", path),
      operation = "open_study", path = path
    )
  }
  study_document <- .fds_json_read(study_file)
  storage <- .fds_json_read(storage_file)$value
  if (!identical(storage$dataset_format, .study_io_format)) {
    .study_io_abort(
      sprintf(
        "%s was written in study format %s; this version reads %s.",
        path, storage$dataset_format %||% "unknown", .study_io_format
      ),
      operation = "open_study"
    )
  }
  .require_frame_store("open_frame_h5")
  manifest <- study_document$value
  representations <- list()
  for (id in names(manifest$representations)) {
    entry <- manifest$representations[[id]]
    stored <- storage$representations[[id]]
    if (identical(entry$type, "fmri_collection")) {
      members <- list()
      member_manifests <- list()
      for (member in names(entry$members)) {
        frame <- .study_open_frame(path, stored$members[[member]], entry$members[[member]]$semantic_digest)
        members[[member]] <- frame
        member_manifests[[member]] <- fds_frame_manifest(frame)
      }
      representations[[id]] <- fmri_collection(members, metadata = entry$metadata, provenance = entry$provenance)
      manifest$representations[[id]]$members <- member_manifests
    } else {
      frame <- .study_open_frame(path, stored$file, entry$semantic_digest)
      representations[[id]] <- frame
      manifest$representations[[id]] <- list(type = "fmri_frame", manifest = fds_frame_manifest(frame))
    }
  }
  if (!identical(fds_study_manifest_digest(manifest), storage$study_digest)) {
    stop(source_error(
      sprintf("The study in %s does not match the digest recorded when it was written.", path),
      type = "stale", source = path,
      expected = storage$study_digest, actual = fds_study_manifest_digest(manifest)
    ))
  }
  bindings <- .study_open_arrays(path, storage$arrays, manifest$arrays)
  study_from_fds_manifest(manifest, representations, bindings)
}

.study_open_frame <- function(path, file, expected) {
  full <- file.path(path, file)
  if (is.null(file) || !file.exists(full)) {
    .study_io_abort(
      sprintf("The saved study is missing the frame file %s.", file %||% "(unrecorded)"),
      operation = "open_study", path = path
    )
  }
  frame <- open_frame(full)
  actual <- fds_frame_manifest_digest(frame)
  if (!identical(actual, expected)) {
    stop(source_error(
      sprintf("%s is not the frame this study was saved with.", file),
      type = "stale", source = full, expected = expected, actual = actual
    ))
  }
  frame
}

.study_open_arrays <- function(path, stored, descriptors) {
  if (!length(descriptors)) {
    return(list())
  }
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    .study_io_abort("Opening study-level arrays requires the hdf5r package.", operation = "open_study")
  }
  out <- lapply(names(descriptors), function(key) {
    location <- stored[[key]]
    h5 <- hdf5r::H5File$new(file.path(path, location$file), mode = "r")
    on.exit(h5$close_all(), add = TRUE)
    values <- h5[[location$dataset]]$read()
    values <- matrix(values, nrow = descriptors[[key]]$shape[[1L]], ncol = descriptors[[key]]$shape[[2L]])
    dtype <- descriptors[[key]]$dtype
    storage.mode(values) <- .realized_dtype_mode(dtype)
    memory_source(values, dtype = dtype)
  })
  names(out) <- names(descriptors)
  out
}
