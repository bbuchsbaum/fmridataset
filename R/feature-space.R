.canonical_digest <- function(x) {
  digest::digest(x, algo = "sha256", serialize = TRUE)
}

#' Feature-space contract
#'
#' @param x A feature-space object.
#' @param y A second feature-space object.
#' @param index Feature positions used to restrict a space.
#' @param spatial_object A native spatial object to vectorize.
#' @param vector A feature vector to reconstruct.
#' @param ... Additional arguments for methods.
#' @name feature-space
NULL

#' @rdname feature-space
#' @export
n_features <- function(x, ...) UseMethod("n_features")

#' @rdname feature-space
#' @export
feature_ids <- function(x, ...) UseMethod("feature_ids")

#' @rdname feature-space
#' @export
native_shape <- function(x, ...) UseMethod("native_shape")

#' @rdname feature-space
#' @export
feature_data <- function(x, ...) UseMethod("feature_data")

#' @rdname feature-space
#' @export
space_digest <- function(x, ...) UseMethod("space_digest")

#' @rdname feature-space
#' @export
restrict_space <- function(x, index, ...) UseMethod("restrict_space")

#' @rdname feature-space
#' @export
vectorize_space <- function(x, spatial_object, ...) UseMethod("vectorize_space")

#' @rdname feature-space
#' @export
reconstruct_space <- function(x, vector, ...) UseMethod("reconstruct_space")

#' @rdname feature-space
#' @export
adjacency <- function(x, ...) UseMethod("adjacency")

#' @rdname feature-space
#' @export
compatible_space <- function(x, y, ...) {
  same_class <- identical(class(x), class(y))
  same_digest <- identical(space_digest(x), space_digest(y))
  same_ids <- identical(feature_ids(x), feature_ids(y))
  ok <- same_class && same_digest && same_ids
  structure(
    list(
      compatible = ok,
      same_class = same_class,
      same_digest = same_digest,
      same_feature_ids = same_ids,
      x_digest = space_digest(x),
      y_digest = space_digest(y),
      reason = if (ok) NULL else "Feature spaces differ in type, digest, or IDs."
    ),
    class = "space_compatibility"
  )
}

#' @rdname feature-space
#' @export
assert_compatible_space <- function(x, y, ...) {
  report <- compatible_space(x, y, ...)
  if (!isTRUE(report$compatible)) {
    .frame_abort(
      report$reason,
      "fmridataset_error_space_mismatch",
      compatibility = report
    )
  }
  invisible(report)
}

#' Construct a generic indexed feature space
#'
#' @param n Number of features.
#' @param ids Optional stable feature IDs.
#' @param namespace Namespace used for generated IDs.
#' @param data Optional feature metadata.
#' @return An `index_space`.
#' @export
index_space <- function(n, ids = NULL, namespace = NULL, data = NULL) {
  n <- as.integer(n)
  if (length(n) != 1L || is.na(n) || n < 0L) {
    .frame_abort("n must be one non-negative integer.", "fmridataset_error_space_mismatch")
  }
  namespace <- namespace %||% uuid::UUIDgenerate()
  if (is.null(ids)) {
    ids <- sprintf("feature-%s-%06d", namespace, seq_len(n))
  }
  ids <- .validate_stable_ids(as.character(ids), "feature")
  if (length(ids) != n) {
    .frame_abort("Feature ID count does not equal n.", "fmridataset_error_space_mismatch")
  }
  if (is.null(data)) data <- data.frame(.feature_id = ids)
  data <- tibble::as_tibble(data)
  if (nrow(data) != n) {
    .frame_abort("Feature data must have n rows.", "fmridataset_error_space_mismatch")
  }
  data$.feature_id <- ids
  structure(
    list(
      n = n,
      ids = ids,
      namespace = namespace,
      data = data,
      schema_version = 1L
    ),
    class = c("index_space", "feature_space")
  )
}

#' @export
n_features.index_space <- function(x, ...) x$n
#' @export
feature_ids.index_space <- function(x, ...) x$ids
#' @export
native_shape.index_space <- function(x, ...) x$n
#' @export
feature_data.index_space <- function(x, ...) x$data
#' @export
space_digest.index_space <- function(x, ...) {
  .canonical_digest(list(
    type = "index_space",
    schema_version = x$schema_version,
    namespace = x$namespace,
    ids = x$ids
  ))
}
#' @export
restrict_space.index_space <- function(x, index, ...) {
  index_space(
    length(index),
    ids = x$ids[index],
    namespace = x$namespace,
    data = x$data[index, , drop = FALSE]
  )
}
#' @export
vectorize_space.index_space <- function(x, spatial_object, ...) {
  out <- as.numeric(spatial_object)
  if (length(out) != x$n) {
    .frame_abort("Object does not match index space.", "fmridataset_error_space_mismatch")
  }
  out
}
#' @export
reconstruct_space.index_space <- function(x, vector, ...) {
  if (length(vector) != x$n) {
    .frame_abort("Vector does not match index space.", "fmridataset_error_space_mismatch")
  }
  vector
}
#' @export
adjacency.index_space <- function(x, ...) NULL

#' Construct a packed volumetric feature space
#'
#' @param dim Three spatial dimensions.
#' @param affine A 4 by 4 voxel-to-world affine.
#' @param support Logical full-volume support or packed linear indices.
#' @param template Optional template/native-space identity.
#' @param units Spatial units.
#' @param metadata Additional serializable metadata.
#' @return A `volume_space`.
#' @export
volume_space <- function(dim, affine = diag(4), support = NULL,
                         template = NULL, units = "mm", metadata = list()) {
  dim <- as.integer(dim)
  if (length(dim) != 3L || anyNA(dim) || any(dim <= 0L)) {
    .frame_abort("dim must contain three positive integers.", "fmridataset_error_space_mismatch")
  }
  affine <- as.matrix(affine)
  storage.mode(affine) <- "double"
  if (!identical(dim(affine), c(4L, 4L)) || any(!is.finite(affine))) {
    .frame_abort("affine must be a finite 4 by 4 matrix.", "fmridataset_error_space_mismatch")
  }
  total <- prod(dim)
  if (is.null(support)) support <- seq_len(total)
  if (is.logical(support)) {
    if (length(support) != total || anyNA(support)) {
      .frame_abort("Logical support must cover the full volume.", "fmridataset_error_space_mismatch")
    }
    support <- which(support)
  }
  support <- as.integer(support)
  if (anyNA(support) || any(support < 1L | support > total) || anyDuplicated(support)) {
    .frame_abort("support contains invalid or duplicate indices.", "fmridataset_error_space_mismatch")
  }
  structure(
    list(
      dim = dim,
      affine = affine,
      support = support,
      template = template,
      units = units,
      metadata = metadata,
      schema_version = 1L
    ),
    class = c("volume_space", "feature_space")
  )
}

#' @export
n_features.volume_space <- function(x, ...) length(x$support)
#' @export
feature_ids.volume_space <- function(x, ...) paste0("voxel-", x$support)
#' @export
native_shape.volume_space <- function(x, ...) x$dim
#' @export
feature_data.volume_space <- function(x, ...) {
  ijk <- arrayInd(x$support, .dim = x$dim)
  tibble::tibble(
    .feature_id = feature_ids(x),
    .linear_index = x$support,
    i = ijk[, 1L],
    j = ijk[, 2L],
    k = ijk[, 3L]
  )
}
#' @export
space_digest.volume_space <- function(x, ...) {
  .canonical_digest(list(
    type = "volume_space",
    schema_version = x$schema_version,
    dim = x$dim,
    affine = unname(x$affine),
    support = x$support,
    template = x$template,
    units = x$units
  ))
}
#' @export
restrict_space.volume_space <- function(x, index, ...) {
  volume_space(
    dim = x$dim,
    affine = x$affine,
    support = x$support[index],
    template = x$template,
    units = x$units,
    metadata = x$metadata
  )
}
#' @export
vectorize_space.volume_space <- function(x, spatial_object, ...) {
  values <- if (methods::is(spatial_object, "NeuroVol")) {
    as.numeric(spatial_object)
  } else {
    as.numeric(spatial_object)
  }
  if (length(values) != prod(x$dim)) {
    .frame_abort("Spatial object does not match the native volume shape.", "fmridataset_error_space_mismatch")
  }
  values[x$support]
}
#' @export
reconstruct_space.volume_space <- function(x, vector, ...) {
  if (length(vector) != length(x$support)) {
    .frame_abort("Vector does not match the volume support.", "fmridataset_error_space_mismatch")
  }
  values <- rep(NA_real_, prod(x$dim))
  values[x$support] <- vector
  arr <- array(values, dim = x$dim)
  sp <- neuroim2::NeuroSpace(dim = x$dim, trans = x$affine)
  neuroim2::NeuroVol(arr, sp)
}
#' @export
adjacency.volume_space <- function(x, ...) {
  coords <- feature_data(x)[c("i", "j", "k")]
  if (!nrow(coords)) return(Matrix::Matrix(0, 0, 0, sparse = TRUE))
  key <- paste(coords$i, coords$j, coords$k, sep = ":")
  lookup <- stats::setNames(seq_along(key), key)
  from <- integer()
  to <- integer()
  shifts <- rbind(c(1, 0, 0), c(-1, 0, 0), c(0, 1, 0), c(0, -1, 0), c(0, 0, 1), c(0, 0, -1))
  for (s in seq_len(nrow(shifts))) {
    target <- sweep(as.matrix(coords), 2L, shifts[s, ], "+")
    hit <- unname(lookup[paste(target[, 1L], target[, 2L], target[, 3L], sep = ":")])
    ok <- !is.na(hit)
    from <- c(from, which(ok))
    to <- c(to, hit[ok])
  }
  Matrix::sparseMatrix(i = from, j = to, x = TRUE, dims = c(nrow(coords), nrow(coords)))
}
