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

.surface_asset <- function(x, type, n_vertices) {
  if (is.null(x)) {
    return(list(reference = NULL, digest = .canonical_digest(NULL), data = NULL))
  }
  if (is.matrix(x)) {
    data <- x
    reference <- NULL
    supplied_digest <- NULL
  } else if (is.list(x)) {
    data <- x$data %||% x[[if (type == "topology") "faces" else "coordinates"]]
    reference <- x$reference %||% NULL
    supplied_digest <- x$digest %||% NULL
    if (is.null(supplied_digest) && is.null(data)) {
      .frame_abort(paste(type, "reference requires a digest."), "fmridataset_error_space_mismatch")
    }
  } else {
    .frame_abort(paste(type, "must be a matrix or asset descriptor."), "fmridataset_error_space_mismatch")
  }
  if (!is.null(data)) {
    if (type == "topology") {
      data <- as.matrix(data)
      if (ncol(data) != 3L || anyNA(data) || any(data != as.integer(data)) ||
          any(data < 1L | data > n_vertices) ||
          any(apply(data, 1L, anyDuplicated) > 0L)) {
        .frame_abort("topology must contain valid non-degenerate vertex triangles.", "fmridataset_error_space_mismatch")
      }
      storage.mode(data) <- "integer"
    } else {
      data <- as.matrix(data)
      storage.mode(data) <- "double"
      if (!identical(dim(data), c(n_vertices, 3L)) || any(!is.finite(data))) {
        .frame_abort("geometry must contain one finite xyz row per vertex.", "fmridataset_error_space_mismatch")
      }
    }
  }
  computed_digest <- if (!is.null(data)) .canonical_digest(data) else NULL
  digest <- supplied_digest %||% computed_digest
  if (!is.null(supplied_digest) && !is.null(computed_digest) &&
      !identical(supplied_digest, computed_digest)) {
    .frame_abort(paste(type, "digest does not match its data."), "fmridataset_error_space_mismatch")
  }
  if (!is.character(digest) || length(digest) != 1L ||
      !grepl("^[0-9a-f]{64}$", digest)) {
    .frame_abort(paste(type, "digest must be one SHA-256 string."), "fmridataset_error_space_mismatch")
  }
  list(reference = reference, digest = digest, data = data)
}

#' Construct a packed cortical surface feature space
#'
#' @param vertex_ids Stable IDs for every vertex in the full mesh.
#' @param hemisphere One `"left"` or `"right"` label per full-mesh vertex.
#' @param support Active vertex positions or IDs. By default, all non-medial-wall
#'   vertices are active.
#' @param topology A three-column face matrix or asset descriptor with
#'   `reference`, `digest`, and optional `data`/`faces`.
#' @param geometry A vertex-by-three coordinate matrix or asset descriptor with
#'   `reference`, `digest`, and optional `data`/`coordinates`.
#' @param medial_wall Logical full-mesh medial-wall mask.
#' @param template Optional template identity such as `"fsLR-32k"`.
#' @param units Coordinate units.
#' @param surf_to_world A finite 4 by 4 surface-to-world transform, following
#'   the `neurosurf::SurfaceGeometry` convention.
#' @param metadata Additional serializable metadata.
#' @return A `surface_space`.
#' @export
surface_space <- function(vertex_ids, hemisphere, support = NULL,
                          topology = NULL, geometry = NULL,
                          medial_wall = NULL, template = NULL,
                          units = "mm", surf_to_world = diag(4),
                          metadata = list()) {
  vertex_ids <- .validate_stable_ids(as.character(vertex_ids), "vertex")
  n <- length(vertex_ids)
  hemisphere <- as.character(hemisphere)
  if (length(hemisphere) != n || anyNA(hemisphere) ||
      any(!hemisphere %in% c("left", "right"))) {
    .frame_abort("hemisphere must label every vertex as left or right.", "fmridataset_error_space_mismatch")
  }
  if (is.null(medial_wall)) medial_wall <- rep(FALSE, n)
  if (!is.logical(medial_wall) || length(medial_wall) != n || anyNA(medial_wall)) {
    .frame_abort("medial_wall must be one complete logical value per vertex.", "fmridataset_error_space_mismatch")
  }
  if (is.null(support)) support <- which(!medial_wall)
  if (is.character(support)) support <- match(support, vertex_ids)
  if (is.logical(support)) {
    if (length(support) != n || anyNA(support)) {
      .frame_abort("Logical surface support must span the full mesh.", "fmridataset_error_space_mismatch")
    }
    support <- which(support)
  }
  support <- as.integer(support)
  if (anyNA(support) || any(support < 1L | support > n) || anyDuplicated(support)) {
    .frame_abort("surface support contains invalid or duplicate vertices.", "fmridataset_error_space_mismatch")
  }
  if (any(medial_wall[support])) {
    .frame_abort("surface support cannot include medial-wall vertices.", "fmridataset_error_space_mismatch")
  }
  surf_to_world <- as.matrix(surf_to_world)
  storage.mode(surf_to_world) <- "double"
  if (!identical(dim(surf_to_world), c(4L, 4L)) ||
      any(!is.finite(surf_to_world))) {
    .frame_abort("surf_to_world must be a finite 4 by 4 matrix.",
                 "fmridataset_error_space_mismatch")
  }
  structure(
    list(
      vertex_ids = vertex_ids,
      hemisphere = hemisphere,
      support = support,
      medial_wall = medial_wall,
      topology = .surface_asset(topology, "topology", n),
      geometry = .surface_asset(geometry, "geometry", n),
      template = template,
      units = units,
      surf_to_world = surf_to_world,
      metadata = metadata,
      schema_version = 2L
    ),
    class = c("surface_space", "feature_space")
  )
}

#' @export
n_features.surface_space <- function(x, ...) length(x$support)
#' @export
feature_ids.surface_space <- function(x, ...) x$vertex_ids[x$support]
#' @export
native_shape.surface_space <- function(x, ...) c(vertex = length(x$vertex_ids))
#' @export
feature_data.surface_space <- function(x, ...) {
  hemi_index <- stats::ave(seq_along(x$vertex_ids), x$hemisphere,
                           FUN = seq_along)
  tibble::tibble(
    .feature_id = feature_ids(x),
    .vertex_index = x$support,
    vertex_id = x$vertex_ids[x$support],
    hemisphere = x$hemisphere[x$support],
    .hemisphere_vertex_index = as.integer(hemi_index[x$support]),
    medial_wall = x$medial_wall[x$support]
  )
}
#' @export
space_digest.surface_space <- function(x, ...) {
  .canonical_digest(list(
    type = "surface_space",
    schema_version = x$schema_version,
    vertex_ids = x$vertex_ids,
    hemisphere = x$hemisphere,
    support = x$support,
    medial_wall = x$medial_wall,
    topology_digest = x$topology$digest,
    geometry_digest = x$geometry$digest,
    template = x$template,
    units = x$units,
    surf_to_world = unname(x$surf_to_world %||% diag(4))
  ))
}
#' @export
restrict_space.surface_space <- function(x, index, ...) {
  surface_space(
    vertex_ids = x$vertex_ids,
    hemisphere = x$hemisphere,
    support = x$support[index],
    topology = x$topology,
    geometry = x$geometry,
    medial_wall = x$medial_wall,
    template = x$template,
    units = x$units,
    surf_to_world = x$surf_to_world %||% diag(4),
    metadata = x$metadata
  )
}
#' @export
vectorize_space.surface_space <- function(x, spatial_object, ...) {
  if (inherits(spatial_object, "surface_map")) {
    if (!identical(spatial_object$vertex_ids, x$vertex_ids)) {
      .frame_abort("Surface map vertex IDs do not match the space.", "fmridataset_error_space_mismatch")
    }
    values <- spatial_object$values
  } else {
    values <- spatial_object
    if (!is.null(names(values))) values <- values[x$vertex_ids]
  }
  values <- as.numeric(values)
  if (length(values) != length(x$vertex_ids)) {
    .frame_abort("Surface object does not match the full mesh.", "fmridataset_error_space_mismatch")
  }
  values[x$support]
}
#' @rdname feature-space
#' @param format Surface reconstruction format. The backend-neutral default is
#'   `"surface_map"`; `"neurosurf"` returns a `neurosurf::NeuroSurface` when
#'   embedded unilateral geometry is available.
#' @export
reconstruct_space.surface_space <- function(x, vector,
                                            format = c("surface_map", "neurosurf"),
                                            ...) {
  format <- match.arg(format)
  if (length(vector) != length(x$support)) {
    .frame_abort("Vector does not match surface support.", "fmridataset_error_space_mismatch")
  }
  if (identical(format, "neurosurf")) {
    if (!requireNamespace("neurosurf", quietly = TRUE)) {
      .frame_abort("format = 'neurosurf' requires the neurosurf package.",
                   "fmridataset_error_space_mismatch")
    }
    if (is.null(x$geometry$data) || is.null(x$topology$data)) {
      .frame_abort(
        "A neurosurf reconstruction requires embedded geometry and topology.",
        "fmridataset_error_space_mismatch"
      )
    }
    hemis <- unique(x$hemisphere)
    if (length(hemis) != 1L) {
      .frame_abort(
        "A single neurosurf NeuroSurface cannot represent both hemispheres.",
        "fmridataset_error_space_mismatch"
      )
    }
    hemi <- if (identical(hemis, "left")) "lh" else "rh"
    geom <- neurosurf::SurfaceGeometry(
      x$geometry$data,
      x$topology$data - 1L,
      hemi = hemi,
      label = x$metadata$surface_label %||% NA_character_,
      surf_to_world = x$surf_to_world
    )
    return(neurosurf::NeuroSurface(geom, x$support, as.numeric(vector)))
  }
  values <- rep(NA_real_, length(x$vertex_ids))
  values[x$support] <- vector
  structure(
    list(
      values = values,
      vertex_ids = x$vertex_ids,
      hemisphere = x$hemisphere,
      template = x$template,
      space_digest = space_digest(x)
    ),
    class = "surface_map"
  )
}
#' @export
adjacency.surface_space <- function(x, ...) {
  faces <- x$topology$data
  if (is.null(faces)) return(NULL)
  edges <- rbind(faces[, c(1L, 2L), drop = FALSE],
                 faces[, c(2L, 3L), drop = FALSE],
                 faces[, c(3L, 1L), drop = FALSE])
  lookup <- integer(length(x$vertex_ids))
  lookup[x$support] <- seq_along(x$support)
  from <- lookup[edges[, 1L]]
  to <- lookup[edges[, 2L]]
  keep <- from > 0L & to > 0L
  Matrix::sparseMatrix(
    i = c(from[keep], to[keep]),
    j = c(to[keep], from[keep]),
    x = TRUE,
    dims = rep(length(x$support), 2L),
    use.last.ij = TRUE
  )
}

#' Adapt a neurosurf geometry to a surface feature space
#'
#' `neurosurf` remains the owner of mesh geometry and algorithms. This adapter
#' extracts its stable topology, coordinates, hemisphere, and world transform
#' into the backend-neutral identity required by an `fmri_frame`.
#'
#' @param geometry A `neurosurf::SurfaceGeometry`.
#' @param vertex_ids Optional stable full-mesh vertex IDs.
#' @param support,medial_wall,template,units,metadata Passed to [surface_space()].
#' @return A `surface_space`.
#' @export
surface_space_from_neurosurf <- function(geometry, vertex_ids = NULL,
                                         support = NULL,
                                         medial_wall = NULL,
                                         template = NULL, units = "mm",
                                         metadata = list()) {
  if (!requireNamespace("neurosurf", quietly = TRUE)) {
    .frame_abort("surface_space_from_neurosurf() requires neurosurf.",
                 "fmridataset_error_space_mismatch")
  }
  if (!methods::is(geometry, "SurfaceGeometry")) {
    .frame_abort("geometry must be a neurosurf SurfaceGeometry.",
                 "fmridataset_error_space_mismatch")
  }
  nodes <- neurosurf::nodes(geometry)
  hemi_raw <- methods::slot(geometry, "hemi")
  hemi <- switch(
    tolower(hemi_raw[[1L]]),
    lh =, left = "left",
    rh =, right = "right",
    .frame_abort("neurosurf hemisphere must be lh/rh or left/right.",
                 "fmridataset_error_space_mismatch")
  )
  if (is.null(vertex_ids)) {
    prefix <- if (identical(hemi, "left")) "L" else "R"
    vertex_ids <- paste0(prefix, "-", nodes)
  }
  metadata$surface_label <- methods::slot(geometry, "label")
  surface_space(
    vertex_ids = vertex_ids,
    hemisphere = rep(hemi, length(nodes)),
    support = support,
    topology = neurosurf::faces(geometry),
    geometry = neurosurf::vertices(geometry, nodes),
    medial_wall = medial_wall,
    template = template,
    units = units,
    surf_to_world = methods::slot(geometry, "surf_to_world"),
    metadata = metadata
  )
}

.sparse_operator <- function(x, dims, what) {
  if (!(is.matrix(x) || methods::is(x, "Matrix"))) {
    .frame_abort(paste(what, "must be a matrix or Matrix."),
                 "fmridataset_error_space_mismatch")
  }
  if (!identical(as.integer(dim(x)), as.integer(dims))) {
    .frame_abort(
      paste(what, "dimensions must match the parent features and parcels."),
      "fmridataset_error_space_mismatch"
    )
  }
  entries <- Matrix::summary(Matrix::Matrix(x, sparse = TRUE))
  values <- if ("x" %in% names(entries)) entries$x else rep(1, nrow(entries))
  if (anyNA(values) || any(!is.finite(values)) || any(values < 0)) {
    .frame_abort(paste(what, "must contain finite non-negative weights."),
                 "fmridataset_error_space_mismatch")
  }
  Matrix::sparseMatrix(
    i = entries$i,
    j = entries$j,
    x = as.numeric(values),
    dims = as.integer(dims)
  )
}

.operator_digest_payload <- function(x) {
  entries <- Matrix::summary(x)
  list(
    dim = as.integer(dim(x)),
    i = as.integer(entries$i),
    j = as.integer(entries$j),
    x = as.numeric(entries$x)
  )
}

.normalize_atlas_identity <- function(atlas, n_parcels) {
  if (is.character(atlas) && length(atlas) == 1L) atlas <- list(id = atlas)
  if (!is.list(atlas) || !is.character(atlas$id) ||
      length(atlas$id) != 1L || is.na(atlas$id) || !nzchar(atlas$id)) {
    .frame_abort("atlas$id must be one non-empty character value.",
                 "fmridataset_error_space_mismatch")
  }
  if (!is.null(atlas$n_parcels) &&
      as.integer(atlas$n_parcels) < as.integer(n_parcels)) {
    .frame_abort("atlas$n_parcels cannot be smaller than the parcel axis.",
                 "fmridataset_error_space_mismatch")
  }
  if (is.null(atlas$n_parcels)) atlas$n_parcels <- as.integer(n_parcels)
  atlas
}

#' Construct a parent-linked parcel feature space
#'
#' A `parcel_space` owns only the feature-axis algebra required by
#' `fmridataset`: stable parcel identity, a parent feature space, and explicit
#' parent-to-parcel operators. Atlas discovery, labels, and provenance remain
#' owned by `neuroatlas` and can be imported with [parcel_space_from_atlas()].
#'
#' @param parent Parent `feature_space` (usually a volume or surface space).
#' @param parcel_ids Stable atlas-native parcel identifiers.
#' @param membership Non-negative parent-feature by parcel membership weights.
#' @param data One metadata row per parcel. The `id`, `label`, and `hemi`
#'   conventions match `neuroatlas::parcel_data` when present.
#' @param atlas Atlas identity list containing at least `id`, or a scalar ID.
#' @param aggregation Either weighted `"mean"` or weighted `"sum"`.
#' @param decoder Optional parent-feature by parcel reconstruction operator.
#'   The default blends overlapping parcel values by row-normalized membership.
#' @param metadata Additional serializable metadata.
#' @return A `parcel_space`.
#' @export
parcel_space <- function(parent, parcel_ids, membership, data = NULL,
                         atlas, aggregation = c("mean", "sum"),
                         decoder = NULL, metadata = list()) {
  if (!inherits(parent, "feature_space")) {
    .frame_abort("parent must be a feature_space.",
                 "fmridataset_error_space_mismatch")
  }
  aggregation <- match.arg(aggregation)
  parcel_ids <- .validate_stable_ids(as.character(parcel_ids), "parcel")
  k <- length(parcel_ids)
  atlas <- .normalize_atlas_identity(atlas, k)
  membership <- .sparse_operator(
    membership, c(n_features(parent), k), "membership"
  )
  totals <- as.numeric(Matrix::colSums(membership))
  if (any(totals <= 0)) {
    .frame_abort("Every parcel must contain at least one parent feature.",
                 "fmridataset_error_space_mismatch")
  }
  if (is.null(data)) data <- data.frame(id = parcel_ids)
  data <- tibble::as_tibble(data)
  if (nrow(data) != k) {
    .frame_abort("Parcel data must contain one row per parcel.",
                 "fmridataset_error_space_mismatch")
  }
  ids <- paste0(atlas$id, ":", parcel_ids)
  if (".feature_id" %in% names(data) &&
      !identical(as.character(data$.feature_id), ids)) {
    .frame_abort("Parcel data .feature_id values do not match atlas parcel IDs.",
                 "fmridataset_error_space_mismatch")
  }
  data$.feature_id <- ids
  data <- data[c(".feature_id", setdiff(names(data), ".feature_id"))]

  if (is.null(decoder)) {
    row_totals <- as.numeric(Matrix::rowSums(membership))
    scale <- ifelse(row_totals > 0, 1 / row_totals, 0)
    decoder <- Matrix::Diagonal(x = scale) %*% membership
  } else {
    decoder <- .sparse_operator(
      decoder, c(n_features(parent), k), "decoder"
    )
  }
  aggregation_operator <- Matrix::t(membership)
  if (identical(aggregation, "mean")) {
    aggregation_operator <- Matrix::Diagonal(x = 1 / totals) %*%
      aggregation_operator
  }

  structure(
    list(
      parent = parent,
      parcel_ids = parcel_ids,
      ids = ids,
      membership = membership,
      aggregation_operator = aggregation_operator,
      decoder = decoder,
      data = data,
      atlas = atlas,
      aggregation = aggregation,
      metadata = metadata,
      schema_version = 1L
    ),
    class = c("parcel_space", "feature_space")
  )
}

#' Inspect parcel-space operators
#'
#' @param x A `parcel_space`.
#' @return `parent_space()` returns its parent feature space;
#'   `parcel_membership()` and `parcel_aggregation()` return sparse operators.
#' @name parcel-operators
NULL

#' @rdname parcel-operators
#' @export
parent_space <- function(x) {
  if (!inherits(x, "parcel_space")) stop("x must be a parcel_space.", call. = FALSE)
  x$parent
}

#' @rdname parcel-operators
#' @export
parcel_membership <- function(x) {
  if (!inherits(x, "parcel_space")) stop("x must be a parcel_space.", call. = FALSE)
  x$membership
}

#' @rdname parcel-operators
#' @export
parcel_aggregation <- function(x) {
  if (!inherits(x, "parcel_space")) stop("x must be a parcel_space.", call. = FALSE)
  x$aggregation_operator
}

#' @export
n_features.parcel_space <- function(x, ...) length(x$parcel_ids)
#' @export
feature_ids.parcel_space <- function(x, ...) x$ids
#' @export
native_shape.parcel_space <- function(x, ...) c(parcel = length(x$parcel_ids))
#' @export
feature_data.parcel_space <- function(x, ...) x$data
#' @export
space_digest.parcel_space <- function(x, ...) {
  .canonical_digest(list(
    type = "parcel_space",
    schema_version = x$schema_version,
    parent_digest = space_digest(x$parent),
    parcel_ids = x$parcel_ids,
    membership = .operator_digest_payload(x$membership),
    decoder = .operator_digest_payload(x$decoder),
    atlas = x$atlas,
    aggregation = x$aggregation
  ))
}
#' @export
restrict_space.parcel_space <- function(x, index, ...) {
  parcel_space(
    parent = x$parent,
    parcel_ids = x$parcel_ids[index],
    membership = x$membership[, index, drop = FALSE],
    data = x$data[index, , drop = FALSE],
    atlas = x$atlas,
    aggregation = x$aggregation,
    decoder = x$decoder[, index, drop = FALSE],
    metadata = x$metadata
  )
}
#' @export
vectorize_space.parcel_space <- function(x, spatial_object, ...) {
  parent_values <- vectorize_space(x$parent, spatial_object, ...)
  if (length(parent_values) != n_features(x$parent)) {
    .frame_abort("Parent vectorization returned an incompatible feature vector.",
                 "fmridataset_error_space_mismatch")
  }
  as.numeric(x$aggregation_operator %*% parent_values)
}
#' @export
reconstruct_space.parcel_space <- function(x, vector, ...) {
  if (!is.null(names(vector))) vector <- vector[feature_ids(x)]
  vector <- as.numeric(vector)
  if (length(vector) != n_features(x)) {
    .frame_abort("Vector does not match the parcel axis.",
                 "fmridataset_error_space_mismatch")
  }
  parent_values <- as.numeric(x$decoder %*% vector)
  uncovered <- as.numeric(Matrix::rowSums(x$membership)) == 0
  parent_values[uncovered] <- NA_real_
  reconstruct_space(x$parent, parent_values, ...)
}
#' @export
adjacency.parcel_space <- function(x, ...) {
  parent_graph <- adjacency(x$parent, ...)
  if (is.null(parent_graph)) return(NULL)
  contact <- Matrix::t(x$membership) %*% parent_graph %*% x$membership
  contact <- methods::as(contact != 0, "CsparseMatrix")
  diag(contact) <- FALSE
  Matrix::drop0(contact)
}

#' Build a parcel space from a neuroatlas atlas
#'
#' The adapter delegates atlas identity and label interpretation to
#' `neuroatlas`. For surface atlases it uses `get_roi()` so atlas-specific
#' hemisphere-local coding is not duplicated here.
#'
#' @param atlas A `neuroatlas` atlas or surfatlas.
#' @param parent The aligned parent `volume_space` or `surface_space`.
#' @param aggregation Aggregation method passed to [parcel_space()].
#' @param metadata Serializable metadata passed to [parcel_space()].
#' @return A `parcel_space` aligned to `parent`.
#' @export
parcel_space_from_atlas <- function(atlas, parent,
                                    aggregation = c("mean", "sum"),
                                    metadata = list()) {
  if (!requireNamespace("neuroatlas", quietly = TRUE)) {
    .frame_abort("parcel_space_from_atlas() requires neuroatlas.",
                 "fmridataset_error_space_mismatch")
  }
  if (!inherits(atlas, "atlas")) {
    .frame_abort("atlas must inherit from neuroatlas class 'atlas'.",
                 "fmridataset_error_space_mismatch")
  }
  pd <- neuroatlas::as_parcel_data(atlas)
  parcels <- pd$parcels
  k <- nrow(parcels)
  rows <- integer()
  cols <- integer()
  if (inherits(atlas, "surfatlas")) {
    if (!inherits(parent, "surface_space")) {
      .frame_abort("A surface atlas requires a surface_space parent.",
                   "fmridataset_error_space_mismatch")
    }
    fd <- feature_data(parent)
    for (j in seq_len(k)) {
      roi <- neuroatlas::get_roi(atlas, id = parcels$id[[j]])[[1L]]
      local_index <- neuroim2::indices(roi)
      hit <- which(
        fd$hemisphere == parcels$hemi[[j]] &
          fd$.hemisphere_vertex_index %in% local_index
      )
      rows <- c(rows, hit)
      cols <- c(cols, rep.int(j, length(hit)))
    }
  } else {
    if (!inherits(parent, "volume_space")) {
      .frame_abort("A volumetric atlas requires a volume_space parent.",
                   "fmridataset_error_space_mismatch")
    }
    labels <- vectorize_space(parent, atlas$atlas)
    matched <- match(as.character(labels), as.character(parcels$id))
    keep <- which(!is.na(matched))
    rows <- keep
    cols <- matched[keep]
  }
  membership <- Matrix::sparseMatrix(
    i = rows, j = cols, x = 1,
    dims = c(n_features(parent), k)
  )
  empty <- which(Matrix::colSums(membership) == 0)
  if (length(empty)) {
    .frame_abort(
      paste0("Parent support contains no features for atlas parcel(s): ",
             paste(parcels$id[empty], collapse = ", "), "."),
      "fmridataset_error_space_mismatch"
    )
  }
  parcel_space(
    parent = parent,
    parcel_ids = parcels$id,
    membership = membership,
    data = parcels,
    atlas = pd$atlas,
    aggregation = match.arg(aggregation),
    metadata = metadata
  )
}
