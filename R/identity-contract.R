#' Inspect typed identity domains
#'
#' Semantic and schema identities are independent of physical locations.
#' Source fingerprints identify descriptors and selectors, not array contents.
#' Content digests are optional backend-supplied receipts and are never inferred
#' by reading data.
#'
#' @param x A frame, frame schema, feature space, array source, FDS manifest,
#'   provenance graph, collection, or study.
#' @param domain Identity domain. Usually inferred from `x`.
#' @param content_digest Optional externally computed content digest.
#' @return A serializable typed identity descriptor.
#' @export
identity_descriptor <- function(
    x,
    domain = c("auto", "semantic", "schema", "space", "source", "provenance", "content"),
    content_digest = NULL) {
  domain <- match.arg(domain)
  inferred <- if (inherits(x, "fmri_frame_schema")) "schema"
  else if (inherits(x, "feature_space")) "space"
  else if (inherits(x, "array_source")) "source"
  else if (inherits(x, "provenance_graph")) "provenance"
  else "semantic"
  if (identical(domain, "auto")) domain <- inferred
  digest_value <- switch(
    domain,
    schema = frame_schema_digest(x),
    space = space_digest(x),
    source = source_fingerprint(x),
    provenance = provenance_digest(x),
    content = {
      if (!is.character(content_digest) || length(content_digest) != 1L ||
          is.na(content_digest) || !nzchar(content_digest)) {
        .frame_abort(
          "content identity requires one externally computed digest.",
          "fmridataset_error_identity", field = "content_digest"
        )
      }
      content_digest
    },
    semantic = {
      if (inherits(x, "fmri_frame")) fds_manifest_digest(fds_frame_manifest(x))
      else if (inherits(x, "fmri_collection")) collection_digest(x)
      else if (inherits(x, "fmri_study")) study_digest(x)
      else if (is.list(x) && identical(x$object_type, "fmri_frame")) fds_manifest_digest(x)
      else .canonical_digest(x)
    }
  )
  structure(
    list(
      domain = domain, digest = digest_value,
      canonicalization = canonicalization_contract(),
      content_digest = if (identical(domain, "content")) digest_value else NULL
    ),
    class = "fmri_identity"
  )
}
