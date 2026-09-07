#' fmridataset: spatially typed, annotated fMRI datasets
#'
#' `fmridataset` keeps fMRI arrays aligned with what their rows and columns
#' mean. Its canonical container, [fmri_frame()], represents one observation
#' domain by one feature domain and carries stable IDs, annotations, entities,
#' relations, and explicit spatial identity through lazy views, feature maps,
#' binding, and storage round trips. [fmri_collection()] groups equivalent
#' frames that cannot yet share a feature axis, and [fmri_study()] links
#' heterogeneous frames through shared entities and typed links.
#'
#' Numerical values live behind serializable array sources
#' ([memory_source()], [nifti_array_source()], [zarr_array_source()], and the
#' sources provided by storage packages) and are read only on request, under a
#' realization budget. The logical frame contract is persisted through FDS
#' manifests ([fds_frame_manifest()]) and HDF5 via [write_frame()].
#'
#' See `inst/architecture/` for the decision records that define the data
#' model, the FDS schema, and the identity, temporal, and ID policies.
#'
#' @keywords internal
#' @importFrom stats setNames
#' @importFrom utils head tail
"_PACKAGE"
