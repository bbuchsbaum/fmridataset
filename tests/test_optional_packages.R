#!/usr/bin/env Rscript

# Installed-package smoke checks for optional 1.0 integrations.

cat("Testing optional fmridataset 1.0 integrations...\n\n")

optional_packages <- list(
  fmristore = "Certified HDF5 frame and study persistence",
  fmrilatent = "Latent-space decoder interoperability",
  neuroatlas = "Atlas-to-parcel-space interoperability",
  neurosurf = "Surface geometry interoperability",
  multidesign = "Frame-native design compilation",
  fmrigds = "Frame-native statistical execution",
  zarr = "Experimental Zarr array source",
  jsonlite = "Machine-readable FDS envelopes"
)

installed_names <- rownames(utils::installed.packages())
installed <- names(optional_packages) %in% installed_names
names(installed) <- names(optional_packages)

for (package in names(optional_packages)) {
  cat(sprintf(
    "%s %-12s : %s\n",
    if (installed[[package]]) "available" else "missing  ",
    package,
    optional_packages[[package]]
  ))
}

stopifnot(exists(
  "zarr_array_source",
  envir = asNamespace("fmridataset"),
  inherits = FALSE
))

cat(sprintf(
  "\nAvailable: %d/%d optional integrations\n",
  sum(installed),
  length(installed)
))
