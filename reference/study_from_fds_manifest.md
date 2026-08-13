# Reconstruct a study from semantic and physical components

Reconstruct a study from semantic and physical components

## Usage

``` r
study_from_fds_manifest(manifest, representations, bindings = list())
```

## Arguments

- manifest:

  A valid FDS v1 study manifest.

- representations:

  Named lazy frames or collections matching the representation
  manifests.

- bindings:

  Named physical bindings for shared study arrays.

## Value

An `fmri_study`.
