# Reconstruct a frame from an FDS manifest and physical sources

Reconstruct a frame from an FDS manifest and physical sources

## Usage

``` r
frame_from_fds_manifest(manifest, bindings)
```

## Arguments

- manifest:

  A valid FDS v1 frame manifest.

- bindings:

  Named physical array payloads or `array_source` descriptors, one per
  manifest array declaration.

## Value

An `fmri_frame` whose semantic state comes from `manifest` and whose
lazy arrays come from `bindings`.
