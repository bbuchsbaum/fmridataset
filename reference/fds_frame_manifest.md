# Construct and validate an FDS v1 frame manifest

The manifest owns semantic alignment but deliberately excludes physical
source descriptors. Storage packages bind assay names to physical array
locations separately and reconstruct frames with
[`frame_from_fds_manifest()`](https://bbuchsbaum.github.io/fmridataset/reference/frame_from_fds_manifest.md).

## Usage

``` r
fds_frame_manifest(x)

validate_fds_manifest(manifest)
```

## Arguments

- x:

  An `fmri_frame`.

- manifest:

  An FDS manifest.

## Value

A serializable backend-neutral manifest.
