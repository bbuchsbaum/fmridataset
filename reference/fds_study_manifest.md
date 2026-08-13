# Construct and validate an FDS v1 study manifest

Study manifests retain shared entities, typed links, relational tables,
and the semantic manifests of every frame or collection member.
Numerical sources remain separate bindings so physical storage packages
do not own or reinterpret study semantics.

## Usage

``` r
fds_study_manifest(x)

validate_fds_study_manifest(manifest)
```

## Arguments

- x:

  An `fmri_study` or filtered study view.

- manifest:

  An FDS study manifest.

## Value

`fds_study_manifest()` returns a serializable source-free manifest;
`validate_fds_study_manifest()` returns `manifest` invisibly.
