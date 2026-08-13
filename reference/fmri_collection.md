# Construct a collection of semantically equivalent fMRI frames

A collection keeps frames separate when they share an observational and
assay contract but cannot share a feature axis, as with
participant-native volume or surface spaces. Equal feature dimensions or
IDs are not required; feature-space type and annotation semantics are
validated explicitly.

## Usage

``` r
fmri_collection(frames, metadata = list(), provenance = NULL)
```

## Arguments

- frames:

  A non-empty named list of `fmri_frame` objects or lazy views.

- metadata:

  Serializable collection metadata.

- provenance:

  Serializable provenance records.

## Value

An `fmri_collection`.
