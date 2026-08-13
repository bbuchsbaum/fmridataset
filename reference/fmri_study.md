# Construct a linked fMRI study

Construct a linked fMRI study

## Usage

``` r
fmri_study(
  frames,
  entities = list(),
  links = list(),
  tables = list(),
  metadata = list(),
  provenance = NULL
)
```

## Arguments

- frames:

  Named `fmri_frame` or `fmri_collection` representations.

- entities:

  Shared authoritative entity registry.

- links:

  Named `frame_link` descriptors.

- tables:

  Named relational tables, including `event_table` objects.

- metadata:

  Serializable study metadata.

- provenance:

  Serializable provenance records.

## Value

An `fmri_study`.
