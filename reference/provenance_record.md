# Create a content-addressed provenance record

Create a content-addressed provenance record

## Usage

``` r
provenance_record(
  operation,
  parents = character(),
  inputs = list(),
  parameters = list(),
  outputs = list(),
  software = list(package = "fmridataset"),
  metadata = list()
)
```

## Arguments

- operation:

  Stable operation name.

- parents:

  IDs of direct parent records.

- inputs, parameters, outputs, software, metadata:

  Serializable record data.

## Value

A `provenance_record`.
