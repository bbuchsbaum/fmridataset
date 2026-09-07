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

## Examples

``` r
r <- provenance_record("normalize", inputs = list(method = "zscore"))
r$operation
#> [1] "normalize"
r$id
#> [1] "8acf06efadec5badb0d153c7ef17603a143dea261745f1cb4b0513413021c3dc"
```
