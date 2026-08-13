# Serializable numerical array sources

Serializable numerical array sources

## Usage

``` r
as_array_source(x, ...)

source_shape(x, ...)

source_dtype(x, ...)

source_chunks(x, ...)

source_capabilities(x, ...)

source_fingerprint(x, ...)

source_open(x, ...)

source_read(x, observations = NULL, features = NULL, ...)

source_read_native(x, observations = NULL, ...)

source_close(x, ...)
```

## Arguments

- x:

  An array source or object coercible to one.

- ...:

  Additional method arguments.

- observations:

  Optional observation positions.

- features:

  Optional feature positions.
