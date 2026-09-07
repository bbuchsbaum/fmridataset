# Compose source-to-target frame links

The target of `first` must be the source of `second`, and their
addressed axes must agree. Explicit ID maps are joined through the
intermediate IDs. Feature operators are composed as `second %*% first`
without changing link direction.

## Usage

``` r
compose_frame_links(first, second, type = NULL, metadata = list())
```

## Arguments

- first:

  A canonical source-to-intermediate `frame_link`.

- second:

  A canonical intermediate-to-target `frame_link`.

- type:

  Result link type. It may be omitted when both inputs have the same
  type.

- metadata:

  Unaligned result-link metadata.

## Value

A canonical source-to-target `frame_link`.

## Examples

``` r
first <- frame_link("bold", "beta", type = "derivation")
second <- frame_link("beta", "summary", type = "derivation")
composed <- compose_frame_links(first, second)
composed$source
#> [1] "bold"
composed$target
#> [1] "summary"
```
