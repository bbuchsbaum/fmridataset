# Upgrade a provisional frame-link descriptor

Version-one links used reverse `*_from` endpoints for derivation,
mapping, and alignment and stored feature operators in metadata. This
explicit migration converts them to the canonical source-to-target
version-two form.

## Usage

``` r
upgrade_frame_link(x)
```

## Arguments

- x:

  A provisional version-one or canonical version-two `frame_link`.

## Value

A canonical version-two `frame_link`.

## Examples

``` r
link <- frame_link("bold", "surface", type = "derivation")
identical(upgrade_frame_link(link), link)
#> [1] TRUE
```
