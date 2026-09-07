# Identity and canonicalization contract

Canonicalization version 1 hashes normalized R objects with SHA-256 over
a package-owned tagged binary encoding. It is stable for supported
package operations and round trips, but is deliberately R-only rather
than a cross-language wire encoding.

## Usage

``` r
canonicalization_contract()
```

## Value

A serializable canonicalization descriptor.

## Examples

``` r
contract <- canonicalization_contract()
contract$algorithm
#> [1] "sha256"
```
