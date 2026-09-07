# Encode and hash canonical R values

Canonicalization version 1 normalizes semantic R values and writes a
tagged, length-prefixed package binary format. It is an explicitly
R-only format. Named lists and attributes use lexicographic field order;
unnamed list order is preserved. Strings are UTF-8 NFC, sparse matrix
storage layouts are normalized, and NaN payloads use one R value while
remaining distinct from `NA_real_`. Negative zero remains distinct from
zero.

## Usage

``` r
canonical_bytes(x)

canonical_sha256(x)
```

## Arguments

- x:

  A serializable R value.

## Value

`canonical_bytes()` returns a raw vector. `canonical_sha256()` returns
its lowercase SHA-256 hexadecimal digest.

## Examples

``` r
canonical_bytes(list(b = 2, a = 1))
#>   [1] 6f 72 67 2e 66 6d 72 69 64 61 74 61 73 65 74 2e 72 2d 63 61 6e 6f 6e 69 63
#>  [26] 61 6c 2f 76 31 0a 76 00 00 00 02 64 00 00 00 01 66 3f f0 00 00 00 00 00 00
#>  [51] 41 00 00 00 00 64 00 00 00 01 66 40 00 00 00 00 00 00 00 41 00 00 00 00 41
#>  [76] 00 00 00 01 31 00 00 00 05 6e 61 6d 65 73 63 00 00 00 02 31 00 00 00 01 61
#> [101] 31 00 00 00 01 62 41 00 00 00 00
canonical_sha256(list(b = 2, a = 1))
#> [1] "3d75746b0bf00b57749d4be797741ab646dfd08d56864db493999cfe6e596e6f"
```
