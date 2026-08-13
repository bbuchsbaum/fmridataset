# Inspect basis-space operators

Inspect basis-space operators

## Usage

``` r
basis_analysis(x)

basis_synthesis(x)

basis_projection_info(x)
```

## Arguments

- x:

  A `basis_space`.

## Value

`basis_analysis()` returns the parent-to-component analysis operator;
`basis_synthesis()` returns the optional component-to-parent synthesis
operator; `basis_projection_info()` returns validation metadata. These
names deliberately avoid colliding with
[`fmrilatent::basis_decoder()`](https://rdrr.io/pkg/fmrilatent/man/basis_decoder.html),
which constructs model-level decoders.
