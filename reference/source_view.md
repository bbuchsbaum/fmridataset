# Construct a lazy view over an array source

A view stores its selectors in the package's normalized selection form
rather than as expanded position vectors: a select-all axis stores no
vector, one contiguous run stores its bounds, and only an arbitrary
subset stores positions. A view over a view composes into one view over
the root source. Selectors follow the package-wide normalization law:
logical masks must match the axis length without `NA`; numeric positions
must be whole numbers, may reorder, may be negative (but not mixed with
positive), drop zero, must be in bounds, and may not repeat an element;
an empty selection is legal. Fingerprints hash the normalized form, so
equal selections agree however they were expressed and a select-all view
fingerprints in constant time.

## Usage

``` r
source_view(source, observations = NULL, features = NULL)
```

## Arguments

- source:

  An `array_source`.

- observations:

  Stored observation selector.

- features:

  Stored feature selector.

## Value

A serializable source view.

## Examples

``` r
src <- memory_source(matrix(seq_len(6), nrow = 2))
view <- source_view(src, observations = 1)
source_shape(view)
#> [1] 1 3
```
