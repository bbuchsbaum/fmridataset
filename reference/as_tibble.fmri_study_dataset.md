# Convert fmri_study_dataset to a tibble or lazy matrix

Primary data access method for study-level datasets. By default this
returns a lazy matrix (typically a `delarr` object) with row-wise
metadata attached. When `materialise = TRUE`, the data matrix is
materialised and returned as a tibble with metadata columns prepended.

## Usage

``` r
# S3 method for class 'fmri_study_dataset'
as_tibble(x, materialise = FALSE, ...)
```

## Arguments

- x:

  An `fmri_study_dataset` object

- materialise:

  Logical; return a materialised tibble? Default `FALSE`.

- ...:

  Additional arguments (unused)

## Value

Either a lazy matrix with metadata attributes or a tibble when
`materialise = TRUE`.
