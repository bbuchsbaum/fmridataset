# Validate Golden Test Data

Check that all expected golden test data files exist and are readable.

## Usage

``` r
validate_golden_test_data(output_dir = "tests/testthat/golden")
```

## Arguments

- output_dir:

  Directory where golden data files will be saved. Defaults to
  "tests/testthat/golden".

## Value

A logical vector indicating which files exist, with names corresponding
to the expected files.

## Examples

``` r
if (FALSE) { # \dontrun{
# Check golden data files
validate_golden_test_data()
} # }
```
