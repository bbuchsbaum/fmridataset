# Update Golden Test Data

Update existing golden test data with new expected outputs. Use this
when intentional changes to the package require updating the reference
data.

## Usage

``` r
update_golden_test_data(
  output_dir = "tests/testthat/golden",
  seed = 42,
  confirm = TRUE
)
```

## Arguments

- output_dir:

  Directory where golden data files will be saved. Defaults to
  "tests/testthat/golden".

- seed:

  Random seed for reproducibility. Defaults to 42.

- confirm:

  Logical. If TRUE, will prompt for confirmation before updating.
  Defaults to TRUE.

## Value

Invisibly returns TRUE on success.

## Details

This function should be used with caution as it will overwrite existing
golden test data. Only use when you are certain that the current outputs
are correct and should become the new reference.

## Examples

``` r
if (FALSE) { # \dontrun{
# Update golden data (will prompt for confirmation)
update_golden_test_data()

# Update without confirmation (use with caution!)
update_golden_test_data(confirm = FALSE)
} # }
```
