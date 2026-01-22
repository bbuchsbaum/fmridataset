# Generate Golden Test Data

Generate reference data for golden tests. This function creates
reproducible test data that can be used to validate consistency across
package versions.

## Usage

``` r
generate_golden_test_data(output_dir = "tests/testthat/golden", seed = 42)
```

## Arguments

- output_dir:

  Directory where golden data files will be saved. Defaults to
  "tests/testthat/golden".

- seed:

  Random seed for reproducibility. Defaults to 42.

## Value

Invisibly returns TRUE on success.

## Details

This function generates the following golden test data:

- reference_data.rds - Basic test matrices and metadata

- matrix_dataset.rds - Example fmri_dataset object

- fmri_series.rds - Example FmriSeries data

- sampling_frame.rds - Example sampling_frame object

- mock_neurvec.rds - Mock NeuroVec object for testing

## Examples

``` r
if (FALSE) { # \dontrun{
# Generate golden test data
generate_golden_test_data()

# Generate with custom seed
generate_golden_test_data(seed = 123)
} # }
```
