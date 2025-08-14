# Golden Test Data

This directory contains reference data files used for golden testing in the fmridataset package.

## What are Golden Tests?

Golden tests (also known as "golden master" or "characterization" tests) compare current outputs against previously saved reference outputs. They help ensure that:

1. Core functionality remains consistent across versions
2. Refactoring doesn't introduce unintended changes
3. Data structures maintain their expected format
4. Complex computations produce consistent results

## Files in this Directory

- `reference_data.rds` - Basic test matrices and metadata used across tests
- `matrix_dataset.rds` - Reference fmri_dataset object  
- `fmri_series.rds` - Reference FmriSeries data structure
- `sampling_frame.rds` - Reference sampling_frame object
- `mock_neurvec.rds` - Mock NeuroVec object for backend testing

## Generating Golden Data

To regenerate the golden test data:

```r
# From package root directory
source("inst/scripts/generate_golden_data.R")
```

Or use the provided R functions:

```r
devtools::load_all()
fmridataset::generate_golden_test_data()
```

## Updating Golden Data

When intentional changes require updating reference data:

```r
# This will prompt for confirmation and backup existing data
fmridataset::update_golden_test_data()
```

**Important**: Only update golden data when you are certain the new outputs are correct!

## Running Golden Tests

```r
# Run all golden tests
devtools::test(filter = "golden")

# Run specific golden test files
testthat::test_file("tests/testthat/test-golden-datasets.R")
testthat::test_file("tests/testthat/test-golden-fmriseries.R")
testthat::test_file("tests/testthat/test-golden-backends.R")
testthat::test_file("tests/testthat/test-golden-sampling-frame.R")
testthat::test_file("tests/testthat/test-golden-snapshots.R")
```

## Best Practices

1. **Version Control**: Always commit golden data files to git
2. **Review Changes**: Carefully review any differences when tests fail
3. **Document Updates**: When updating golden data, document why in commit messages
4. **Reproducibility**: Use consistent random seeds (default: 42)
5. **Cross-Platform**: Be aware that numeric precision may vary slightly across platforms

## Troubleshooting

If golden tests fail:

1. Check if the failure is due to intentional changes
2. Compare actual vs expected outputs to understand differences
3. If changes are correct, update golden data
4. If changes are incorrect, fix the regression

## Snapshot Tests

In addition to golden data files, the package uses testthat's snapshot testing for:
- Print method outputs
- Error messages
- Summary displays

Snapshots are stored in `tests/testthat/_snaps/` and are automatically managed by testthat.