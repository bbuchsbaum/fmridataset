# Golden Tests Status

## Summary

Golden tests have been initialized for the fmridataset package. This framework provides regression testing by comparing current outputs against saved reference data.

## Created Files

### Test Infrastructure
1. **helper-golden.R** - Helper functions for golden testing
   - `generate_reference_data()` - Creates test data
   - `save_golden_data()` - Saves reference data
   - `load_golden_data()` - Loads reference data with path resolution
   - `compare_golden()` - Compares data with tolerance
   - `generate_all_golden_data()` - Main generation function

2. **Golden Test Files**
   - `test-golden-datasets.R` - Tests for fmri_dataset objects ✅
   - `test-golden-fmriseries.R` - Tests for FmriSeries objects ⚠️ 
   - `test-golden-backends.R` - Tests for storage backends ⚠️
   - `test-golden-sampling-frame.R` - Tests for sampling_frame objects
   - `test-golden-snapshots.R` - Snapshot tests for print methods

3. **Utilities**
   - `R/golden_data_generation.R` - User-facing functions
   - `inst/scripts/generate_golden_data.R` - Generation script
   - `_testthat.yml` - Testthat edition configuration

4. **Golden Data Files** (generated)
   - `reference_data.rds` - Basic test data
   - `matrix_dataset.rds` - Example dataset
   - `fmri_series.rds` - Example FmriSeries
   - `sampling_frame.rds` - Example sampling frame
   - `mock_neurvec.rds` - Mock NeuroVec object

## Current Status

### ✅ Working Tests
- `test-golden-datasets.R` - All 38 tests passing
- Basic golden data generation
- Chunking iterator tests
- Multi-run dataset handling

### ⚠️ Known Issues
1. **Backend Tests** - Need to fix:
   - Parameter names (data_matrix vs data)
   - Dimension return format (list vs numeric)
   - Method parameters (rows vs time_idx)

2. **FmriSeries Tests** - Need to fix:
   - run_length parameter in test files
   - Multi-run dataset creation

3. **Snapshot Tests** - Skipped when edition < 3

## Usage

### Generate Golden Data
```r
# From package root
source("inst/scripts/generate_golden_data.R")

# Or using package function
fmridataset::generate_golden_test_data()
```

### Run Golden Tests
```r
# All golden tests
devtools::test(filter = "golden")

# Specific test file
testthat::test_file("tests/testthat/test-golden-datasets.R")
```

### Update Golden Data
```r
# With confirmation prompt
fmridataset::update_golden_test_data()

# Without prompt (use carefully!)
fmridataset::update_golden_test_data(confirm = FALSE)
```

## Next Steps

1. Fix remaining test issues in backend and FmriSeries tests
2. Add more comprehensive test cases
3. Consider adding performance benchmarks
4. Add golden tests for other components (latent datasets, etc.)

## Notes

- Golden data files use RDS format with version = 2 for compatibility
- Tests use tolerance-based comparison for numeric data
- Path resolution handles different working directories
- Snapshot tests require testthat edition 3