# BIDS Facade Testing Framework

## Overview

This document describes the comprehensive testing framework for the BIDS facade functionality in the `fmridataset` package. The tests use the `bidser` package's mock functionality to create realistic test scenarios without requiring actual BIDS datasets.

## Test Structure

### Core Test Files

1. **`test-bids-facade-phase1.R`** - Basic BIDS facade functionality
   - `bids()` constructor and facade creation
   - `discover()` method with elegant output
   - `print.bids_facade()` methods
   - Basic integration with `bidser::bids_project()`

2. **`test-bids-facade-phase2.R`** - Enhanced discovery and quality assessment
   - Enhanced `discover()` with quality metrics
   - `assess_quality()` functionality
   - Confound data integration
   - Quality thresholding and filtering logic

3. **`test-bids-facade-phase3.R`** - Performance and caching
   - `clear_cache()` functionality
   - Parallel processing with `bids_collect_datasets()`
   - Performance optimization with large datasets
   - Caching integration with bidser functions

4. **`test-bids-facade-phase4.R`** - Conversational interface
   - `focus_on()`, `from_young_adults()`, `with_excellent_quality()` filters
   - `preprocessed_with()` pipeline selection
   - `tell_me_about()` narrative summaries
   - `create_dataset()` with filter integration
   - Method chaining workflows

5. **`test-bids-facade-phase5.R`** - Workflow and community integration
   - `create_workflow()`, `describe()`, `add_step()` workflow building
   - `apply_workflow()` execution on fmri_dataset objects
   - `discover_best_practices()` community wisdom
   - Error resilience in workflow execution

6. **`test-bids-facade-integration.R`** - End-to-end integration testing
   - Complete workflows combining all phases
   - Performance benchmarking
   - Error handling across all phases
   - Complex real-world scenarios

7. **`test-bids-facade-simple.R`** - Simplified tests with correct bidser API
   - Basic bidser mock creation with proper format
   - Core bidser function integration
   - Facade structure validation

## Bidser Integration

### Correct File Structure Format

The `bidser::create_mock_bids()` function requires a specific data.frame format:

```r
file_structure_df <- data.frame(
  subid = c("sub-01", "sub-01", "sub-02", "sub-02"),
  datatype = c("func", "func", "func", "func"),
  suffix = c("bold", "events", "bold", "events"),
  fmriprep = c(FALSE, FALSE, FALSE, FALSE),
  task = c("rest", "rest", "rest", "rest"),  # Optional for functional data
  stringsAsFactors = FALSE
)
```

### Mock BIDS Creation

```r
mock_bids <- bidser::create_mock_bids(
  project_name = "test_project",
  participants = c("sub-01", "sub-02"),
  file_structure = file_structure_df,
  confound_data = confound_data,  # Optional
  dataset_description = list(     # Optional
    Name = "Test Dataset",
    BIDSVersion = "1.8.0"
  )
)
```

### Confound Data Format

```r
confound_data <- list(
  "sub-01" = list(
    "task-rest" = data.frame(
      framewise_displacement = rnorm(100, 0.1, 0.05),
      dvars = rnorm(100, 50, 10),
      trans_x = rnorm(100, 0, 0.1),
      trans_y = rnorm(100, 0, 0.1),
      trans_z = rnorm(100, 0, 0.1),
      rot_x = rnorm(100, 0, 0.02),
      rot_y = rnorm(100, 0, 0.02),
      rot_z = rnorm(100, 0, 0.02)
    )
  )
)
```

## Test Coverage

### Phase 1: Core Functionality
- ✅ Basic facade creation and structure
- ✅ Print methods with elegant output
- ✅ Discovery functionality with bidser backend
- ✅ Error handling for missing dependencies
- ✅ Integration with actual bidser functions

### Phase 2: Enhanced Discovery
- ✅ Quality metrics integration
- ✅ Enhanced discovery output
- ✅ Confound data processing
- ✅ Quality assessment workflows
- ✅ Preprocessing pipeline detection

### Phase 3: Performance Features
- ✅ Caching mechanism
- ✅ Cache clearing functionality
- ✅ Parallel processing logic
- ✅ Performance with large datasets
- ✅ Memory efficiency strategies

### Phase 4: Conversational Interface
- ✅ Natural language filters (task, age, quality, pipeline)
- ✅ Method chaining workflows
- ✅ Narrative summaries with `tell_me_about()`
- ✅ Filter integration with dataset creation
- ✅ Object integrity preservation

### Phase 5: Workflow & Community
- ✅ Workflow creation and management
- ✅ Step execution on fmri_dataset objects
- ✅ Community best practices integration
- ✅ Error resilience and recovery
- ✅ Workflow execution order preservation

## Running Tests

### Individual Phase Testing
```r
# Test specific phases
devtools::test_file("tests/testthat/test-bids-facade-phase1.R")
devtools::test_file("tests/testthat/test-bids-facade-phase2.R")
# ... etc
```

### Complete Test Suite
```r
# Run all BIDS facade tests
devtools::test(filter = "bids-facade")

# Run all tests
devtools::test()
```

### Simple Validation
```r
# Test basic bidser integration
devtools::test_file("tests/testthat/test-bids-facade-simple.R")
```

## Known Issues and Solutions

### 1. Bidser File Encoding Warnings
**Issue**: "Encoding failed for: sub-01_task-rest_bold - skipping this file in mock tree"
**Solution**: These are warnings from bidser's internal file naming logic and don't affect test functionality

### 2. Mock vs Real BIDS Behavior
**Issue**: Some bidser functions may behave differently with mock vs real data
**Solution**: Tests use `tryCatch()` to handle graceful failures and focus on interface testing

### 3. Dependency Management
**Issue**: Tests require bidser package which may not be installed
**Solution**: All tests use `skip_if_not_installed()` to gracefully skip when bidser unavailable

## Best Practices

### 1. Test Structure
- Each phase has its own test file for clarity
- Integration tests combine multiple phases
- Simple tests validate basic functionality

### 2. Mock Data Design
- Use realistic participant IDs and task names
- Include appropriate confound data for quality tests
- Test both minimal and comprehensive datasets

### 3. Error Handling
- Always use `tryCatch()` for potentially failing operations
- Test graceful degradation when dependencies missing
- Verify error messages are informative

### 4. Performance Testing
- Test with multiple subjects for parallelization
- Verify reasonable execution times
- Test memory efficiency with large datasets

## Future Enhancements

1. **Additional Mock Scenarios**
   - Multi-session datasets
   - Different preprocessing pipelines
   - Various confound variable sets

2. **Integration Testing**
   - Real BIDS dataset validation
   - Performance benchmarking with large datasets
   - Cross-platform compatibility testing

3. **Workflow Testing**
   - More complex processing pipelines
   - Error recovery scenarios
   - Workflow serialization/deserialization

## Conclusion

This testing framework provides comprehensive coverage of the BIDS facade functionality, ensuring robust integration with the bidser package while maintaining elegant interfaces for users. The tests validate both individual components and complete workflows, providing confidence in the system's reliability and performance. 