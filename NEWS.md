# fmridataset (development version)

## New features

* Added comprehensive CI/CD pipeline with GitHub Actions
* Added test coverage reporting with codecov
* Added code style checking and automatic formatting
* Added issue and PR templates for better project management
* Implemented `as_tibble.fmri_study_dataset` with metadata optimization

## Bug fixes

* Fixed chunking edge case when `nchunks > number of voxels`
* Updated deprecated `with_mock()` calls to `with_mocked_bindings()`
* Fixed dimensional consistency issues in storage backends
* Resolved all test failures from package refactoring

## Documentation

* Added comprehensive README with badges and examples
* Improved package architecture documentation
* Added codecov configuration for coverage reporting

## Internal changes

* Refactored monolithic codebase into modular architecture
* Improved test organization and coverage
* Enhanced error handling and validation
* Modernized CI/CD workflows and tooling 