# Contributing to fmridataset

We welcome contributions to the fmridataset package! This document provides guidelines for contributing.

## Code Style

This package follows the tidyverse style guide with some modifications:

- **Line length**: Maximum 120 characters
- **Naming**: Use snake_case for functions and variables
- **Documentation**: All exported functions must have roxygen2 documentation
- **S3 methods**: Follow the pattern `method_name.class_name`

## Development Setup

1. Fork and clone the repository
2. Install development dependencies:
   ```r
   install.packages(c("devtools", "testthat", "lintr", "roxygen2"))
   ```
3. Load the package in development mode:
   ```r
   devtools::load_all()
   ```

## Making Changes

1. Create a new branch for your feature or bug fix
2. Write tests for any new functionality
3. Ensure all tests pass: `devtools::test()`
4. Check the package: `devtools::check()`
5. Run the linter: `lintr::lint_package()`
6. Update documentation if needed: `devtools::document()`

## Backend Development

If you're adding a new storage backend:

1. Implement all methods from the `StorageBackend` contract (see `R/storage_backend.R`)
2. Validate your backend with `validate_backend()`
3. Add comprehensive tests
4. Document the backend in the extending vignette

## Testing

- Write tests using testthat
- Place test files in `tests/testthat/`
- Name test files as `test-<functionality>.R`
- Use descriptive test names

## Pull Request Process

1. Update NEWS.md with your changes
2. Ensure all CI checks pass
3. Request review from maintainers
4. Address any feedback

## Questions?

Feel free to open an issue for any questions about contributing.