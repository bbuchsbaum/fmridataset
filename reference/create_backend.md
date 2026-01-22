# Create Backend Instance

Creates a backend instance using the registered factory function. This
is the main interface for creating backends by name.

## Usage

``` r
create_backend(name, ..., validate = TRUE)
```

## Arguments

- name:

  Character string, name of registered backend type

- ...:

  Arguments passed to the backend factory function

- validate:

  Logical, whether to validate the created backend (default: TRUE)

## Value

A storage backend object

## Examples

``` r
if (FALSE) { # \dontrun{
# Create a NIfTI backend (assuming it's registered)
backend <- create_backend("nifti",
  source = "data.nii",
  mask_source = "mask.nii"
)

# Create with validation disabled (faster, but riskier)
backend <- create_backend("nifti",
  source = "data.nii",
  mask_source = "mask.nii",
  validate = FALSE
)
} # }
```
