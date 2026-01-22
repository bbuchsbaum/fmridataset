# Helper Functions for Vignettes

Internal functions to support vignette examples with synthetic data and
consistent demonstrations.

## Usage

``` r
generate_example_fmri_data(
  n_timepoints = 200,
  n_voxels = 1000,
  n_active = 100,
  activation_periods = NULL,
  signal_strength = 0.5,
  seed = 123
)
```

## Arguments

- n_timepoints:

  Number of time points

- n_voxels:

  Number of voxels

- n_active:

  Number of active voxels with signal

- activation_periods:

  Time indices where activation occurs

- signal_strength:

  Strength of activation signal

- seed:

  Random seed for reproducibility

## Value

Matrix of synthetic fMRI data

## Examples

``` r
if (FALSE) { # \dontrun{
data <- generate_example_fmri_data(200, 1000)
} # }
```
