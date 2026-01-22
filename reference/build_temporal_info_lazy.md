# Temporal metadata builders for fmri_series

Internal helpers used to construct the `temporal_info` component of
`fmri_series` objects. These functions return data.frame objects
describing each selected timepoint. They are not exported for users.

## Usage

``` r
build_temporal_info_lazy(dataset, time_indices)
```
