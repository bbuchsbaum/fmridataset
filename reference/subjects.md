# Access the subjects tibble stored inside an fmri_group

Access the subjects tibble stored inside an fmri_group

## Usage

``` r
subjects(x)

subjects(x) <- value
```

## Arguments

- x:

  An `fmri_group`.

- value:

  A replacement table containing the dataset column used by the group.

## Value

The underlying `data.frame` with one row per subject.

An updated `fmri_group`.
