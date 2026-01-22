# Create an fmri_group (one row per subject)

Create an fmri_group (one row per subject)

## Usage

``` r
fmri_group(
  subjects,
  id,
  dataset_col = "dataset",
  space = NULL,
  mask_strategy = c("subject_specific", "intersect", "union")
)
```

## Arguments

- subjects:

  A `data.frame` (or tibble) with one row per subject where one column
  contains per-subject `fmri_dataset` objects stored as a list column.

- id:

  Character scalar giving the name of the subject identifier column.

- dataset_col:

  Character scalar naming the list column that stores the per-subject
  dataset handles.

- space:

  Optional character string describing the nominal common space for all
  subjects (e.g., "MNI152NLin2009cAsym").

- mask_strategy:

  One of "subject_specific", "intersect", or "union" describing how
  masks should be handled when combining subjects. This is a declarative
  flag only; no resampling is performed by the constructor.

## Value

An object of class `fmri_group` that wraps the input table.
