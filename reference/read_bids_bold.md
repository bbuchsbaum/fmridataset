# Read one subject's preprocessed BIDS BOLD data as an fmri_frame

`read_bids_bold()` is the narrow BIDS on-ramp to the canonical frame
API. It discovers fMRIPrep `desc-preproc` BOLD runs, creates
deterministic volume IDs, resolves one common volume space, and returns
a lazy NIfTI-backed frame. BOLD values are not read until an assay or
spatial map is explicitly collected.

## Usage

``` r
read_bids_bold(
  path,
  subject,
  task = NULL,
  session = NULL,
  run = NULL,
  space = NULL,
  derivative = "fmriprep",
  mask = "intersection",
  events = TRUE,
  chunks = NULL
)
```

## Arguments

- path:

  Path to a BIDS dataset containing fMRIPrep derivatives.

- subject:

  One exact subject label, with or without the `sub-` prefix.

- task, session, run, space:

  Optional exact BIDS entity selectors.

- derivative:

  Currently only `"fmriprep"`.

- mask:

  `"intersection"` (default), an explicit NIfTI mask path, or a
  compatible `volume_space`.

- events:

  Whether to attach matching BIDS events as a keyed event table.

- chunks:

  Optional observation-by-feature chunk hint passed to
  [`nifti_array_source()`](https://bbuchsbaum.github.io/fmridataset/reference/nifti_array_source.md).

## Value

A lazy `fmri_frame` with a `signal` assay.

## Examples

``` r
if (FALSE) { # \dontrun{
bold <- read_bids_bold(
  "/data/my-study",
  subject = "01",
  task = "memory",
  space = "MNI152NLin2009cAsym"
)
first_run <- filter_obs(bold, run_id == "run-1")
map <- spatial_map(bold, observation = 1)
} # }
```
