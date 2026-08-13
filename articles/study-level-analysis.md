# Study-Level Analysis: From Single Subjects to Group Studies

Multi-subject fMRI studies require a unified interface that scales from
a single participant to a full cohort without rewriting analysis code.
`fmridataset` provides
[`fmri_study_dataset()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study_dataset.md)
for exactly this purpose: wrap per-subject datasets into one object,
then apply the same methods you already use on single subjects.

## Create per-subject datasets

Each subject is an ordinary `matrix_dataset` (or any other backend —
NIfTI, HDF5, Zarr). Create them once and collect them in a named list.

``` r

ds_01 <- matrix_dataset(
  datamat    = matrix(rnorm(200 * 500), nrow = 200, ncol = 500),
  TR         = 2.0,
  run_length = c(100, 100)
)
ds_02 <- matrix_dataset(
  datamat    = matrix(rnorm(200 * 500), nrow = 200, ncol = 500),
  TR         = 2.0,
  run_length = c(100, 100)
)
```

## Compose into fmri_study_dataset()

Pass a named list of datasets and matching subject identifiers.

``` r

study <- fmri_study_dataset(
  datasets    = subject_list,
  subject_ids = subject_ids
)
print(study)
#> 
#> === fMRI Dataset ===
#> 
#> ** Dimensions:
#>   - Timepoints: 1000 
#>   - Runs: 10  
#>   - Voxels in mask: (lazy)
#> 
#> ** Temporal Structure:
#>   - TR: 2 seconds
#>   - Run lengths: 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 
#> 
#> ** Event Table:
#>   - Empty event table
```

The study object stores a lazy reference to each subject’s data; nothing
is loaded into memory until you explicitly request it.

## Access data

Retrieve the full concatenated matrix, or pull one subject at a time.

``` r

all_data <- get_data_matrix(study)
dim(all_data) # (total timepoints) x (voxels)
#> [1] 1000  500
```

``` r

s01 <- get_data_matrix(study, subject_id = "sub-01")
dim(s01) # timepoints for sub-01 x voxels
#> [1] 200 500
```

Subject-specific access is the most memory-efficient pattern for large
cohorts: load, compute, discard, repeat.

## Chunked iteration

[`data_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md)
returns an iterator that yields successive voxel blocks. Use it when the
full data matrix does not fit in memory.

``` r

chunks <- data_chunks(study, nchunks = 4)
repeat {
  ch <- tryCatch(iterators::nextElem(chunks), error = function(e) NULL)
  if (is.null(ch)) break
  cat("chunk", ch$chunk_num, ":", ncol(ch$data), "voxels\n")
}
#> chunk 1 : 125 voxels
#> chunk 2 : 125 voxels
#> chunk 3 : 125 voxels
#> chunk 4 : 125 voxels
```

Each chunk element exposes `$data` (timepoints x voxels) and
`$voxel_ind` (the column indices this chunk corresponds to).

## Group operations with fmri_group()

[`fmri_group()`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_group.md)
wraps a data-frame with one row per subject and a list column of
datasets. It is the entry point for group-level modelling and
participant-level metadata.

``` r

subject_table <- data.frame(
  sub = subject_ids,
  age = c(24, 28, 31, 25, 29),
  dataset = I(lapply(subject_ids, function(id) list(subject_list[[id]]))),
  stringsAsFactors = FALSE
)

grp <- fmri_group(subject_table, id = "sub", dataset_col = "dataset")
print(grp)
#> <fmri_group>
#>   subjects       : 5
#>   id column      : sub
#>   dataset column : dataset
#>   mask strategy  : subject_specific
#>   subject attrs  : sub, age
```

With `grp` in hand you can attach demographic covariates, filter
subjects with
[`filter_subjects()`](https://bbuchsbaum.github.io/fmridataset/reference/filter_subjects.md),
and pass the group object to modelling functions in downstream packages
such as **fmrireg**.

## Next steps

- [`vignette("bids-h5-archive")`](https://bbuchsbaum.github.io/fmridataset/articles/bids-h5-archive.md)
  — build a compressed, chunked HDF5 archive for an entire BIDS study,
  then read it back through the same interface shown here.
- [`?fmri_study_dataset`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_study_dataset.md)
  — full argument reference and constructor options.
- [`?data_chunks`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md)
  — chunk size strategies, run-wise vs. voxel-wise iteration.
- [`?fmri_group`](https://bbuchsbaum.github.io/fmridataset/reference/fmri_group.md)
  — group-level metadata and subject filtering.
