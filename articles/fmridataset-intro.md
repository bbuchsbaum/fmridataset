# Getting Started with fmridataset

## Motivation

fMRI analyses involve heterogeneous sources: NIfTI files, BIDS datasets,
preprocessed matrices, and HDF5 archives. fmridataset provides a unified
interface that abstracts format differences so the same functions work
regardless of how data is stored.

## Creating a Dataset

The simplest starting point is a matrix dataset, which wraps an
in-memory matrix with temporal metadata.

``` r

set.seed(42)
mat <- matrix(rnorm(150 * 1000), nrow = 150, ncol = 1000)

ds <- matrix_dataset(
  datamat = mat,
  TR = 2.0,
  run_length = c(75, 75)
)

print(ds)
#> 
#> === fMRI Dataset ===
#> 
#> ** Dimensions:
#>   - Timepoints: 150 
#>   - Runs: 2  
#>   - Matrix: 150 x 1000 (timepoints x voxels)
#>   - Voxels in mask: (lazy)
#> 
#> ** Temporal Structure:
#>   - TR: 2 seconds
#>   - Run lengths: 75, 75 
#> 
#> ** Event Table:
#>   - Empty event table
```

The dataset holds 150 timepoints split into two runs of 75, with a
2-second TR.

## Accessing Data

[`get_data_matrix()`](https://bbuchsbaum.github.io/fmridataset/reference/get_data_matrix.md)
returns a standard timepoints-by-voxels matrix. You can retrieve all
runs or a single run by index.

``` r

all_data <- get_data_matrix(ds)
cat("Full dimensions:", dim(all_data), "\n")
#> Full dimensions: 150 1000

run1 <- get_data_matrix(ds, run_id = 1)
cat("Run 1 dimensions:", dim(run1), "\n")
#> Run 1 dimensions: 150 1000
```

The returned matrix is always in timepoints x voxels orientation.

## Temporal Structure

Every dataset contains a `sampling_frame` that records run boundaries,
TR, and duration.

``` r

sf <- ds$sampling_frame
cat("TR:", get_TR(sf), "seconds\n")
#> TR: 2 seconds
cat("Runs:", n_runs(sf), "\n")
#> Runs: 2
cat("Run lengths:", get_run_lengths(sf), "timepoints\n")
#> Run lengths: 75 75 timepoints
cat("Total duration:", get_total_duration(sf), "seconds\n")
#> Total duration: 300 seconds

# First six acquisition times (seconds)
head(samples(sf))
#> [1] 1 2 3 4 5 6
```

[`blockids()`](https://bbuchsbaum.github.io/fmridataset/reference/blockids.md)
maps each timepoint to its run index, which is useful for run-specific
indexing.

## Event Tables

Experimental design attaches to the dataset as a data frame in
`$event_table`.

``` r

events <- data.frame(
  onset = c(10, 30, 50, 70, 110, 130, 150, 170),
  duration = 2,
  trial_type = rep(c("faces", "houses"), 4),
  run = c(1, 1, 1, 1, 2, 2, 2, 2)
)

ds$event_table <- events
head(ds$event_table)
#>   onset duration trial_type run
#> 1    10        2      faces   1
#> 2    30        2     houses   1
#> 3    50        2      faces   1
#> 4    70        2     houses   1
#> 5   110        2      faces   2
#> 6   130        2     houses   2
```

Event onsets are in seconds and align with the sampling frame’s time
axis.

## Chunked Processing

For large datasets,
[`data_chunks()`](https://bbuchsbaum.github.io/fmridataset/reference/data_chunks.md)
partitions voxels into memory-manageable pieces without loading the
entire matrix at once.

``` r

chunks <- data_chunks(ds, nchunks = 4)

results <- lapply(chunks, function(chunk) {
  colMeans(chunk$data)
})

voxel_means <- do.call(c, results)
cat("Computed means for", length(voxel_means), "voxels\n")
#> Computed means for 1000 voxels
```

Each element of `results` corresponds to one chunk; `do.call(c, ...)`
reassembles them in voxel order.

## Run-wise Processing

Set `runwise = TRUE` to get one chunk per run. This is appropriate for
analyses that must respect run boundaries such as detrending or temporal
filtering.

``` r

run_chunks <- data_chunks(ds, runwise = TRUE)

run_stats <- lapply(run_chunks, function(chunk) {
  cat("Run", chunk$chunk_num, ":", nrow(chunk$data), "timepoints\n")
  rowMeans(chunk$data)
})
#> Run 1 : 75 timepoints
#> Run 2 : 75 timepoints

cat("Processed", length(run_stats), "runs\n")
#> Processed 2 runs
```

Each chunk’s `$data` contains only that run’s timepoints.

## File-based Datasets

When working with NIfTI files, use `fmri_file_dataset()`. Data remains
on disk until explicitly accessed.

``` r

ds_files <- fmri_file_dataset(
  scans = c("/path/to/run1.nii.gz", "/path/to/run2.nii.gz"),
  mask = "/path/to/mask.nii.gz",
  TR = 2.0,
  run_length = c(180, 180)
)

# Inspect metadata without loading data
print(ds_files)

# Load one run only
run1 <- get_data_matrix(ds_files, run_id = 1)
```

Use `run_id` to load only the run you need, which keeps peak memory
usage low.

## See Also

- [`vignette("architecture-overview")`](https://bbuchsbaum.github.io/fmridataset/articles/architecture-overview.md) -
  Design principles and backend extensibility
- `vignette("h5-backend-usage")` - Efficient HDF5 storage for large
  datasets
- [`vignette("study-level-analysis")`](https://bbuchsbaum.github.io/fmridataset/articles/study-level-analysis.md) -
  Multi-subject studies and group analyses

## Session Information

``` r

sessionInfo()
#> R version 4.6.1 (2026-06-24)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] fmridataset_0.10.0.9000
#> 
#> loaded via a namespace (and not attached):
#>  [1] gtable_0.3.6        xfun_0.60           bslib_0.12.0       
#>  [4] delarr_0.1.0.9000   ggplot2_4.0.3       htmlwidgets_1.6.4  
#>  [7] lattice_0.22-9      bigassertr_0.2.0    numDeriv_2016.8-1.1
#> [10] vctrs_0.7.3         tools_4.6.1         generics_0.1.4     
#> [13] parallel_4.6.1      tibble_3.3.1        pkgconfig_2.0.3    
#> [16] Matrix_1.7-5        RColorBrewer_1.1-3  bigstatsr_1.6.2    
#> [19] S7_0.2.2            desc_1.4.3          RcppParallel_6.2.0 
#> [22] assertthat_0.2.1    lifecycle_1.0.5     stringr_1.6.0      
#> [25] compiler_4.6.1      neuroim2_0.19.0     farver_2.1.2       
#> [28] textshaping_1.0.5   RNifti_1.9.0        bigparallelr_0.3.2 
#> [31] codetools_0.2-20    htmltools_0.5.9     sass_0.4.10        
#> [34] yaml_2.3.12         deflist_0.2.0       pillar_1.11.1      
#> [37] pkgdown_2.2.1       jquerylib_0.1.4     RNiftyReg_2.8.5    
#> [40] cachem_1.1.0        dbscan_1.2.5        iterators_1.0.14   
#> [43] foreach_1.5.2       tidyselect_1.2.1    digest_0.6.39      
#> [46] stringi_1.8.9       dplyr_1.2.1         purrr_1.2.2        
#> [49] splines_4.6.1       cowplot_1.2.0       fastmap_1.2.0      
#> [52] grid_4.6.1          mmap_0.6-26         cli_3.6.6          
#> [55] magrittr_2.0.5      fmrihrf_0.3.0       scales_1.4.0       
#> [58] rmarkdown_2.31      albersdown_2.0.0    rmio_0.4.0         
#> [61] otel_0.2.0          ragg_1.5.2          memoise_2.0.1      
#> [64] evaluate_1.0.5      knitr_1.51          doParallel_1.0.17  
#> [67] rlang_1.3.0         Rcpp_1.1.2          glue_1.8.1         
#> [70] jsonlite_2.0.0      R6_2.6.1            systemfonts_1.3.2  
#> [73] fs_2.1.0            flock_0.7
```
