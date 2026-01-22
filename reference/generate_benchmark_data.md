# Generate Performance Benchmark Data

Generate Performance Benchmark Data

## Usage

``` r
generate_benchmark_data(
  dataset_sizes = c(100, 500, 1000, 5000),
  operations = c("load", "chunk", "process")
)
```

## Arguments

- dataset_sizes:

  Vector of dataset sizes to test

- operations:

  Operations to benchmark

## Value

Data frame with benchmark results
