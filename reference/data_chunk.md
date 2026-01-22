# Create a Data Chunk Object

Creates a data chunk object that represents a subset of data from an
fMRI dataset. A data chunk contains the data matrix along with indices
indicating which voxels and time points (rows) are included in the
chunk.

## Usage

``` r
data_chunk(mat, voxel_ind, row_ind, chunk_num)
```

## Arguments

- mat:

  A matrix containing the chunk data (rows = time points, columns =
  voxels)

- voxel_ind:

  Integer vector of voxel indices included in this chunk

- row_ind:

  Integer vector of row (time point) indices included in this chunk

- chunk_num:

  Integer indicating the chunk number

## Value

A data_chunk object containing:

- data:

  The data matrix for this chunk

- voxel_ind:

  Indices of voxels in this chunk

- row_ind:

  Indices of rows (time points) in this chunk

- chunk_num:

  The chunk number

## Examples

``` r
# Create a simple data chunk
mat <- matrix(rnorm(100), nrow = 10, ncol = 10)
chunk <- data_chunk(mat, voxel_ind = 1:10, row_ind = 1:10, chunk_num = 1)
print(chunk)
#> Data Chunk Object
#>   chunk 1 of 1
#>   Number of voxels: 10 
#>   Number of rows: 10 
#>   Data dimensions: 10 x 10 
```
