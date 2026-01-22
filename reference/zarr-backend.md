# Zarr Storage Backend

A storage backend implementation for Zarr array format using the Rarr
package. Zarr is a cloud-native array storage format that supports
chunked, compressed n-dimensional arrays with concurrent read/write
access.

## Details

This backend provides efficient access to neuroimaging data stored in
Zarr format, which is particularly well-suited for:

- Large datasets that don't fit in memory

- Cloud storage (S3, GCS, Azure)

- Parallel processing workflows

- Progressive data access patterns

The backend expects Zarr arrays organized as:

- 4D array with dimensions (x, y, z, time)

- Optional mask array at "mask" key

- Metadata stored as Zarr attributes
