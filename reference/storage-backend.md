# Storage Backend S3 Contract

Defines the S3 generic functions that all storage backends must
implement. This provides a pluggable architecture for different data
storage formats.

## Details

A storage backend is responsible for:

- Managing stateful resources (file handles, connections)

- Providing dimension information

- Reading data in canonical timepoints × voxels orientation

- Providing mask information

- Extracting metadata
