# H5 Storage Backend

A storage backend implementation for H5 format neuroimaging data using
fmristore. Each scan is stored as an H5 file that loads to an H5NeuroVec
object.

## Details

The H5Backend integrates with the fmristore package to work with:

- File paths to H5 neuroimaging files

- Pre-loaded H5NeuroVec objects from fmristore

- Multiple H5 files representing different scans
