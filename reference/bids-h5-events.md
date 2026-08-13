# BIDS H5 Event and Confound Helpers

Internal functions for reading and writing events, confounds, and censor
vectors stored as column arrays inside a BIDS HDF5 archive.

## Details

Events are stored as one HDF5 dataset per column (column-array layout),
which gives better performance than compound datasets for
variable-length strings. The number of events is stored as the attribute
`n_events` on the parent group so readers can allocate correctly without
probing lengths.

Confounds are stored as a single `[T, n_confounds]` float64 matrix
dataset with a `names` attribute listing column names.

Censor vectors are stored as `uint8` arrays of length T (0 = keep, 1 =
censor), matching the fmridataset convention.
