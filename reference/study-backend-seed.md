# Study Backend Seed for DelayedArray

A DelayedArray-compatible seed that provides lazy access to
multi-subject fMRI data without loading all subjects into memory at
once. Implemented as an S4 class inheriting from `Array` so it
integrates natively with DelayedArray.
