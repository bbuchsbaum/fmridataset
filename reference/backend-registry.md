# Backend Registry System

A pluggable registry system for storage backends that allows external
packages to register new backend types without modifying the fmridataset
package. This enables extensibility while maintaining backward
compatibility.

## Details

The registry system manages backend factories that create backend
instances. Each backend must implement the storage backend contract
defined in
[`storage-backend`](https://bbuchsbaum.github.io/fmridataset/reference/storage-backend.md).
