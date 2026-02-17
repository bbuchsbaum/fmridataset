"""Custom exception hierarchy for fmridataset.

Mirrors the R package's error classes:
- fmridataset_error        -> FmriDatasetError
- fmridataset_error_backend_io -> BackendIOError
- fmridataset_error_config     -> ConfigError
"""

from __future__ import annotations


class FmriDatasetError(Exception):
    """Base exception for all fmridataset errors."""


class BackendIOError(FmriDatasetError):
    """Raised when a storage backend encounters read/write failures.

    Parameters
    ----------
    message : str
        Description of the I/O error.
    file : str | None
        Path to the file that caused the error.
    operation : str | None
        The operation that failed (e.g., "read", "write").
    """

    def __init__(
        self,
        message: str,
        *,
        file: str | None = None,
        operation: str | None = None,
    ) -> None:
        self.file = file
        self.operation = operation
        super().__init__(message)


class ConfigError(FmriDatasetError):
    """Raised when invalid configuration is provided.

    Parameters
    ----------
    message : str
        Description of the configuration error.
    parameter : str | None
        The parameter that was invalid.
    value : object
        The invalid value provided.
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: object = None,
    ) -> None:
        self.parameter = parameter
        self.value = value
        super().__init__(message)
