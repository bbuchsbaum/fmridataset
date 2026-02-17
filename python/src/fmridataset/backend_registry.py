"""Backend registry singleton.

Port of ``R/backend_registry.R``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .backend_protocol import StorageBackend
from .errors import ConfigError


@dataclass
class _Registration:
    name: str
    factory: Callable[..., StorageBackend]
    description: str
    validate_function: Callable[[StorageBackend], bool] | None
    registered_at: datetime = field(default_factory=datetime.now)


class BackendRegistry:
    """Singleton registry for storage backend factories."""

    _instance: BackendRegistry | None = None
    _entries: dict[str, _Registration]

    def __init__(self) -> None:
        self._entries = {}

    @classmethod
    def instance(cls) -> BackendRegistry:
        """Return the global singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- public API --------------------------------------------------------

    def register(
        self,
        name: str,
        factory: Callable[..., StorageBackend],
        *,
        description: str | None = None,
        validate_function: Callable[[StorageBackend], bool] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Register a backend factory.

        Parameters
        ----------
        name : str
            Unique identifier for the backend type.
        factory : callable
            Callable that returns a :class:`StorageBackend` instance.
        description : str or None
            Human-readable description.
        validate_function : callable or None
            Optional extra validation beyond the standard contract.
        overwrite : bool
            If *False* (default), raise on duplicate name.
        """
        if not name:
            raise ConfigError("name must be a non-empty string", parameter="name")

        if name in self._entries and not overwrite:
            raise ConfigError(
                f"Backend '{name}' is already registered. "
                "Use overwrite=True to replace.",
                parameter="name",
            )

        self._entries[name] = _Registration(
            name=name,
            factory=factory,
            description=description or f"Backend: {name}",
            validate_function=validate_function,
        )

    def create(
        self,
        name: str,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> StorageBackend:
        """Create a backend instance by registered name.

        Parameters
        ----------
        name : str
            Registered backend name.
        validate : bool
            Run the standard + custom validation after creation.
        **kwargs
            Passed to the backend factory.
        """
        if name not in self._entries:
            raise ConfigError(
                f"Backend '{name}' is not registered. "
                f"Available: {', '.join(self.list_names())}",
                parameter="name",
            )

        reg = self._entries[name]

        try:
            backend = reg.factory(**kwargs)
        except Exception as exc:
            raise ConfigError(
                f"Failed to create backend '{name}': {exc}"
            ) from exc

        if validate:
            backend.validate()
            if reg.validate_function is not None:
                reg.validate_function(backend)

        return backend

    def list_names(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._entries)

    def is_registered(self, name: str) -> bool:
        return name in self._entries

    def unregister(self, name: str) -> bool:
        """Remove a backend.  Returns True if it existed."""
        return self._entries.pop(name, None) is not None

    def get_info(self, name: str) -> dict[str, Any]:
        """Return registration metadata for *name*."""
        if name not in self._entries:
            raise ConfigError(f"Backend '{name}' is not registered")
        reg = self._entries[name]
        return {
            "name": reg.name,
            "description": reg.description,
            "has_validate": reg.validate_function is not None,
            "registered_at": reg.registered_at,
        }


def _register_builtins() -> None:
    """Register the built-in backends.  Called from ``__init__``."""
    from .backends.matrix_backend import MatrixBackend

    registry = BackendRegistry.instance()
    registry.register(
        "matrix",
        lambda **kw: MatrixBackend(**kw),
        description="In-memory matrix backend",
        overwrite=True,
    )

    # Optional-dep backends: register factories that do lazy imports
    def _nifti_factory(**kw: Any) -> StorageBackend:
        from .backends.nifti_backend import NiftiBackend
        return NiftiBackend(**kw)

    registry.register(
        "nifti",
        _nifti_factory,
        description="NIfTI file backend (requires nibabel)",
        overwrite=True,
    )

    def _h5_factory(**kw: Any) -> StorageBackend:
        from .backends.h5_backend import H5Backend
        return H5Backend(**kw)

    registry.register(
        "h5",
        _h5_factory,
        description="HDF5 storage backend (requires h5py)",
        overwrite=True,
    )

    def _zarr_factory(**kw: Any) -> StorageBackend:
        from .backends.zarr_backend import ZarrBackend
        return ZarrBackend(**kw)

    registry.register(
        "zarr",
        _zarr_factory,
        description="Zarr array backend (requires zarr)",
        overwrite=True,
    )

    def _latent_factory(**kw: Any) -> StorageBackend:
        from .backends.latent_backend import LatentBackend
        return LatentBackend(**kw)

    registry.register(
        "latent",
        _latent_factory,
        description="Latent-decomposition HDF5 backend (requires h5py)",
        overwrite=True,
    )

    def _study_factory(**kw: Any) -> StorageBackend:
        from .backends.study_backend import StudyBackend
        return StudyBackend(**kw)

    registry.register(
        "study",
        _study_factory,
        description="Multi-subject composite backend",
        overwrite=True,
    )
