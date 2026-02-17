"""Tests for BackendRegistry."""

import numpy as np
import pytest

from fmridataset import BackendRegistry, ConfigError, MatrixBackend


class TestRegistrySingleton:
    def test_singleton(self) -> None:
        r1 = BackendRegistry.instance()
        r2 = BackendRegistry.instance()
        assert r1 is r2

    def test_matrix_registered(self) -> None:
        reg = BackendRegistry.instance()
        assert reg.is_registered("matrix")


class TestRegisterAndCreate:
    def test_register_custom(self) -> None:
        reg = BackendRegistry.instance()
        reg.register(
            "test_custom",
            lambda **kw: MatrixBackend(data_matrix=np.zeros((5, 10)), **kw),
            description="test backend",
            overwrite=True,
        )
        assert reg.is_registered("test_custom")
        backend = reg.create("test_custom")
        assert backend.get_dims().time == 5
        reg.unregister("test_custom")

    def test_duplicate_raises(self) -> None:
        reg = BackendRegistry.instance()
        reg.register("dup_test", lambda **kw: None, overwrite=True)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="already registered"):
            reg.register("dup_test", lambda **kw: None)  # type: ignore[arg-type]
        reg.unregister("dup_test")

    def test_create_unregistered(self) -> None:
        reg = BackendRegistry.instance()
        with pytest.raises(ConfigError, match="not registered"):
            reg.create("nonexistent_backend_xyz")


class TestListAndUnregister:
    def test_list_names(self) -> None:
        reg = BackendRegistry.instance()
        names = reg.list_names()
        assert "matrix" in names
        assert isinstance(names, list)

    def test_unregister(self) -> None:
        reg = BackendRegistry.instance()
        reg.register("to_remove", lambda **kw: None, overwrite=True)  # type: ignore[arg-type]
        assert reg.unregister("to_remove") is True
        assert not reg.is_registered("to_remove")

    def test_unregister_nonexistent(self) -> None:
        reg = BackendRegistry.instance()
        assert reg.unregister("does_not_exist") is False

    def test_get_info(self) -> None:
        reg = BackendRegistry.instance()
        info = reg.get_info("matrix")
        assert info["name"] == "matrix"
        assert "description" in info
