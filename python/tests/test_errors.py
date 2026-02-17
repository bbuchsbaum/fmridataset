"""Tests for the error hierarchy."""

from fmridataset import BackendIOError, ConfigError, FmriDatasetError


def test_hierarchy() -> None:
    assert issubclass(BackendIOError, FmriDatasetError)
    assert issubclass(ConfigError, FmriDatasetError)
    assert issubclass(FmriDatasetError, Exception)


def test_backend_io_error_attrs() -> None:
    err = BackendIOError("read failed", file="/tmp/x.nii", operation="read")
    assert str(err) == "read failed"
    assert err.file == "/tmp/x.nii"
    assert err.operation == "read"


def test_config_error_attrs() -> None:
    err = ConfigError("bad param", parameter="TR", value=-1)
    assert str(err) == "bad param"
    assert err.parameter == "TR"
    assert err.value == -1


def test_catch_as_base() -> None:
    try:
        raise BackendIOError("oops")
    except FmriDatasetError:
        pass  # expected
