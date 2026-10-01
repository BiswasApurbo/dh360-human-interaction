"""Unit tests for the M0 foundation layer."""

import dh360_hi
from dh360_hi import backend, contracts, edge, vision


def test_package_version(package_version: str) -> None:
    """Verify package version is defined and matches semantic versioning format."""
    assert dh360_hi.__version__ == package_version
    assert isinstance(dh360_hi.__version__, str)
    assert len(dh360_hi.__version__.split(".")) >= 3


def test_submodules_exist() -> None:
    """Verify core submodules are importable."""
    assert vision is not None
    assert edge is not None
    assert backend is not None
    assert contracts is not None
